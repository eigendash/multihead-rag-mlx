"""Multi-space retrieval with head importance scores and rank voting.

A document is stored as S vectors, one per embedding space (for multi-head retrieval
a space is an attention head). A query is searched in every space separately, and the
per-space ranked lists are merged by voting:

    importance of space i:  s_i = a_i * b_i
        a_i  mean L2 norm of the stored vectors in space i
        b_i  mean cosine distance between random pairs of stored vectors in space i
    vote of the document at rank p (from 0) in space i:  s_i * 2^(-p)

A document's votes are summed over spaces and the top k are returned. With one space
this is plain cosine nearest-neighbour search.
"""

import numpy as np


def unit(x, eps=1e-9):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def head_importance(spaces, n_pairs=4000, seed=0):
    """spaces: (N, S, d). Returns s of shape (S,)."""
    rng = np.random.default_rng(seed)
    n = spaces.shape[0]
    first, second = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    norm = np.linalg.norm(spaces, axis=-1).mean(axis=0)
    cosine = (unit(spaces[first]) * unit(spaces[second])).sum(-1)
    return norm * (1.0 - cosine).mean(axis=0)


class MultiSpaceIndex:
    def __init__(self, doc_spaces, weights=None, depth=None):
        """doc_spaces: (N, S, d). weights: per-space importance; None means all ones.
        depth: candidates taken per space before voting (default: all documents)."""
        self.docs = unit(np.asarray(doc_spaces, dtype=np.float64))
        self.n_docs, self.n_spaces = self.docs.shape[:2]
        self.weights = np.ones(self.n_spaces) if weights is None else np.asarray(weights, dtype=np.float64)
        self.depth = depth

    def search(self, query_spaces, k):
        """query_spaces: (Q, S, d). Returns document ids (Q, k), best first."""
        q = unit(np.asarray(query_spaces, dtype=np.float64))
        depth = self.depth or self.n_docs
        votes = np.zeros((q.shape[0], self.n_docs))
        rows = np.arange(q.shape[0])[:, None]
        for s in range(self.n_spaces):
            sims = q[:, s] @ self.docs[:, s].T
            top = np.argsort(-sims, axis=1, kind="stable")[:, :depth]
            votes[rows, top] += self.weights[s] * 0.5 ** np.arange(top.shape[1])
        return np.argsort(-votes, axis=1, kind="stable")[:, :k]


def split_spaces(vectors, parts):
    """Cut (N, d) vectors into `parts` equal chunks, giving (N, parts, d // parts)."""
    n, d = vectors.shape
    return vectors.reshape(n, parts, d // parts)
