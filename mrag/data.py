"""A synthetic multi-aspect retrieval benchmark.

The corpus is a set of short documents. Each belongs to one category and is written
with a few tokens specific to the document, a few tokens drawn from its category's
vocabulary, and some filler. A query about one aspect is a short mention of one
document: two of its specific tokens, two category tokens and a filler token. A
multi-aspect query concatenates mentions of n documents from n different categories,
so a retriever has to return n documents that have little in common. This mirrors the
paper's evaluation, where a query names n documents of n distinct categories and
success is scored on exact documents and on categories.
"""

from dataclasses import dataclass

import numpy as np

PAD, EMB = 0, 1


@dataclass
class WorldConfig:
    n_cats: int = 20
    docs_per_cat: int = 30
    cat_pool: int = 24
    entity_pool: int = 500
    fillers: int = 80
    ent_per_doc: int = 4
    cat_per_doc: int = 4
    filler_per_doc: int = 6


class World:
    def __init__(self, seed=0, cfg=None):
        cfg = cfg or WorldConfig()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.filler0 = 2
        self.cat0 = self.filler0 + cfg.fillers
        self.ent0 = self.cat0 + cfg.n_cats * cfg.cat_pool
        self.vocab = self.ent0 + cfg.entity_pool

        docs, cats = [], []
        for c in range(cfg.n_cats):
            for _ in range(cfg.docs_per_cat):
                ent = self.ent0 + rng.choice(cfg.entity_pool, cfg.ent_per_doc, replace=False)
                cat = self.cat0 + c * cfg.cat_pool + rng.choice(cfg.cat_pool, cfg.cat_per_doc, replace=False)
                fill = self.filler0 + rng.choice(cfg.fillers, cfg.filler_per_doc, replace=False)
                tokens = rng.permutation(np.concatenate([ent, cat, fill]))
                docs.append(np.append(tokens, EMB))
                cats.append(c)
        self.docs = np.stack(docs).astype(np.int32)
        self.doc_cat = np.array(cats)
        self.doc_len = np.full(len(docs), self.docs.shape[1], dtype=np.int32)
        self.doc_entities = self.docs[np.isin(self.docs, np.arange(self.ent0, self.vocab))].reshape(len(docs), -1)

    @property
    def n_docs(self):
        return len(self.docs)

    def mention(self, doc, rng):
        cfg = self.cfg
        ent = rng.choice(self.doc_entities[doc], 2, replace=False)
        c = self.doc_cat[doc]
        cat = self.cat0 + c * cfg.cat_pool + rng.choice(cfg.cat_pool, 2, replace=False)
        fill = self.filler0 + rng.choice(cfg.fillers, 1)
        return rng.permutation(np.concatenate([ent, cat, fill]))

    def query(self, doc_ids, rng):
        parts = [self.mention(d, rng) for d in doc_ids]
        return np.append(np.concatenate(parts), EMB).astype(np.int32)

    def sample_queries(self, n_aspects, count, rng):
        """Return padded tokens (count, L), lengths (count,), relevant doc ids (count, n)."""
        if n_aspects > self.cfg.n_cats:
            raise ValueError("a query cannot have more aspects than there are categories")
        queries, relevant = [], []
        for _ in range(count):
            cats = rng.choice(self.cfg.n_cats, n_aspects, replace=False)
            docs = [rng.choice(np.nonzero(self.doc_cat == c)[0]) for c in cats]
            queries.append(self.query(docs, rng))
            relevant.append(docs)
        tokens, lengths = pad(queries)
        return tokens, lengths, np.array(relevant)

    def training_batch(self, batch, rng):
        """Single-aspect contrastive pairs: a mention of one document, and that document.

        Like the embedding models in the paper, the encoder never sees a multi-aspect
        query during training.
        """
        docs = rng.choice(self.n_docs, batch, replace=False)
        tokens, lengths = pad([self.query([d], rng) for d in docs])
        return tokens, lengths, docs


def pad(seqs):
    length = max(len(s) for s in seqs)
    out = np.full((len(seqs), length), PAD, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, np.array([len(s) for s in seqs], dtype=np.int32)
