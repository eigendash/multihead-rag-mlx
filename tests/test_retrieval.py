import numpy as np
import pytest

from mrag.retrieval import MultiSpaceIndex, head_importance, split_spaces, unit


def test_importance_is_zero_for_a_collapsed_space_and_scales_with_norm():
    rng = np.random.default_rng(0)
    diverse = rng.normal(size=(200, 1, 8))
    collapsed = np.ones((200, 1, 8))
    scores = head_importance(np.concatenate([diverse, collapsed], axis=1))
    assert scores[1] == pytest.approx(0.0, abs=1e-6)
    assert scores[0] > 1.0
    scaled = head_importance(np.concatenate([2 * diverse, collapsed], axis=1))
    assert scaled[0] == pytest.approx(2 * scores[0], rel=1e-9)


def test_importance_prefers_spread_over_clustered_vectors_of_equal_norm():
    rng = np.random.default_rng(1)
    spread = unit(rng.normal(size=(300, 8)))
    base = unit(rng.normal(size=8))
    clustered = unit(base + 0.05 * rng.normal(size=(300, 8)))
    s = head_importance(np.stack([spread, clustered], axis=1))
    assert s[0] > 3 * s[1]


def test_single_space_is_plain_cosine_ranking():
    rng = np.random.default_rng(2)
    docs = rng.normal(size=(40, 1, 6))
    query = rng.normal(size=(5, 1, 6))
    got = MultiSpaceIndex(docs).search(query, k=7)
    expected = np.argsort(-(unit(query[:, 0]) @ unit(docs[:, 0]).T), axis=1)[:, :7]
    np.testing.assert_array_equal(got, expected)


def make_two_space_index(weights):
    # Space 0 ranks the documents 0, 1, 2, 3; space 1 ranks them 2, 1, 0, 3 (ties in
    # cosine are broken by index). Document 1 is the runner-up in both spaces.
    e = np.eye(4)
    docs = np.zeros((4, 2, 4))
    docs[0, 0], docs[1, 0], docs[2, 0], docs[3, 0] = e[0], 0.6 * e[0] + 0.8 * e[1], e[2], e[3]
    docs[2, 1], docs[1, 1], docs[0, 1], docs[3, 1] = e[2], 0.6 * e[2] + 0.8 * e[1], e[0], e[3]
    query = np.zeros((1, 2, 4))
    query[0, 0], query[0, 1] = e[0], e[2]
    return MultiSpaceIndex(docs, weights=weights), query


def test_voting_sums_rank_discounted_scores_by_hand():
    # Votes with equal weights, rank p earning 2^-p:
    #   doc 0: rank 0 in space 0, rank 2 in space 1 -> 1 + 0.25 = 1.25
    #   doc 2: rank 2 in space 0, rank 0 in space 1 -> 0.25 + 1 = 1.25
    #   doc 1: rank 1 in both                        -> 0.5 + 0.5 = 1.0
    #   doc 3: rank 3 in both                        -> 0.125 + 0.125 = 0.25
    index, query = make_two_space_index([1.0, 1.0])
    order = index.search(query, k=4)[0].tolist()
    assert set(order[:2]) == {0, 2} and order[2:] == [1, 3]


def test_importance_weights_decide_between_spaces():
    heavy0, query = make_two_space_index([10.0, 1.0])
    heavy1, _ = make_two_space_index([1.0, 10.0])
    assert heavy0.search(query, k=1)[0, 0] == 0
    assert heavy1.search(query, k=1)[0, 0] == 2


def test_depth_limits_which_documents_can_receive_votes():
    index, query = make_two_space_index([1.0, 1.0])
    shallow = MultiSpaceIndex(index.docs, weights=index.weights, depth=1)
    out = shallow.search(query, k=4)[0].tolist()
    assert set(out[:2]) == {0, 2}


def test_split_spaces_cuts_a_vector_into_contiguous_chunks():
    x = np.arange(12.0).reshape(1, 12)
    parts = split_spaces(x, 3)
    assert parts.shape == (1, 3, 4)
    np.testing.assert_array_equal(parts[0, 1], [4, 5, 6, 7])
