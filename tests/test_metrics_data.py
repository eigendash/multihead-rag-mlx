import numpy as np
import pytest

from mrag.data import EMB, World
from mrag.metrics import success_ratios


def test_success_ratios_on_a_worked_example():
    doc_cat = np.array([0, 0, 1, 1, 2, 2])
    relevant = np.array([[0, 2, 4]])
    retrieved = np.array([[0, 3, 5, 1]])  # exact: doc 0; categories hit: 0, 1, 2
    exact, category, weighted = success_ratios(retrieved, relevant, doc_cat, w=2.0)
    assert exact == pytest.approx(1 / 3)
    assert category == pytest.approx(1.0)
    assert weighted == pytest.approx((2 * (1 / 3) + 1.0) / 3)


def test_success_ratios_penalise_missing_categories():
    doc_cat = np.array([0, 0, 1, 1])
    exact, category, _ = success_ratios(np.array([[0, 1]]), np.array([[0, 2]]), doc_cat)
    assert exact == 0.5 and category == 0.5


def test_world_is_deterministic_and_documents_are_unique():
    a, b = World(seed=3), World(seed=3)
    np.testing.assert_array_equal(a.docs, b.docs)
    assert len({tuple(sorted(r[:-1])) for r in a.docs}) == a.n_docs
    assert (a.docs[:, -1] == EMB).all()


def test_multi_aspect_queries_use_distinct_categories_and_real_mentions():
    world = World(seed=0)
    rng = np.random.default_rng(0)
    tokens, lengths, relevant = world.sample_queries(5, 20, rng)
    assert tokens.shape == (20, 26) and (lengths == 26).all()
    for row, docs in zip(tokens, relevant):
        assert len(set(world.doc_cat[docs])) == 5
        for i, d in enumerate(docs):
            mention = row[5 * i : 5 * i + 5]
            ent = [t for t in mention if t >= world.ent0]
            assert len(ent) == 2 and set(ent) <= set(world.doc_entities[d].tolist())
            cat = [t for t in mention if world.cat0 <= t < world.ent0]
            lo = world.cat0 + world.doc_cat[d] * world.cfg.cat_pool
            assert len(cat) == 2 and all(lo <= t < lo + world.cfg.cat_pool for t in cat)


def test_too_many_aspects_is_rejected():
    with pytest.raises(ValueError):
        World().sample_queries(25, 1, np.random.default_rng(0))
