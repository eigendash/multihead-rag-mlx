"""Retrieval success ratios for multi-aspect queries.

For a query with relevant set R of n documents and retrieved set S:

    exact     |S & R| / n
    category  share of documents in R whose category is represented in S
    weighted  (w * exact + category) / (w + 1)

The category score is the lenient one: it credits a retrieved document from the right
category even when it is not the one the query named.
"""

import numpy as np


def success_ratios(retrieved, relevant, doc_cat, w=2.0):
    """retrieved: (Q, k) ids, relevant: (Q, n) ids. Returns mean (exact, category, weighted)."""
    exact, category = [], []
    for got, want in zip(retrieved, relevant):
        got_set = set(got.tolist())
        got_cats = set(doc_cat[got].tolist())
        exact.append(np.mean([d in got_set for d in want]))
        category.append(np.mean([doc_cat[d] in got_cats for d in want]))
    exact, category = float(np.mean(exact)), float(np.mean(category))
    return exact, category, (w * exact + category) / (w + 1)
