"""Compare single-vector retrieval with multi-head retrieval on the synthetic benchmark.

    python scripts/run_experiment.py                 # 3 seeds, writes results/
    python scripts/run_experiment.py --seeds 0 --steps 500   # quick check

For every seed a fresh corpus is drawn and a fresh encoder is trained with an
InfoNCE loss on single-aspect pairs only, using its final-layer embedding. The four
retrievers then share that one encoder:

    standard      one vector per document: the final hidden state at the last token
    split         the same vector cut into `heads` chunks, one space each, with voting
    mrag          the last attention layer's per-head vectors, importance-weighted voting
    mrag-uniform  as mrag, but every head votes with weight 1 (no importance scores)
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrag import (  # noqa: E402
    Encoder,
    EncoderConfig,
    MultiSpaceIndex,
    World,
    head_importance,
    split_spaces,
    success_ratios,
)

ASPECTS = [1, 2, 4, 8, 12]
METHODS = ["standard", "split", "mrag", "mrag-uniform"]
QUERIES_PER_SETTING = 300


def normalise(x):
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + 1e-9)


def train(world, seed, steps, batch, lr=1e-3, temperature=0.05):
    mx.random.seed(seed)
    model = Encoder(EncoderConfig(vocab=world.vocab))
    mx.eval(model.parameters())
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, lr, 100), optim.cosine_decay(lr, steps - 100)], [100]
    )
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    def loss_fn(model, q_tok, q_len, d_tok, d_len):
        q, _ = model.encode(q_tok, q_len)
        d, _ = model.encode(d_tok, d_len)
        logits = normalise(q) @ normalise(d).T / temperature
        labels = mx.arange(logits.shape[0])
        return 0.5 * (nn.losses.cross_entropy(logits, labels).mean() + nn.losses.cross_entropy(logits.T, labels).mean())

    step = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(seed + 1000)
    for i in range(1, steps + 1):
        q_tok, q_len, docs = world.training_batch(batch, rng)
        loss, grads = step(model, mx.array(q_tok), mx.array(q_len), mx.array(world.docs[docs]), mx.array(world.doc_len[docs]))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if i % 500 == 0 or i == steps:
            print(f"    step {i:5d}  loss {float(loss):.4f}", flush=True)
    return model


def embed(model, tokens, lengths, chunk=256):
    finals, heads = [], []
    for i in range(0, len(tokens), chunk):
        f, h = model.encode(mx.array(tokens[i : i + chunk]), mx.array(lengths[i : i + chunk]))
        mx.eval(f, h)
        finals.append(np.array(f))
        heads.append(np.array(h))
    return np.concatenate(finals), np.concatenate(heads)


def build_indices(doc_final, doc_heads, n_heads):
    split = split_spaces(doc_final, n_heads)
    return {
        "standard": MultiSpaceIndex(doc_final[:, None, :]),
        "split": MultiSpaceIndex(split, weights=head_importance(split)),
        "mrag": MultiSpaceIndex(doc_heads, weights=head_importance(doc_heads)),
        "mrag-uniform": MultiSpaceIndex(doc_heads),
    }


def query_spaces(method, final, heads, n_heads):
    if method == "standard":
        return final[:, None, :]
    if method == "split":
        return split_spaces(final, n_heads)
    return heads


def run_seed(seed, steps, batch):
    print(f"seed {seed}", flush=True)
    world = World(seed=seed)
    model = train(world, seed, steps, batch)
    n_heads = model.cfg.heads
    doc_final, doc_heads = embed(model, world.docs, world.doc_len)
    indices = build_indices(doc_final, doc_heads, n_heads)

    rng = np.random.default_rng(seed + 2000)
    results = {}
    for n in ASPECTS:
        tokens, lengths, relevant = world.sample_queries(n, QUERIES_PER_SETTING, rng)
        q_final, q_heads = embed(model, tokens, lengths)
        for method in METHODS:
            spaces = query_spaces(method, q_final, q_heads, n_heads)
            for mult in (1, 2, 3):
                got = indices[method].search(spaces, mult * n)
                results[f"{method}|n={n}|k={mult}n"] = success_ratios(got, relevant, world.doc_cat)
    diagnostics = {
        "head_importance": [float(x) for x in head_importance(doc_heads)],
        "split_importance": [float(x) for x in head_importance(split_spaces(doc_final, n_heads))],
    }
    return results, diagnostics


def table(all_results, mult, field, title):
    names = ["exact", "category", "weighted"]
    col = names.index(field)
    lines = [f"{title} (k = {mult}n, mean over seeds, std in brackets)", "", "| aspects | " + " | ".join(METHODS) + " |", "|---:|" + "---:|" * len(METHODS)]
    for n in ASPECTS:
        cells = []
        for method in METHODS:
            vals = [r[f"{method}|n={n}|k={mult}n"][col] for r in all_results]
            cells.append(f"{np.mean(vals):.3f} ({np.std(vals):.3f})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    runs = [run_seed(s, args.steps, args.batch) for s in args.seeds]
    all_results = [r for r, _ in runs]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"seeds": args.seeds, "steps": args.steps, "results": all_results, "diagnostics": [d for _, d in runs]}, indent=1)
    )
    sections = [
        table(all_results, 1, "exact", "Exact document match"),
        table(all_results, 2, "exact", "Exact document match"),
        table(all_results, 2, "weighted", "Weighted success, w = 2"),
        table(all_results, 2, "category", "Category match"),
    ]
    text = "\n\n".join(sections)
    (out / "results.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
