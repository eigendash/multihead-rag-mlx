# multihead-rag-mlx

A small-scale implementation of the retrieval scheme from "Multi-Head RAG: Solving Multi-Aspect Problems with LLMs" (Besta et al., arXiv:2406.05085), written for MLX on Apple silicon.

This is an independent implementation built from the paper's description. It is not a reproduction of the paper's results. The paper works with 7B-parameter pretrained embedding models and Wikipedia, legal and accident-report data; here everything is a few hundred thousand parameters trained from scratch on synthetic text, and the conclusions of the paper should not be read off these numbers. In this setup the multi-head method did not beat the ordinary single-vector baseline, and the results section says so plainly.

## The idea

A retrieval-augmented system normally embeds a document as one vector, usually the final hidden state of a transformer at the last token. A query that asks about several unrelated things has to land near all of the relevant documents with a single point in that space, which is hard when the documents are far apart. The paper proposes to read the embedding one step earlier. In the last attention layer, before the output projection merges the heads, each head produces its own vector for the last token. Keeping those vectors separate gives h embeddings per document, each of size d/h, at no extra cost since they come from the same forward pass. The hope is that different heads attend to different aspects of the text, so that a query with several aspects can be matched aspect by aspect.

At query time each head's space is searched on its own, the top candidates from every space are pooled, and a rank-based vote picks the final list. Two details from the paper are implemented as described. Each head gets an importance score s = a * b, where a is the mean L2 norm of that head's stored vectors and b is the mean cosine distance between random pairs of them, so that heads with weak activations or collapsed outputs count for less. A document at rank p (counting from zero) in head i's list receives a vote of s_i * 2^(-p), and votes are summed over heads.

## What is here

`mrag/model.py` is a small causal transformer (3 layers, width 128, 8 heads) whose `encode` returns two things for the last real token of each row: the final hidden state, and the last block's per-head attention outputs. A test checks that the head vectors are exactly the inputs of the output projection.

`mrag/retrieval.py` holds the importance scores and the multi-space index with voting. With a single space the index reduces to cosine nearest-neighbour search, which is also tested.

`mrag/data.py` generates the benchmark. Documents belong to one of 20 categories and consist of four document-specific tokens, four tokens from the category vocabulary and some filler. A one-aspect query is a short mention of a document (two of its specific tokens, two category tokens, one filler). A query with n aspects concatenates mentions of n documents from n distinct categories, which follows the structure of the paper's evaluation. `mrag/metrics.py` implements the exact-match success ratio, a category-level ratio that also credits a retrieved document from the right category, and their weighted combination.

`scripts/run_experiment.py` trains the encoder and runs the comparison. Retrieval itself is done in NumPy, since the corpus is only 600 documents; the encoder and the head extraction are in MLX.

## Experiment

For each of three seeds a new corpus is generated and a new encoder is trained for 3000 steps with an InfoNCE loss, using the final hidden state, on single-aspect pairs only (a mention and its document, in-batch negatives). This mirrors the paper's setting, where the embedding model is trained for ordinary single-vector retrieval and the heads are used afterwards without further training. Four retrievers then share the same trained encoder.

The standard retriever uses the final hidden state as one vector. The split retriever cuts that same vector into eight chunks and treats each as a space, with the same importance scores and voting; this is the paper's control for showing that any gain comes from the heads and not merely from using several spaces. The multi-head retriever uses the eight per-head vectors. A fourth variant gives every head the same weight, to see whether the importance scores matter.

Each setting uses 300 queries with 1, 2, 4, 8 or 12 aspects, and the retrievers return k documents for k equal to n, 2n and 3n.

## Results

Numbers are means over three seeds with the standard deviation in brackets. Exact match is the fraction of the n named documents that appear in the top 2n.

| aspects | standard | split | multi-head | multi-head, uniform |
|---:|---:|---:|---:|---:|
| 1 | 1.000 (0.000) | 1.000 (0.000) | 0.991 (0.004) | 0.991 (0.002) |
| 2 | 0.714 (0.036) | 0.531 (0.030) | 0.496 (0.007) | 0.501 (0.010) |
| 4 | 0.353 (0.009) | 0.214 (0.003) | 0.255 (0.011) | 0.258 (0.010) |
| 8 | 0.192 (0.026) | 0.126 (0.017) | 0.152 (0.009) | 0.155 (0.009) |
| 12 | 0.146 (0.006) | 0.105 (0.002) | 0.125 (0.004) | 0.126 (0.003) |

Category match in the top 2n, which is the lenient measure:

| aspects | standard | split | multi-head | multi-head, uniform |
|---:|---:|---:|---:|---:|
| 1 | 1.000 (0.000) | 1.000 (0.000) | 0.998 (0.002) | 0.999 (0.002) |
| 2 | 0.838 (0.027) | 0.739 (0.043) | 0.629 (0.018) | 0.630 (0.024) |
| 4 | 0.625 (0.003) | 0.611 (0.012) | 0.552 (0.003) | 0.551 (0.004) |
| 8 | 0.559 (0.009) | 0.598 (0.004) | 0.547 (0.017) | 0.543 (0.017) |
| 12 | 0.594 (0.032) | 0.661 (0.026) | 0.582 (0.014) | 0.579 (0.013) |

Further tables, including k = n and the weighted score with w = 2, are in `results/results.md`, and the per-seed numbers and head importance scores are in `results/results.json`.

Three things stand out. First, the standard single-vector retriever is the best or close to it on exact match at every aspect count, and clearly ahead for two aspects. The paper reports the opposite on its data, so the claim that multi-head embeddings help did not carry over to this toy. Second, the multi-head retriever does beat the split control on exact match from four aspects upward (for example 0.255 against 0.214 at four aspects, and 0.152 against 0.126 at eight), which is consistent with the paper's point that head outputs are more useful than an arbitrary partition of the final vector. On category match the split control is ahead at eight and twelve aspects, so even this comparison is not one-sided. Third, the importance scores are almost flat, between roughly 1.2 and 1.7 across heads, and the uniform-weight variant is statistically indistinguishable from the weighted one. With these heads the scoring step does nothing measurable.

On single-aspect queries the multi-head retriever is slightly below the baseline at every cutoff: 0.976, 0.991 and 0.996 exact when one, two or three documents are returned, against 0.999 and 1.000 for the standard retriever. The paper reports parity here, so this is a small loss in this setting.

## Why this might differ from the paper

These are guesses, not findings, and I did not test them. The encoder here is trained with a loss on the final vector over a very clean synthetic task, so that vector becomes a strong key by itself, and nothing in training pushes individual heads to specialise by category. The corpus has only 600 documents and 20 categories, and queries are bags of tokens, not text. The paper's heads belong to models pretrained on large corpora, where head specialisation has been observed. A fair test of the method needs a pretrained embedding model with a real attention-head structure, which this repository does not attempt.

## Choices the paper leaves open

The voting description says candidates are sorted globally by their scores; I sum the votes of a document that appears in several heads' lists. Ranks start at zero. Importance scores use 4000 random pairs of stored vectors. The category-level score credits each named document whose category appears anywhere in the retrieved set. Candidates are taken from the full ranking of every space, so the per-space candidate count is not a limit here. The query is not split into per-aspect sub-queries in any retriever.

## Limitations

The data is synthetic and the model is tiny and trained from scratch. Only the retrieval step is evaluated; there is no generation step and no comparison with the paper's other baselines such as fusion-style query expansion. The experiment uses three seeds, so small differences, for instance among the multi-head variants, should not be over-read. No timing or throughput claims are made.

## Running it

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
    python scripts/run_experiment.py

`--seeds` and `--steps` shorten a run, for example `--seeds 0 --steps 500` as a quick check. Training one seed takes a few minutes on an M-series laptop.

## Reference

Maciej Besta et al. "Multi-Head RAG: Solving Multi-Aspect Problems with LLMs." arXiv:2406.05085, 2024.
