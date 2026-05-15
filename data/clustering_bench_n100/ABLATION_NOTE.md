# Clustering bench: N=11 vs N=161 gap corpus

The N=11 run was a plumbing smoke test (extraction bench's gold-section subset).
The N=161 run uses the full N=100-paper corpus produced by the new v2 PDF
extraction pipeline (style-aware blocks + expanded heading vocab + TOC filter).

## Headline shifts

| metric | N=11 best | N=161 best | what changed |
|---|---|---|---|
| silhouette (cosine) | 0.174 (bertopic × bge) | **0.100** (agglomerative × e5 / bge) | small-corpus tightness was inflated |
| NPMI | 0.916 (bertopic × mpnet) | **0.596** (agglomerative × MiniLM) | N=11 top-word overlap was a coincidence |
| bootstrap mean ARI | 0.869 (agglomerative × mpnet) | **0.356** (kmeans × bge) | small-corpus partitions were trivially stable |
| HDBSCAN clusters | 0/3/4 embedders | 0/4 embedders | still over-conservative `min_cluster_size` |

## Reading the N=161 results

- **Best (clusterer × embedder) by silhouette:** agglomerative × bge / e5 — ~0.10.
- **Best by NPMI:** agglomerative × MiniLM / bge / mpnet — ~0.59.
- **Best by stability (mean ARI):** kmeans × bge — 0.356.
- **Agglomerative beats kmeans on NPMI consistently** (~0.6 vs ~0.4) on the
  same embedder. Suggests agglomerative groups produce more semantically
  coherent top-word sets, even when geometric tightness is similar.
- **HDBSCAN and BERTopic mostly fail to find clusters** at this corpus size
  with our default `min_cluster_size` (auto-tuned to `min(5, n//4)`). 3 of
  4 BERTopic cells emitted 0 clusters; HDBSCAN 4 of 4. Both would benefit
  from corpus-aware retuning rather than the current scale-blind defaults.

## Takeaway

For the production theme-mining stage on a corpus this size:
- **Agglomerative × bge-small-en-v1.5** is the best-balanced choice
  (silhouette 0.099, NPMI 0.591, mean ARI 0.325 ± 0.306).
- KMeans is a fine fast-baseline (faster, comparable stability) but
  produces lower-coherence topics on average.
- Don't reach for HDBSCAN/BERTopic without fixing the cluster-size floor.

## What dimensionality reduction (UMAP) buys us

Adding `hdbscan_umap` (UMAP → 10-d, cosine; then HDBSCAN with `min_cluster_size=5`)
populates all 4 cells where raw HDBSCAN failed:

| metric | raw HDBSCAN | hdbscan_umap | note |
|---|---|---|---|
| n_clusters | 0 (all 4 embedders) | **6–7 (all 4 embedders)** | curse-of-dimensionality solved |
| silhouette | n/a | 0.07–0.13 | comparable to agglomerative |
| NPMI | n/a | **0.47–0.52** | competitive with agglomerative |
| bootstrap ARI | n/a | 0.06–0.23 | lower — UMAP is stochastic across resamples |

Reading: density-based methods are not unusable on 384–768d embeddings, they
just need a dim-reduction step. After UMAP, HDBSCAN produces more clusters
(6–7) than the partition-based methods (2–4) — these are finer-grained topics.
The tradeoff is stability: UMAP's stochasticity makes per-bootstrap clusters
different enough that mean ARI drops vs the deterministic partition methods.

## Honest limitations of this run

- N=161 is still small; ARI std (0.06–0.46) is wide, so single-run rankings
  are not robust. Bootstrap×10 was the chosen budget; ×30 would tighten CIs.
- The corpus was filtered through the same gap-extraction pipeline that
  produced the clusters, so any systematic bias in extraction (e.g.
  always-verbatim sentences from one section type) propagates.
- Some 0-cluster cells (BERTopic / HDBSCAN) are not fairly comparable to
  cells that did produce clusters — the metric tables show `nan` for
  those, but readers should not interpret it as 0 performance.
