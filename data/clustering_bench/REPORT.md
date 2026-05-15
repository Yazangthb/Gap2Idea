# Clustering-quality benchmark (N=11 gap sentences)

> ⚠ Corpus is small (< 30 statements). Intrinsic scores are noisy and bootstrap ARI may swing wildly. Use a larger gaps.tsv for publication-grade numbers.


## silhouette

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.168 |              0.155 |               0.173 |                 0.112 |
| bertopic      |                    0.174 |              0.122 |               0.16  |                 0.114 |
| hdbscan       |                  nan     |              0.069 |             nan     |               nan     |
| kmeans        |                    0.162 |              0.166 |               0.173 |                 0.112 |

## davies_bouldin

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.751 |              1.145 |               1.384 |                 1.896 |
| bertopic      |                    1.767 |              2.098 |               1.748 |                 2.191 |
| hdbscan       |                  nan     |              1.532 |             nan     |               nan     |
| kmeans        |                    2.055 |              1.393 |               1.384 |                 1.896 |

## calinski_harabasz

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    1.446 |              1.563 |               1.674 |                 1.615 |
| bertopic      |                    1.963 |              1.699 |               1.777 |                 1.686 |
| hdbscan       |                  nan     |              1.183 |             nan     |               nan     |
| kmeans        |                    2.092 |              1.604 |               1.674 |                 1.615 |

## npmi

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.875 |              0.904 |               0.895 |                 0.868 |
| bertopic      |                    0.867 |              0.863 |               0.916 |                 0.815 |
| hdbscan       |                  nan     |              0.843 |             nan     |               nan     |
| kmeans        |                    0.879 |              0.866 |               0.895 |                 0.868 |

## bootstrap_mean_ari

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.235 |              0.863 |               0.869 |                 0.532 |
| bertopic      |                    0.119 |              0.439 |               0.623 |                 0.189 |
| hdbscan       |                    0.8   |              0.225 |               0.5   |                 0.7   |
| kmeans        |                    0.296 |              0.626 |               0.747 |                 0.461 |

## n_clusters

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                        2 |                  5 |                   4 |                     3 |
| bertopic      |                        2 |                  2 |                   3 |                     2 |
| hdbscan       |                        0 |                  2 |                   0 |                     0 |
| kmeans        |                        2 |                  5 |                   4 |                     3 |