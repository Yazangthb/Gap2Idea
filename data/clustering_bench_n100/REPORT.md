# Clustering-quality benchmark (N=161 gap sentences)


## silhouette

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.099 |              0.044 |               0.08  |                 0.1   |
| bertopic      |                  nan     |            nan     |               0.086 |               nan     |
| hdbscan       |                  nan     |            nan     |             nan     |               nan     |
| hdbscan_umap  |                    0.105 |              0.124 |               0.09  |                 0.073 |
| kmeans        |                    0.091 |              0.059 |               0.073 |                 0.066 |

## davies_bouldin

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    1.359 |              3.059 |               2.812 |                 1.661 |
| bertopic      |                  nan     |            nan     |               3.776 |               nan     |
| hdbscan       |                  nan     |            nan     |             nan     |               nan     |
| hdbscan_umap  |                    2.767 |              2.646 |               2.918 |                 3.015 |
| kmeans        |                    3.753 |              4.068 |               4.195 |                 4.869 |

## calinski_harabasz

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    1.676 |              3.017 |               3.362 |                 1.776 |
| bertopic      |                  nan     |            nan     |               4.865 |               nan     |
| hdbscan       |                  nan     |            nan     |             nan     |               nan     |
| hdbscan_umap  |                    4.538 |              3.9   |               4.566 |                 3.768 |
| kmeans        |                    7.598 |              5.096 |               6.46  |                 6.659 |

## npmi

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.591 |              0.596 |               0.585 |                 0.411 |
| bertopic      |                  nan     |            nan     |               0.44  |               nan     |
| hdbscan       |                  nan     |            nan     |             nan     |               nan     |
| hdbscan_umap  |                    0.524 |              0.518 |               0.467 |                 0.485 |
| kmeans        |                    0.41  |              0.376 |               0.378 |                 0.259 |

## bootstrap_mean_ari

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                    0.325 |              0.24  |               0.229 |                 0.018 |
| bertopic      |                    0.1   |              0.1   |               0.228 |                 0.1   |
| hdbscan       |                    0.5   |              0.2   |               0     |                 0.3   |
| hdbscan_umap  |                    0.056 |              0.115 |               0.233 |                 0.16  |
| kmeans        |                    0.356 |              0.295 |               0.318 |                 0.178 |

## n_clusters

| clusterer     |   BAAI/bge-small-en-v1.5 |   all-MiniLM-L6-v2 |   all-mpnet-base-v2 |   intfloat/e5-base-v2 |
|:--------------|-------------------------:|-------------------:|--------------------:|----------------------:|
| agglomerative |                        3 |                  4 |                   3 |                     2 |
| bertopic      |                        0 |                  0 |                   3 |                     0 |
| hdbscan       |                        0 |                  0 |                   0 |                     0 |
| hdbscan_umap  |                        6 |                  7 |                   6 |                     7 |
| kmeans        |                        3 |                  4 |                   3 |                     2 |