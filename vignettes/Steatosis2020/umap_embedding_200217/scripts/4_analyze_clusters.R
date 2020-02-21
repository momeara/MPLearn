
library(plyr)
library(tidyverse)
library(arrow)

cell_meta <- arrow::read_parquet("intermediate_data/cell_meta.parquet")
cluster_labels <- arrow::read_parquet("intermediate_data/full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid/hbscan_clustering_min300.parquet")
cell_clusters <- dplyr::bind_cols(cell_meta, cluster_labels)
