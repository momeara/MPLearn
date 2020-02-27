#!/bin/bash

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cf10k_pca196_umap_2_15_0.0 \
       --ref_embed_dir ../umap_embedding_200217/cell_features_pca196_umap_2_15_0.0
