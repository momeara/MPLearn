#!/bin/bash

for umap_n_neighbors in 1 3 10 30 100
do
    for umap_min_dist in 0 0.1 0.3 1 3 10
    do
	command="python scripts/umap_embedding.py --dataset intermediate_data/cell_features_normed.joblib \
	       --embed_dir intermediate_data/full_normed_embedding_pca20_umap2_spectral_${umap_n_neighbors}_${umap_min_dist}_euclid \
	       --umap_n_neighbors ${umap_n_neighbors} \
	       --umap_min_dist ${umap_min_dist}"
	echo "Executing in background: $command &"
	$command &
    done
done
