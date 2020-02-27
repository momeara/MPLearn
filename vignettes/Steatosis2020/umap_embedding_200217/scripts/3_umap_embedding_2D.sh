#!/bin/bash

#############################
# embed sample of 10k cells #
#############################
# ~1 minute
embed_umap \
       --dataset intermediate_data/cf10k_normed.parquet \
       --tag cf10k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf10k_normed.parquet \
       --tag cf10k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cf10k_normed.parquet \
       --tag cf10k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf10k_normed.parquet \
       --tag cf10k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2


##############################
# embed sample of 100k cells #
##############################
# ~5 minutes
embed_umap \
       --dataset intermediate_data/cf100k_normed.parquet \
       --tag cf100k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf100k_normed.parquet \
       --tag cf100k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cf100k_normed.parquet \
       --tag cf100k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf100k_normed.parquet \
       --tag cf100k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

##############################
# embed sample of 200k cells #
##############################
# ~10 minutes
### 15 neighbors
embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_15_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_15_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_15_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_15_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2


### 30 neighbors
embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2


### 200 neighbors
embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_200_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_200_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_200_0.0 \
       --pca_n_components 196 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_pca196_umap_2_200_0.2 \
       --pca_n_components 196 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

######################
# Full cell features #
######################
## 1D
### 15 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

### 30 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

### 200 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_1_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 1 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

#2D
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2


### 30 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

### 200 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_2_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 2 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2


## 3D
### 15 neighbors 
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2


### 30 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

### 200 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_3_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 3 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2


# 6D
### 15 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_15_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_15_0.2 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 15 \
       --umap_min_dist 0.2

### 30 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_30_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_30_0.2 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.2


### 200 neighbors
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_200_0.2 \
       --umap_n_components 6 \
       --pca_n_components 196 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_200_0.0 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag cell_features_pca196_umap_6_200_0.2 \
       --pca_n_components 196 \
       --umap_n_components 6 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.2



dataset="cell_features"
#for umap_n_components in 1 2 3 6; do
umap_n_components=3
for umap_n_neighbors in 15 30 200; do
for umap_min_dist in 0.0 0.2; do
	
    command="embed_umap \
    	--dataset intermediate_data/${dataset}.parquet \
        --tag ${dataset}_pca196_umap_${umap_n_components}_${umap_n_neighbors}_${umap_min_dist} \
        --pca_n_components 196 \
	--umap_n_components ${umap_n_components} \
        --umap_n_neighbors ${umap_n_neighbors} \
        --umap_min_dist 0.0"
    echo "Executing in background: $command &"
    $command &
done
done
