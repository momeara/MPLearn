#!/bin/bash

#############################
# embed sample of 10k cells #
#############################
# ~1 minute
embed_umap \
       --dataset intermediate_data/cf10k_normed.parquet \
       --tag cf10k_normed_embedding_pca200_umap2_spectral_30_0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

##############################
# embed sample of 100k cells #
##############################
# ~5 minutes
embed_umap \
       --dataset intermediate_data/cf100k_normed.parquet \
       --tag cf100k_normed_embedding_pca200_umap2_spectral_30_0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

##############################
# embed sample of 200k cells #
##############################
# ~10 minutes
embed_umap \
       --dataset intermediate_data/cf200k_normed.parquet \
       --tag cf200k_normed_embedding_pca200_umap2_spectral_30_0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

# scan number of neighbors parameters
# spectral initialization
for umap_n_neighbors in 3 10 30 100
do
    command="embed_umap \
                 --dataset intermediate_data/cf200k_normed.parquet \
                 --tag cf200k_normed_embedding_pca200_umap2_${umap_n_neighbors}_0.0_euclid \
                 --pca_n_components 200 \
                 --umap_n_neighbors ${umap_n_neighbors} \
                 --umap_min_dist 0.0 \
                 --umap_init spectral"
    echo "Executing in background: $command &"
    $command &
done

# scan min_dist parameters
# spectral initialization
for umap_min_dist in 0.0 0.1 0.25 0.5 0.8 0.99
do
    command="embed_umap \
                 --dataset intermediate_data/cf200k_normed.parquet \
                 --tag cf200k_normed_embedding_pca200_umap2_30_${umap_min_dist}_euclid \
                 --pca_n_components 200 \
                 --umap_n_neighbors 30 \
                 --umap_min_dist ${umap_min_dist} \
                 --umap_init spectral"
    echo "Executing in background: $command &"
    $command &
done


################################
# embed full set of 1.4M cells #
################################
# ~90 minutes
# fewer neighbors is better for more local structure
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag full_normed_embedding_pca200_umap2_spectral_30_0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0 \
       --umap_init spectral

# more neighbors is better for more global structure
embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag full_normed_embedding_pca200_umap2_spectral_200_0_euclid \
       --pca_n_components 200 \       
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0 \
       --umap_init spectral


embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.0 \
       --umap_init spectral

embed_umap \
       --dataset intermediate_data/cell_features_normed.parquet \
       --tag full_normed_embedding_pca200_umap2_spectral_200_0.0_euclid \
       --pca_n_components 200 \
       --umap_n_neighbors 200 \
       --umap_min_dist 0.0 \
       --umap_init spectral



# scan number of neighbors and minimum distance parameters
# random initialization
for umap_n_neighbors in 3 10 30 100 200
do
    for umap_min_dist in 0.0 0.1 0.25 0.5 0.8 0.99
    do
	command="embed_umap \
               --dataset intermediate_data/cell_features_normed.parquet \
	       --tag full_normed_embedding_pca200_umap2_${umap_n_neighbors}_${umap_min_dist}_euclid \
               --pca_n_components 200 \
	       --umap_n_neighbors ${umap_n_neighbors} \
	       --umap_min_dist ${umap_min_dist} \
               --umap_init random"
	echo "Executing in background: $command &"
	$command &
    done
done



# scan number of neighbors and minimum distance parameters
# spectral initialization
for umap_n_neighbors in 3 10 30 100 200
do
    for umap_min_dist in 0.0 0.1 0.25
    do
	command="embed_umap \
               --dataset intermediate_data/cell_features_normed.parquet \
	       --tag full_normed_embedding_pca200_umap2_spectral_${umap_n_neighbors}_${umap_min_dist}_euclid \
               --pca_n_components 200 \
	       --umap_n_neighbors ${umap_n_neighbors} \
	       --umap_min_dist ${umap_min_dist} \
               --umap_init spectral"
	echo "Executing in background: $command &"
	$command &
    done
done
