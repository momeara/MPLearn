#!/bin/bash

#############################
# embed sample of 10k cells #
#############################
# ~1 minute
python scripts/umap_embedding.py \
       --dataset intermediate_data/cf10k_normed.joblib \
       --tag cf10k_normed_embedding_pca20_umap2_spectral_30_0_euclid \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

##############################
# embed sample of 100k cells #
##############################
# ~5 minutes
python scripts/umap_embedding.py \
       --dataset intermediate_data/cf100k_normed.joblib \
       --tag cf100k_normed_embedding_pca20_umap2_spectral_30_0_euclid \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

##############################
# embed sample of 200k cells #
##############################
# ~10 minutes
python scripts/umap_embedding.py \
       --dataset intermediate_data/cf200k_normed.joblib \
       --tag cf200k_normed_embedding_pca20_umap2_spectral_30_0_euclid \
       --umap_n_neighbors 30 \
       --umap_min_dist 0

# scan number of neighbors parameters
# spectral initialization
for umap_n_neighbors in 3 10 30 100
do
    command="python scripts/umap_embedding.py \
                 --dataset intermediate_data/cf200k_normed.joblib \
                 --tag cf200k_normed_embedding_pca20_umap2_${umap_n_neighbors}_0.0_euclid \
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
    command="python scripts/umap_embedding.py \
                 --dataset intermediate_data/cf200k_normed.joblib \
                 --tag cf200k_normed_embedding_pca20_umap2_30_${umap_min_dist}_euclid \
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
python scripts/umap_embedding.py \
       --dataset intermediate_data/cell_features_normed.joblib \
       --tag full_normed_embedding_pca20_umap2_spectral_30_0_euclid \
       --umap_n_neighbors 30 \
       --umap_min_dist 0.1 \
       --umap_init spectral

# scan number of neighbors and minimum distance parameters
# random initialization
for umap_n_neighbors in 3 10 30 100 200
do
    for umap_min_dist in 0.0 0.1 0.25 0.5 0.8 0.99
    do
	command="python scripts/umap_embedding.py \
               --dataset intermediate_data/cell_features_normed.joblib \
	       --tag full_normed_embedding_pca20_umap2_${umap_n_neighbors}_${umap_min_dist}_euclid \
	       --umap_n_neighbors ${umap_n_neighbors} \
	       --umap_min_dist ${umap_min_dist} \
               --umap_init random"
	echo "Executing in background: $command &"
	$command &
    done
done



# scan number of neighbors and minimum distance parameters
# spectral initialization
for umap_n_neighbors in 3 10 30 100
do
    for umap_min_dist in 0.0 0.1 0.25 0.5 0.8 0.99
    do
	command="python scripts/umap_embedding.py \
               --dataset intermediate_data/cell_features_normed.joblib \
	       --tag full_normed_embedding_pca20_umap2_spectral_${umap_n_neighbors}_${umap_min_dist}_euclid \
	       --umap_n_neighbors ${umap_n_neighbors} \
	       --umap_min_dist ${umap_min_dist} \
               --umap_init spectral"
	echo "Executing in background: $command &"
	$command &
    done
done
