# -*- tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


import hdbscan
import joblib

cf10k_embedding = joblib.load("intermediate_data/cf10k_embedding_pca20_umap2_100_0_euclid/umap_embedding.joblib")

cf10k_clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
cf10k_cluster_labels = cf10k_clusterer.fit_predict(cf10k_embedding)

joblib.dump(
    value=cf10k_cluster_labels,
    filename="intermediate_data/cf10k_embedding_pca20_umap2_10_0_euclid/hdbscan_clusterer.joblib")
joblib.dump(
    value=cf10k_cluster_labels,
    filename="intermediate_data/cf10k_embedding_pca20_umap2_10_0_euclid/hdbscan_clustering.joblib")


#############
full_embedding = joblib.load("intermediate_data/full_embedding_pca20_umap2_100_0_euclid/umap_embedding.joblib")

full_clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
full_cluster_labels = full_clusterer.fit_predict(full_embedding)

joblib.dump(
    value=full_clusterer,
    filename="intermediate_data/full_embedding_pca20_umap2_100_0_euclid/hdbscan_clusterer.joblib")
joblib.dump(
    value=full_cluster_labels,
    filename="intermediate_data/full_embedding_pca20_umap2_100_0_euclid/hdbscan_clustering.joblib")
