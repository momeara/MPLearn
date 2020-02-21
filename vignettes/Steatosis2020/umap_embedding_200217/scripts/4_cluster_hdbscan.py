# -*- tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


import hdbscan
import joblib
import pandas as pd
import pyarrow.parquet
import pyarrow as pa



cf10k_embedding = pa.parquet.read_table(
		source="intermediate_data/cf10k_embedding_pca20_umap2_100_0_euclid/umap_embedding.parquet")
cf10k_embedding = cf10k_embedding.to_pandas()

cf10k_clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
cf10k_cluster_labels = cf10k_clusterer.fit_predict(cf10k_embedding)
joblib.dump(
    value=cf10k_cluster_labels,
    filename="intermediate_data/cf10k_embedding_pca20_umap2_10_0_euclid/hdbscan_clusterer.joblib")
joblib.dump(
    value=cf10k_cluster_labels,
    filename="intermediate_data/cf10k_embedding_pca20_umap2_10_0_euclid/hdbscan_clustering.joblib")


#############
embed_dir = "intermediate_data/full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid"
embedding = pa.parquet.read_table(source="{}/umap_embedding.parquet".format(embed_dir)).to_pandas()
for min_cluster_size in [30, 100, 300, 1000]:
    print("fitting HDBSCAN model with min_cluster_size={} ...".format(min_cluster_size))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(embedding)
    cluster_labels = pd.DataFrame(cluster_labels, columns=['cluster_label'])
    joblib.dump(
    		value=clusterer,
    		filename="{}/hdbscan_clusterer.joblib".format(embed_dir))
    pa.parquet.write_table(
    		pa.Table.from_pandas(cluster_labels),
    		"{}/hdbscan_clustering_min{}.parquet".format(embed_dir,min_cluster_size))
