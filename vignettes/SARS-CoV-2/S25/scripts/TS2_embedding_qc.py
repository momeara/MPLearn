#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import sys

import pandas as pd
import pyarrow.parquet
import pyarrow as pa
import numpy as np
import MPLearn
import MPLearn.embedding_qc

cell_features_fname = "/home/ubuntu/projects/SARS-CoV-2/product/covid19cq1_SARS_TS2_2M_Cell_MasterDataTable.parquet"
cell_feature_columns_fname = "/home/ubuntu/projects/SARS-CoV-2/raw_data/cell_feature_columns_TS_202008.tsv" 
cell_feature_columns = pd.read_csv(cell_feature_columns_fname, sep="\t")

cell_embedding_fname = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/umap_embedding.parquet"


cell_features = pa.parquet.read_table(
    source=cell_features_fname,
    columns=cell_feature_columns['feature'].to_list()
    ).to_pandas().astype('float32')

cell_embedding = pa.parquet.read_table(
    source=cell_embedding_fname).to_pandas().astype('float32')


random_subset = 10000
sample_indices = np.random.choice(cell_features.shape[0], random_subset, replace=False)
cell_features = cell_features.iloc[sample_indices]
cell_embedding = cell_embedding.iloc[sample_indices]


(
    native_normed_distances,
    embedded_normed_distances,
    pearson_correlation,
    earth_movers_distance
) = MPLearn.embedding_qc.distortion_statistics(
    native_coordinates = cell_features,
    embedded_coordinates = cell_embedding,
    metric = "euclidean",
    use_dask = False,
    verbose = True)

sp_plot = MPLearn.embedding_qc.SP_plot(
    native_normed_distances,
    embedded_normed_distances)

pa.parquet.write_table(
    table=pa.Table.from_pandas(native_normed_distances),
    where="intermediate_data/{}/hdbscan_clustering_min{}.parquet".format(arguments.tag, arguments.hdbscan_min_cluster_size))



sp_plot.plot_cell_distances(
    save_to = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/figures/sp_cell_distances.png")

sp_plot.plot_distributions(
    save_to = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/figures/sp_distributions.png")

sp_plot.plot_cumulative_distributions(
    save_to = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/figures/sp_cumulative_distributions.png")

sp_plot.plot_distance_correlation(
    save_to = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/figures/sp_distance_correlation.png")

sp_plot.joint_plot_distance_correlation(
    save_to = "/home/ubuntu/opt/MPLearn/vignettes/SARS-CoV-2/S25/intermediate_data/UMAP_embedding_TS2_2M_epochs=2000_20200901/figures/sp_joint_distance_correlation.png")
