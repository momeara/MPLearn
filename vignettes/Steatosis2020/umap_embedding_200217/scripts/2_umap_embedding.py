
import pandas as pd
import joblib
from MPLearn import embedding

cf10k = joblib.load("intermediate_data/cf10k.joblib")


cf10k_embedding = embedding.fit_embedding(
    dataset=cf10k,
    embed_dir="intermediate_data/cf10k_embedding_pca20_umap2_100_0_euclid")

embedding.plot_embedding(
    embedding=cf10k_embedding,
    output_fname="product/figures/cf10_embedding_pca20_umap2_100_0_euclid.png")



############3
cell_features = joblib.load("intermediate_data/cell_features.joblib")


full_embedding = embedding.fit_embedding(
    dataset=cell_features,
    embed_dir="intermediate_data/full_embedding_pca20_umap2_100_0_euclid")

embedding.plot_embedding(
    embedding=full_embedding,
    output_fname="product/figures/full_embedding_pca20_umap2_100_0_euclid.png")



