
import pandas as pd
import joblib
from MPLearn import embedding

cf10k = joblib.load("intermediate_data/cf10k.joblib")
cf10k_embedding = embedding.fit_embedding(
    dataset=cf10k,
    embed_dir="intermediate_data/cf10k_embedding_pca20_umap2_100_0_euclid")
embedding.plot_embedding(
    embedding=cf10k_embedding,
    output_fname="product/figures/cf10k_embedding_pca20_umap2_100_0_euclid.png")

cf100k = joblib.load("intermediate_data/cf100k.joblib")
cf100k_embedding = embedding.fit_embedding(
    dataset=cf100k,
    embed_dir="intermediate_data/cf100k_embedding_pca20_umap2_100_0_euclid")
embedding.plot_embedding(
    embedding=cf100k_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/cf100k_embedding_pca20_umap2_100_0_euclid.png")


##########
cf200k = joblib.load("intermediate_data/cf200k.joblib")
cf200k_embedding = embedding.fit_embedding(
    dataset=cf200k,
    embed_dir="intermediate_data/cf200k_embedding_pca20_umap2_100_0_euclid")
embedding.plot_embedding(
    embedding=cf200k_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/cf200k_embedding_pca20_umap2_100_0_euclid.png")

cf200k_normed = joblib.load("intermediate_data/cf200k_normed.joblib")
cf200k_normed_embedding = embedding.fit_embedding(
    dataset=cf200k_normed,
    embed_dir="intermediate_data/cf200k_normed_embedding_pca20_umap2_100_0_euclid")
embedding.plot_embedding(
    embedding=cf200k_normed_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/cf200k_normed_embedding_pca20_umap2_100_0_euclid.png")

cf200k_normed = joblib.load("intermediate_data/cf200k_normed.joblib")
cf200k_normed_embedding = embedding.fit_embedding(
    dataset=cf200k_normed,
    embed_dir="intermediate_data/cf200k_normed_embedding_pca20_umap2_spectral_100_0_euclid",
    umap_init='spectral')
embedding.plot_embedding(
    embedding=cf200k_normed_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/cf200k_normed_embedding_pca20_umap2_spectral_100_0_euclid.png")


###################
cf200k = joblib.load("intermediate_data/cf200k.joblib")
cf200k_embedding = embedding.fit_embedding(
    dataset=cf200k,
    embed_dir="intermediate_data/cf200k_embedding_pca20_umap2_spectral_100_0_euclid",
    umap_init='spectral')    
embedding.plot_embedding(
    embedding=cf200k_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/cf200k_embedding_pca20_umap2_spectral_100_0_euclid.png")

#####################
cf200k_normed = joblib.load("intermediate_data/cf200k_normed.joblib")
for umap_n_neighbors in [10, 30, 100, 300, 1000, 30000]:
    for umap_min_dist in [0.0, 0.1, 0.3, 1.0, 3.0, 10, 30, 100]:
        print("umap_neighbors: {} umap_min_dist: {}".format(umap_n_neighbors, umap_min_dist))
        tag = "cf200k_normed_embedding_pca20_umap2_spectral_{}_{}_euclid".format(
            umap_n_neighbors,
            umap_min_dist) 
        cf200k_normed_embedding = embedding.fit_embedding(
            dataset=cf200k_normed,
            embed_dir="intermediate_data/{}".format(tag),
            umap_init='spectral')
        embedding.plot_embedding(
            embedding=cf200k_normed_embedding,
            plot_width=800,
            plot_height=800,
            output_fname="product/figures/{}.png".format(tag))
        






############3
cell_features = joblib.load("intermediate_data/cell_features.joblib")
full_embedding = embedding.fit_embedding(
    dataset=cell_features,
    embed_dir="intermediate_data/full_embedding_pca20_umap2_100_0_euclid")
embedding.plot_embedding(
    embedding=full_embedding,
    plot_width=800,
    plot_height=800,
    output_fname="product/figures/full_embedding_pca20_umap2_100_0_euclid.png")



