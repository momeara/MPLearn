
import sys
import argparse

import pandas as pd
import joblib
from MPLearn import embedding

DESCRIPTION="""Do the embedding for one dataset with one set of parameters
version 0.0.1

Usage, for example:

    cd MPLearn/vignettes/Steatosis2020/umap_embedding_200217
    python scripts/umap_embedding.py \
        --dataset intermediate_data/cf10k.joblib \
        --embed_dir intermediate_data/cf10k_embedding_pca20_umap2_1_1_euclid \
        --umap_n_neighbors 1 \
        --umap_min_dist 1 
"""

def main(argv):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--dataset", type=str, action="store", dest="dataset",
        help="""a pandas dataframe located in .joblib file""")
    parser.add_argument(
        "--embed_dir", type=str, action="store", dest="embed_dir",
        help="""directory where to store embedding""")
    parser.add_argument(
        "--umap_n_neighbors", type=int, action="store", dest="umap_n_neighbors", default=0,
        help="""Number of neighbors when computing UMAP embedding""")
    parser.add_argument(
        "--umap_min_dist", type=float, action="store", dest="umap_min_dist", default=0.0,
        help="""Minimum distance between points in UMAP embedding""")
    parser.add_argument(
        "--verbose", type=bool, action="store", dest="verbose", default=False,
        help="""Verbose output""")

    arguments = parser.parse_args()

    dataset = joblib.load(arguments.dataset)
    dataset_embedding = embedding.fit_embedding(
        dataset=dataset,
        embed_dir=arguments.embed_dir,
        umap_init='spectral')
    embedding.plot_embedding(
        embedding=dataset_embedding,
        plot_width=2000,
        plot_height=2000,
        output_fname="{}/embedding.png".format(arguments.embed_dir))

if __name__ == "__main__":
    main(sys.argv)
