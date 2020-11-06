library(plyr)
library(tidyverse)
library(arrow)

vignette_directory <- "~/opt/MPLearn/vignettes/race_bile_acid"

embedding_tag <- "UMAP_images_20201106"


data <- readr::read_csv("raw_data/race_bile_acid_images.csv")

# write out the data as a .parquet file
data %>%
    dplyr::filter(!is.na(Mean_Perinuclear_Texture_Variance_Vimentin_3_00_256)) %>%
    arrow::write_parquet("raw_data/race_bile_acid_images_MasterDataTable.parquet")


#####################################################
# Gather metadata and feature columns for the image #
#####################################################
metadata_columns <- tibble::tibble(
    feature = data %>% names()) %>%
    dplyr::filter(
        !(feature %>% stringr::str_detect("^Texture")),
        !(feature %>% stringr::str_detect("^Granularity")),
        !(feature %>% stringr::str_detect("^Mean")))

metadata_columns %>%
    readr::write_tsv("raw_data/race_bile_acid_images_metadata_columns.tsv")


feature_columns <- tibble::tibble(
    feature = data %>% names()) %>%
    dplyr::filter(feature != "Mean_Cell_AreaShape_NormalizedMoment_0_0") %>%
    dplyr::filter(feature != "Mean_Cell_AreaShape_NormalizedMoment_0_1") %>%
    dplyr::filter(feature != "Mean_Cell_AreaShape_NormalizedMoment_1_0") %>%
    dplyr::filter(feature != "Mean_Cytoplasm_AreaShape_NormalizedMoment_0_0") %>%
    dplyr::filter(feature != "Mean_Cytoplasm_AreaShape_NormalizedMoment_0_1") %>%
    dplyr::filter(feature != "Mean_Cytoplasm_AreaShape_NormalizedMoment_1_0") %>%
    dplyr::filter(feature != "Mean_Nuclei_AreaShape_NormalizedMoment_0_0") %>%
    dplyr::filter(feature != "Mean_Nuclei_AreaShape_NormalizedMoment_0_1") %>%
    dplyr::filter(feature != "Mean_Nuclei_AreaShape_NormalizedMoment_1_0") %>%
    dplyr::filter(feature != "Mean_Perinuclear_AreaShape_NormalizedMoment_0_0") %>%
    dplyr::filter(feature != "Mean_Perinuclear_AreaShape_NormalizedMoment_0_1") %>%
    dplyr::filter(feature != "Mean_Perinuclear_AreaShape_NormalizedMoment_1_0") %>%
    dplyr::filter(
        feature %>% stringr::str_detect("^Texture") |
        feature %>% stringr::str_detect("^Granularity") |
        feature %>% stringr::str_detect("^Mean")) %>%
    dplyr::mutate(
        transform = "identity")

feature_columns %>%
    readr::write_tsv("raw_data/race_bile_acid_images_feature_columns.tsv")

################
# Do embedding #
################

system(paste0("
        cd ", vignette_directory, " &&
        /home/ubuntu/anaconda3/envs/sextonlab/bin/python \\
            ~/anaconda3/envs/sextonlab/bin/embed_umap \\
            --dataset ", vignette_directory, "/raw_data/race_bile_acid_images_MasterDataTable.parquet \\
            --tag ", embedding_tag, " \\
            --feature_columns ", vignette_directory, "/raw_data/race_bile_acid_images_feature_columns.tsv \\
            --umap_n_neighbors 15 \\
            --pca_batch_size 2089 \\
            --umap_low_memory \\
            --verbose
"))

data <- 

data <- dplyr::bind_cols(
    arrow::read_parquet("raw_data/race_bile_acid_images_MasterDataTable.parquet"),
    arrow::read_parquet(paste0("intermediate_data/", embedding_tag, "/umap_embedding.parquet")))
    

source("scripts/monocle3_support.R")

cds <- populate_cds(
    cell_features = data,
    cell_feature_columns = feature_columns,
    cell_metadata_columns = metadata_columns,
    embedding_type = c("UMAP"),
    embedding = data %>% dplyr::select(UMAP_1, UMAP_2),
    verbose = TRUE)

# as resolution gets bigger --> more clusters
cds <- cds %>%
    monocle3::cluster_cells(
        reduction_method = "UMAP",
        k = 200,
        resolution = .001,
        num_iter = 10,
        verbose = TRUE)

cds %>% serialize_clusters(
    output_fname = paste0("intermediate_data/", embedding_tag, "/clusters_leiden_res=1e-2.parquet"))

system(paste0("cd ", "intermediate_data/", embedding_tag, " && ln -s clusters_leiden_res=1e-2.parquet clusters.parquet"))

# as resolution gets bigger --> more clusters
infected_cds <- infected_cds %>%
    monocle3::cluster_cells(
        reduction_method = "UMAP",
        k = 200,
        resolution = .0001,
        num_iter = 10,
        verbose = TRUE)
infected_cds %>% serialize_clusters(
    output_fname = paste0(embedding_path, "/clusters_leiden_res=5e-4.parquet"))


