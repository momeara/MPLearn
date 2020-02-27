
library(plyr)
library(tidyverse)
library(arrow)
library(ggplot2)

cell_meta <- arrow::read_parquet("intermediate_data/cell_meta.parquet") %>%
  dplyr::transmute(
    condition = Condition,
    dose_uM = Concentration,
    log_dose_uM = log10(Concentration),
    plate_id = Metadata_PlateID,
    row = Metadata_WellID %>%
      stringr::str_extract("^[A-Z]") %>%
      purrr::map_int(~which(LETTERS==., arr.ind=T)),
   column = Metadata_WellID %>%
      stringr::str_extract("[0-9]+$") %>%
      as.integer(),
   is_control = Condition %in% c("Positive Control", "Negative Control"))

cluster_labels <- arrow::read_parquet("intermediate_data/full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid/hdbscan_clustering_min100.parquet")
cell_clusters <- dplyr::bind_cols(cell_meta, cluster_labels)

# cluster sizes
cell_clusters %>%
    dplyr::count(cluster_label, sort=T) %>%
    data.frame

cell_clusters %>%
    dplyr::count(condition, concentration, cluster_label) %>%
    tidyr::pivot_wider(names_from=cluster_label, values_from=n)

treatment_by_cluster <- cell_clusters %>%
    dplyr::count(is_control, condition, log_dose_uM, cluster_label)


plot <- ggplot2::ggplot(
  data=treatment_by_cluster %>%
    dplyr::arrange(condition) %>%
      dplyr::mutate(
        y = paste0(condition, " | ", signif(log_dose_uM, 3)),
        x = as.character(cluster_label)))+
  ggplot2::theme_bw() +
  ggplot2::geom_raster(
    mapping=ggplot2::aes(
      x=x,
      y=y,
      fill=log(n+1))) +
  ggplot2::ggtitle(
    label="Treatment by HDBSCAN CLusters",
    subtitle="113 Steatosis2020") +
  ggplot2::scale_fill_continuous("Cell Count") +
  ggplot2::scale_y_discrete("Treatment | Concentration uM") +
  ggplot2::scale_x_discrete("HDBSCAN Cluster ID")

ggplot2::ggsave(
  filename="product/figures/full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid_hdbscan_clustering_min100.pdf",
  plot=plot,
  width=15,
  height=15)



# treatment by cluster count matrix
plot <- ggplot2::ggplot(
  data=treatment_by_cluster %>%
    dplyr::filter(is_control) %>%
    dplyr::arrange(condition) %>%
      dplyr::mutate(
        y = paste0(condition, " | ", signif(log_dose_uM, 3)),
        x = as.character(cluster_label)))+
  ggplot2::theme_bw() +
  ggplot2::geom_raster(
    mapping=ggplot2::aes(
      x=x,
      y=y,
      fill=log(n+1))) +
  ggplot2::ggtitle(
    label="Treatment by HDBSCAN CLusters",
    subtitle="113 Steatosis2020") +
  ggplot2::scale_fill_continuous("Cell Count") +
  ggplot2::scale_y_discrete("Treatment | Concentration uM") +
  ggplot2::scale_x_discrete("HDBSCAN Cluster ID")

ggplot2::ggsave(
  filename="product/figures/full_normed_embedding_pca200_umap2_spectral_30_0.0_euclid_hdbscan_clustering_min100_control.pdf",
  plot=plot,
  width=15,
  height=15)




1  0.01990000
2  0.01990010
3  0.03900400
4  0.03900420

5  0.07800799
6  0.07800838
7  0.15919996
8  0.15920075

9  0.31839979
10 0.31840138
11 0.62087911
12 0.62088220

13 1.25767623
14 1.25768249
15 2.49944000
16 2.49945275

17 4.99888000
18 4.99890549
19 9.99776000
20 9.99776025


1  0.01990000
2  0.01990010
3  0.03900400
4  0.03900420

5  0.07800799
6  0.07800838
7  0.15919996
8  0.15920075

9  0.31839979
10 0.31840138
11 0.62087911
12 0.62088220

13 1.25767623
14 1.25768249
15 2.49944000
16 2.49945275

17 4.99888000
18 4.99890549
19 9.99776000
20 9.99776025
