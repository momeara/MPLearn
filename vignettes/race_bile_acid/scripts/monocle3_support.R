
library(plyr)
library(tidyverse)
library(monocle3)

#' Popluate a Monocle3 cell data set from cell features
#'
#' The Monocle3 cell data set is a container
#'  
#' @param cell_features data.frame:
#'        rows: cells, columns: features
#' @param cell_metadata_columns data.frame:
#'        rows: features, columns feature metadata
#'        Note that there must be a column named `feature`
#' @param cell_metadata
#'        rows: cells, columns cell metadata
#'
#' 
populate_cds <- function(
    cell_features,
    cell_feature_columns,
    cell_metadata_columns,
    embedding_type = c("UMAP"),
    embedding = NULL,
    verbose = FALSE) {

    assertthat::assert_that("feature" %in% names(cell_feature_columns))

    n_cells <- nrow(cell_features)
    n_features <- nrow(cell_feature_columns)
    cat("Loading cell dataset with dimensions [<feature>, <cell>] = [", n_features, ", ",  n_cells, "]\n", sep = "")
    
    expression_data <- cell_features %>%
        dplyr::select(
            tidyselect::one_of(cell_feature_columns$feature)) %>%
        as.matrix() %>%
        t()
    gene_metadata <- cell_feature_columns %>%
        dplyr::mutate(
            gene_short_name = feature)
    cell_metadata <- cell_features %>%
        dplyr::select(
            tidyselect::one_of(cell_metadata_columns$feature))

    row.names(expression_data) <- cell_feature_columns$feature
    row.names(gene_metadata) <- cell_feature_columns$feature
    names(expression_data) <- expression_data %>% ncol %>% seq_len
    row.names(cell_metadata) <- expression_data %>% ncol %>% seq_len

    if(verbose){
        cat("Creating a SingleCellExperiment object ...\n")
    }
    # unpack monocle3::new_cell_data_set(...)
    # to not use dgCMatrix for the expression matrix they are dense feature matrices
    sce <- SingleCellExperiment(
        list(counts = expression_data),
        rowData = gene_metadata,
        colData = cell_metadata)

    if(verbose){
        cat("Creating a Cell Data Set object ...\n")
    }
    cds <- methods::new(
        Class = "cell_data_set",
        assays = SummarizedExperiment::Assays(list(counts = expression_data)),
        colData = colData(sce),
        int_elementMetadata = int_elementMetadata(sce),
        int_colData = int_colData(sce),
        int_metadata = int_metadata(sce),
        metadata = S4Vectors::metadata(sce),
        NAMES = NULL,
        elementMetadata = elementMetadata(sce)[,0],
        rowRanges = rowRanges(sce))

    if(verbose){
        cat("Configuring the cell data set ...\n")
    }

    S4Vectors::metadata(cds)$cds_version <- Biobase::package.version("monocle3")
    clusters <- stats::setNames(S4Vectors::SimpleList(), character(0))
    cds <- monocle3::estimate_size_factors(cds)
    cds

    row.names(SummarizedExperiment::colData(cds)) <- expression_data %>% ncol %>% seq_len
    if (!is.null(embedding)) {
        SingleCellExperiment::reducedDims(cds)[[embedding_type]] <- embedding
    }
    cds
}


serialize_clusters <- function(
    cds,
    output_fname,
    reduction_method = c("UMAP", "tSNE", "PCA", "LSI", "Aligned"),
    verbose = FALSE) {

    reduction_method <- match.arg(reduction_method)
    assertthat::assert_that(methods::is(cds, "cell_data_set"))
    assertthat::assert_that(is.logical(verbose))

    assertthat::assert_that(!is.null(cds@clusters[[reduction_method]]),
        msg = paste("No cell clusters for", reduction_method,
            "calculated.", "Please run cluster_cells with", "reduction_method =",
            reduction_method, "before trying to serialize clusters."))

    if(verbose) {
        message(
            "Writing clusters for cell data set reduced by ",
            "'", reduction_method, "' reduction method to ",
            "'", output_fname, "'\n", sep = "")
    }
    
    cds@clusters[[reduction_method]]$clusters %>%
        data.frame(cluster_label = .) %>%
        arrow::write_parquet(output_fname)
}
    

