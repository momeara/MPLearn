

library(tidyverse)
library(readxl)
###################################
# Analogs for top series May 21st #
###################################

readr::read_tsv("raw_data/Analogs\ for\ top\ series\ 200521\ -\ Sheet1.tsv") %>%
    dplyr::group_by(`WuXi ID`) %>%
    dplyr::do({
        tag <- paste0("series_", .$`Grace Series`[1], "_analogs") %>%
            stringr::str_replace_all(" ", "_") %>%
            stringr::str_replace_all("/", "_")
        series_analogs_fname <- paste0("intermediate_data/", tag, ".tsv")
        dplyr::bind_rows(
            head(., 1) %>% dplyr::select(id = `WuXi ID`, smiles = smiles),
            dplyr::select(.,
                id = `Analog chemspace ID`,
                smiles = `Analog Smiles`)) %>%
            readr::write_tsv(series_analogs_fname)
        command <- paste0(
            "/home/ubuntu/anaconda3/envs/sextonlab/bin/python ",
            "~/opt/MPLearn/bin/draw_aligned_substances ",
            "--substances_path ", series_analogs_fname, " ",
            "--template_smiles '", .$smiles[1], "' ",
            "--output_path product/figures/", tag, ".png")
        cat(command, "\n", sep = "")
        command %>% system()
        data.frame()
    })


#############################
# Series three SAR Aug 13th #
#############################

readxl::read_xlsx("raw_data/galen_SAR_20200813.xlsx") %>%
    dplyr::transmute(
        series = `Grace's series`,
        subseries = `Subseries`,
        subseries_template = `Subseries Template`,
        substance_id = substance_id,
        smiles = Smiles,
        ic50_uM = dplyr::case_when(
            !is.na(`IC50 (uM) - Galen`) ~ `IC50 (uM) - Galen` %>% as.character(),
            TRUE ~ "Inactive")) %>%
    dplyr::group_by(subseries) %>%
    dplyr::do({
        data <- .
        tag <- paste0("series_", data$subseries[1], "_SAR")
        series_analogs_fname <- paste0("intermediate_data/", tag, "_2020813.tsv")
        data %>%
            readr::write_tsv(., series_analogs_fname)
        command <- paste0(
            "/home/ubuntu/anaconda3/envs/sextonlab/bin/python ",
            "~/opt/MPLearn/bin/draw_aligned_substances ",
            "--substances_id_field substance_id ",
            "--substances_smiles_field smiles ",
            "--substances_path ", series_analogs_fname, " ",
#            "--template_smiles '", data$subseries_template[1], "' ",
            "--output_path product/figures/", tag, ".png ",
            "--verbose")
        cat(command, "\n", sep = "")
        command %>% system()
        data.frame()
    })



