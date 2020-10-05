
library(plyr)
library(tidyverse)
library(DBI)
library(RPostgres)


library(MPStats)

get_acas_database <- function(stage=FALSE){
    con <- DBI::dbConnect(
        RPostgres::Postgres(),
        host=paste0("umsexton", ifelse(stage, "-stage", ""), ".onacaslims.com"),
        port=5432,
        user="acas",
        password="acas")
    con %>% DBI::dbSendQuery("SET search_path TO acas;")
    con
}

con <- get_acas_database()

# get all foreign key constraints in the schema
tables <- tbl(con, dbplyr::in_schema("information_schema", "tables"))
table_constraints <- tbl(con, dbplyr::in_schema("information_schema", "table_constraints"))
key_column_usage <- tbl(con, dbplyr::in_schema("information_schema", "key_column_usage"))
referential_constraints <- tbl(con, dbplyr::in_schema("information_schema", "referential_constraints"))
constraint_column_usage <- tbl(con, dbplyr::in_schema("information_schema", "constraint_column_usage"))

keys <- tables %>%
    dplyr::select(
        table_catalog,
        table_schema,
        table_type,
        table_name) %>%
    dplyr::left_join(
        table_constraints %>%
        dplyr::select(
            constraint_catalog,  # identify the constraint
            constraint_schema,   # identify the constraint
            constraint_name,     # identify the constraint
            table_catalog,     # join with table
            table_schema,      # join with table
            table_name,        # join with thable
            constraint_type),
        by = c(
            "table_catalog" = "table_catalog",
            "table_schema" = "table_schema",
            "table_name" = "table_name")) %>%
    dplyr::filter(
        constraint_type %in% c("FOREIGN KEY", "PRIMARY KEY")) %>%
    dplyr::left_join(
        key_column_usage %>%
        dplyr::select(
            constraint_catalog,  # identify the constraint
            constraint_schema,   # identify the constraint
            constraint_name,     # identify the constraint
            table_catalog,     # join with table
            table_schema,      # join with table
            table_name,        # join with thable
            column_name),
        by = c(
            "constraint_catalog" = "constraint_catalog",
            "constraint_schema" = "constraint_schema",
            "constraint_name" = "constraint_name",            
            "table_catalog" = "table_catalog",
            "table_name" = "table_name",
            "table_schema" = "table_schema",
            "constraint_name" = "constraint_name")) %>%
    dplyr::left_join(
        constraint_column_usage %>%
        dplyr::select(
            constraint_catalog,  # identify the constraint
            constraint_schema,   # identify the constraint
            constraint_name,     # identify the constraint
            table_catalog,         # join with table
            table_schema,          # join with table
            foreign_table_name = table_name,
            foreign_column_name = column_name),
        by = c(
            "constraint_catalog",
            "constraint_schema",
            "constraint_name",            
            "constraint_catalog",
            "table_catalog",
            "table_schema")) %>%
    select(
        table_name,
        table_type,
        constraint_name,
        constraint_type,
        column_name,
        foreign_table_name,
        foreign_column_name) %>%
    arrange(table_name) %>%
    collect()
glimpse(keys)

keys %>% readr::write_tsv("product/ACAS/acas_table_foreign_keys.tsv")



