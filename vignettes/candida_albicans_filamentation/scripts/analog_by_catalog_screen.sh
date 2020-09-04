#!/bin/bash


python ~/opt/MPLearn/bin/library_search \
       --query_path raw_data/Drug-repurposing\ screen\ part1\ top\ hits\ 200414.tsv \
       --library_path raw_data/selleckchem_catalog_200405.sdf_2020-04-08_registered.sdf \
       --output_path product/drug_screen_top_hits_200414_vs_selleckchem_catalog_200405.tsv \
       --query_id_field COMPOUND_NAME \
       --query_smiles_field SMILES_STRING

python ~/opt/MPLearn/bin/library_search \
       --query_path raw_data/Drug-repurposing\ screen\ part1\ top\ hits\ 200414.tsv \
       --library_path raw_data/FDA_20200328.sdf \
       --output_path product/drug_screen_top_hits_200414_vs_FDA_Only_Complete.tsv \
       --query_id_field COMPOUND_NAME \
       --query_smiles_field SMILES_STRING

mkdir -f product/figures
python ~/opt/MPLearn/bin/draw_aligned_substances \
       --substances_path raw_data/Drug-repurposing\ screen\ part1\ top\ hits\ 200414.tsv \
       --substances_id_field COMPOUND_NAME \
       --substances_smiles_field SMILES_STRING \
       --output_path product/figures/drug_screen_top_hits_200414.pdf
