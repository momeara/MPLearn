

for plate_id in 1001 1002 1003 1004 1005; do
    for dose in 0050 0250 0500 1000 2000; do
	dataset="raw_data/SARS_${plate_id}${dose}A_Cells_masterDataTable.parquet"
	tag="SARS_${plate_id}${dose}A_umap2_2M_15_0.0"

	echo "########################################################"
	echo "# Embedding ${tag}"
	echo "########################################################"
        ~/anaconda3/envs/sextonlab/bin/embed_umap \
            --dataset ${dataset} \
            --tag ${tag} \
            --feature_columns raw_data/cell_feature_columns.tsv \
	    --no_save_transform \
	    --random_subset 2000000 \
	    --umap_low_memory \
	    --verbose
    done
done



for umap_n_neighbors in 5, 10, 15, 30, 50, 75; do
    dataset="raw_data/top_hits_plate_scaled_200522a_Cell_MasterDataTable.parquet"
    tag="top_hits_plate_scaled_200522a_umap2_2M_${umap_n_neighbors}_0.0"
    echo "########################################################"
    echo "# Embedding N neighbors ${umap_n_neighbors}"
    echo "########################################################"
    ~/anaconda3/envs/sextonlab/bin/embed_umap \
        --dataset ${dataset} \
        --tag ${tag} \
        --feature_columns raw_data/cell_feature_columns.tsv \
	--no_standardize_features \
	--no_save_transform \
	--umap_n_neighbors ${umap_n_neighbors} \
	--umap_low_memory \
	--verbose
done


for rep in 3 4 5 6 7 8 9 10; do
    dataset="raw_data/top_hits_plate_scaled_200522a_Cell_MasterDataTable.parquet"
    tag="top_hits_plate_scaled_200522a_rep${rep}_umap2_2M_15_0.0"
    echo "########################################################"
    echo "# Embedding replication ${rep}"
    echo "########################################################"
    ~/anaconda3/envs/sextonlab/bin/embed_umap \
        --dataset ${dataset} \
        --tag ${tag} \
        --feature_columns raw_data/cell_feature_columns.tsv \
	--no_standardize_features \
	--no_save_transform \
	--umap_low_memory \
	--verbose --seed=${rep}
done
