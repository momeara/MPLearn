#!/bin/bash

# mount S3 bucket
sudo s3fs \
     cellprofilerdata \
     -o use_cache=/home/ubuntu/tmp \
     -o uid=1001 \
     -o mp_umask=002 \
     -o multireq_max=5 \
     -o iam_role="SextonS3" \
     -o allow_other \
     ~/bucket_cellprofilerdata

# load plate maps
sudo cp -r \
     ~/bucket_cellprofilerdata/PAINS/Plate\ Maps \
     raw_data/
sudo chmod a+x -R raw_data/Plate\ Maps
sudo chmod a+r -R raw_data/Plate\ Maps

# load compound maps
sudo cp \
     ~/bucket_cellprofilerdata/PAINS/NextGenKATinhibitors-V9-SuppData1.xlsx \
     raw_data/Plate\ Maps/
sudo chmod a+r raw_data/Plate\ Maps/NextGenKATinhibitors-V9-SuppData1.xlsx



for dataset_id in $(tail -n +2 raw_data/dataset_ids_todo.tsv)
do
    echo "loading dataset '${dataset_id}' ..."
    mkdir raw_data/${dataset_id}
    sudo cp \
	 ~/bucket_cellprofilerdata/PAINS/${dataset_id}/cpdata.h5 \
	 raw_data/${dataset_id}/cpdata.h5
    sudo chmod a+r raw_data/${dataset_id}/cpdata.h5
done


for dataset_id in $(grep "	Ono20	" raw_data/dataset_ids.tsv | cut -d "	" -f1 )
do
    echo "loading dataset '${dataset_id}' ..."
    rm -rf raw_data/${dataset_id}
    mkdir raw_data/${dataset_id}
    sudo cp \
	 ~/bucket_cellprofilerdata/PAINS/${dataset_id}/cpdata.h5 \
	 raw_data/${dataset_id}/cpdata.h5
    sudo chmod a+r raw_data/${dataset_id}/cpdata.h5
done


for dataset_id in $(grep "	Ono21	" raw_data/dataset_ids.tsv | cut -d "	" -f1 )
do
    echo "loading dataset '${dataset_id}' ..."
    rm -rf raw_data/${dataset_id}
    mkdir raw_data/${dataset_id}
    sudo cp \
	 ~/bucket_cellprofilerdata/PAINS/${dataset_id}/cpdata.h5 \
	 raw_data/${dataset_id}/cpdata.h5
    sudo chmod a+r raw_data/${dataset_id}/cpdata.h5
done


