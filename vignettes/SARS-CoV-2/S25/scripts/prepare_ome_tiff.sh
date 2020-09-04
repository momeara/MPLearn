#!/bin/bash

s3fs \
    sextoncov19 \
    -o use_cache=/home/ubuntu/tmp \
    -o uid=1001 \
    -o mp_umask=002 \
    -o multireq_max=5 \
    -o iam_role="SextonS3" \
    -o allow_other ~/bucket



# copy the images for a plate/well to a local path
plate_id=20200427T024129_1999B
well_id=0001
local_path=intermediate_data/CQ1_projection_images/${plate_id}
mkdir -p ${local_path}/Projection
cp ~/bucket/CQ1/${plate_id}/MeasurementResultMIP.ome.tif ${local_path}/

for s3_image_path in $(ls ~/bucket/CQ1/${plate_id}/Projection/W${well_id}*);
do
    echo "Copying ${s3_image_path} to ${local_path}/Projection/"
    cp ${s3_image_path} ${local_path}/Projection/;
done


# doesn't work because it doesn't like that ${local_path}/Projection is a directory
bfconvert -stitch "${local_path}/Projection/W${well_id}F<0001-0009>T0001Z000C<1-4>.tif" ${local_path}.ome.tif
bioformats2raw ${local_path}.ome.tif ${local_path}/tile_directory
raw2ometiff  ${local_path}/tile_directory ${local_path}_compressed.ome.tif --compression=zlib


cp ${local_path}_compressed.ome.tiff ~/bucket_sextonpublicimages/SARS-COV-2/
echo "{}" > ${local_path}_compressed.offsets.json
sudo cp ${local_path}_compressed.offsets.json ~/bucket_sextonpublicimages/SARS-COV-2/

# https://github.com/hms-dbmi/viv/blob/master/tutorial/README.md
bioformats2raw LuCa-7color_Scan1.qptiff n5_tile_directory/
raw2ometiff n5_tile_directory/ LuCa-7color_Scan1.ome.tif --compression=zlib


# run tile_well_images.py to generate tiles
bfconvert -stitch "${local_path}/Projection/W${well_id}T0001Z000C<1-4>.tif" ${local_path}_tiled.ome.tif
bioformats2raw ${local_path}_tiled.ome.tif ${local_path}/tile_directory
raw2ometiff  ${local_path}/tile_directory ${local_path}_tiled.ome.tif --compression=zlib
sudo cp ${local_path}_tiled.ome.tif ~/bucket_sextonpublicimages/SARS-COV-2/
echo "{}" > ${local_path}_tiled.offsets.json
sudo cp ${local_path}_tiled.offsets.json ~/bucket_sextonpublicimages/SARS-COV-2/

