# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import os
import io
import math
import PIL
import PIL.ImageDraw
import PIL.ImageOps
import boto3
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib as mpl
import xlsxwriter




def retrieve_cell_coordinates_from_db(
        con,
        cell_ids,
        key_object,
        dyes,
        verbose=False):
    """
    cell_ids is a pd.DataFrame with rows representing cells and columns
        Plate_Name
        Image_Metadata_WellID
        Image_Metadata_FieldID
        <cell_id_column>

    Retrieve image information from
        <Plate_Name>_Per_Image
        <Plate_Name>_Per_Cell

    Return a DataFrame for each (cell, dye) with columns
        Plate_Name
        Image_Metadata_PlateID
        Image_Metadata_WellID
        Image_Metadata_FieldID
        ImageNumber
        Dye
        Image_FileName
        Image_Height
        Image_Width
        <key_object>_Number_Object_Number
        <key_object>_AreaShape_Center_X,
        <key_object>_AreaShape_Center_Y

    """
    required_columns = [
        'Plate_Name',
        'Image_Metadata_WellID',
        'Image_Metadata_FieldID',
        f'{key_object}_Number_Object_Number']
    for required_column in required_columns:
        if required_column not in cell_ids.columns:
            raise Exception(f"Missing required column {required_column}")

    cell_coordinates = []
    cursor = con.cursor()
    for cell_index in range(cell_ids.shape[0]):
        cell_params = cell_ids.iloc[cell_index]
        if verbose:
            print(f"Getting coordinates for cell and for each dye in [{', '.join(dyes)}]:")
            print(f"   Plate_Name: '{cell_params['Plate_Name']}'")
            print(f"   Image_Metadata_WellID: '{cell_params['Image_Metadata_WellID']}'")
            print(f"   Image_Metadata_FieldID: '{cell_params['Image_Metadata_FieldID']}'")
            print(f"   {key_object}_Number_Object_Number: '{cell_params[f'{key_object}_Number_Object_Number']}'")

        #Image Info
        file_name_fields = [f"Image_FileName_{dye}" for dye in dyes]
        width_fields = [f"Image_Width_{dye}" for dye in dyes]
        height_fields = [f"Image_Height_{dye}" for dye in dyes]
        query = f"""
            SELECT
                image.Image_Metadata_PlateID,
                image.ImageNumber,
                image.{", ".join(file_name_fields)},
                image.{", ".join(width_fields)},
                image.{", ".join(height_fields)},
                key_object.{key_object}_AreaShape_Center_X,
                key_object.{key_object}_AreaShape_Center_Y
            FROM
                {f"{cell_params['Plate_Name']}_Per_Image"} AS image,
                {f"{cell_params['Plate_Name']}_Per_{key_object}"} AS key_object
            WHERE
                Image_Metadata_WellID = '{cell_params['Image_Metadata_WellID']}' AND
                Image_Metadata_FieldID = '{cell_params['Image_Metadata_FieldID']}' AND
                key_object.ImageNumber = image.ImageNumber AND
                key_object.{key_object}_Number_Object_Number = {cell_params[f'{key_object}_Number_Object_Number']};
        """
        cursor.execute(query)
        values = cursor.fetchone()

        for dye_index, dye in enumerate(dyes):
            cell_coordinates.append(dict(
                cell_params.to_dict(), **{
				"Image_Metadata_PlateID" : values[0],
				"ImageNumber" : values[1],
				"Dye" : dye,
                "Dye_Number" : dye_index + 1,
				"Image_FileName" : values[2 + dye_index],
				"Image_Width" : values[2 + len(dyes) + dye_index],
				"Image_Height" : values[2 + 2*len(dyes) + dye_index],
				f"{key_object}_AreaShape_Center_X" : values[2 + 3*len(dyes)],
				f"{key_object}_AreaShape_Center_Y" : values[2 + 3*len(dyes) + 1]}))
    cursor.close()
    cell_coordinates = pd.DataFrame(cell_coordinates)
    return cell_coordinates


def retrieve_image_from_S3(
        S3_region,
        S3_bucket,
        S3_key,
        verbose):
    """
    Retrieve an image from S3://<bucket>.S3.[<region>.]amazonaws.com/<S3_key>
    and load using PIL Image library
    """

    # for public S3 buckets, the region is not included as a subdomain in the URI
    if verbose:
        print(f"Retriving image from url: url S3://{S3_bucket}.s3.{S3_region}.amazonaws.com/{S3_key}")
    S3_resource = boto3.resource('s3', region_name=S3_region)
    S3_bucket = S3_resource.Bucket(S3_bucket)
    S3_object = S3_bucket.Object(S3_key)
    try:
        response = S3_object.get()
    except Exception as exception:
        print(f"Unrecognized S3 key with url: 'S3://{S3_bucket}.s3.{S3_region}.amazonaws.com/{S3_key}")
        import pdb
        pdb.set_trace()
        raise(exception)
    image = PIL.Image.open(response['Body'], mode='r')
    return image


def retrieve_cell_images_from_S3(
        S3_region,
        S3_bucket,
        S3_key_template,
        cell_coordinates,
        verbose=False):
    cell_images = []
    for cell_index in range(cell_coordinates.shape[0]):
        coords = cell_coordinates.iloc[cell_index]
        S3_key = S3_key_template.format(**coords)
        image = retrieve_image_from_S3(S3_region, S3_bucket, S3_key, verbose)
        cell_images.append(image)
    return cell_images

def crop_image(
        image,
        image_width,
        image_height,
        center_x,
        center_y,
        width,
        height):
    xmin = max(0.0, center_x - width/2)
    xmax = min(image_width, center_x + width/2)
    ymin = max(0.0, center_y - height/2)
    ymax = min(image_height, center_y + height/2)
    return image.crop((xmin, ymin, xmax, ymax))


def style_image(
        image,
        saturation=1.0,
        color_map=None):
    """
    add colormaps to images
    """
    image = image.point(lambda pixel: pixel * saturation)
    if color_map:
        color_map = mpl.cm.get_cmap(color_map)
        image = np.array(image)
        image = color_map(image)
        image = np.uint8(image * 255)
        image = PIL.Image.fromarray(image)
    return image

def montage_images(
        images,
        labels,
        width,
        height,
        border_top=0,
        border_bottom=0,
        verbose=False):

    if verbose:
        print(f"Making a montage of {len(images)} images:")
    montage = PIL.Image.new('RGB', (width, len(images)*height + border_top + border_bottom))
    top = border_top
    left = 0
    for image_index, image in enumerate(images):
        if verbose:
            print(f"   pasting image {labels[image_index]} at ({left}, {top})")
        montage.paste(image, (left, top))
        top = top + height

    draw_labels = PIL.ImageDraw.Draw(montage)
    top = 0
    left = 0
    for label in labels:
        draw_labels.text((left, top+height-10), label, fill=('white'))
        top = top + height

    return montage


def view_cells(
        cell_ids,
        database_connection_info,
        database_options_group,
        S3_region,
        S3_bucket,
        S3_key_template,
        key_object,
        dyes,
        saturations,
        color_maps,
        width,
        height,
        verbose=False):
    """
    Retrieve images and make them easy to see what what's going

    Example options for view_cells for the SARS-CoV-2 pseudo time experiment

        view_cells(
            database_connection_info="/home/ubuntu/.mysql/connectors.cnf",
            database_options_group="covid19cq1",
            S3_region="us-east-1",
            S3_bucket="umich-insitro",
            S3_key_template="{S3_path}/W{Image_Metadata_WellID}F{Image_Metadata_FieldID}T0001Z000C{Dye_Number}.tif",
            key_object="Nuclei",
            dyes=["Hoe", "NP", "ConA", "Spike"],
            saturations=[0.20, 0.50, 0.15, 0.4],
            color_maps=['Blues', 'Greens', 'viridis', 'inferno'],
            width=150,
            height=150,
            verbose=True)

    """
    if verbose:
        print(f"Retriving information about the images for {len(cell_ids)} cells from the database...")

    if not os.path.exists(database_connection_info):
        print("'database_connection_info' path does not exist. This is typically something like '/home/ubuntu/.mysql/connectors.cnf'. See 'https://dev.mysql.com/doc/refman/8.0/en/option-files.html' for more inforamtion")

    con = mysql.connector.connect(
        option_files=database_connection_info,
        option_groups=database_options_group)

    cell_coordinates = retrieve_cell_coordinates_from_db(
        con=con,
        cell_ids=cell_ids,
        key_object=key_object,
        dyes=dyes,
        verbose=verbose)

    if verbose:
        print(f"Retrieving the images from the S3 bucket...")
    dye_images = retrieve_cell_images_from_S3(
        S3_region=S3_region,
        S3_bucket=S3_bucket,
        S3_key_template=S3_key_template,
        cell_coordinates=cell_coordinates,
        verbose=verbose)

    n_dyes = len(dyes)
    if verbose:
        print(f"Assembling the images for {n_dyes} different dye images into a motage for each cell...")
    montages = []
    for cell_index in range(len(cell_ids)):
        cell_images = []
        if verbose:
            print(f"  making image for cell {cell_index}")
        for dye_index in range(len(dyes)):
            coords = cell_coordinates.iloc[cell_index*n_dyes + dye_index]
            dye_image = dye_images[cell_index*n_dyes + dye_index]

            dye_image = crop_image(
                image=dye_image,
                image_width=coords['Image_Width'],
                image_height=coords['Image_Height'],
                center_x=coords[f'{key_object}_AreaShape_Center_X'],
                center_y=coords[f'{key_object}_AreaShape_Center_Y'],
                width=width,
                height=height)

            dye_image = style_image(
                dye_image,
                saturation=saturations[dye_index],
                color_map=color_maps[dye_index])
            cell_images.append(dye_image)
        cell_images = montage_images(
            images=cell_images,
            labels=dyes,
            width=width,
            height=height,
            verbose=verbose)
        montages.append(cell_images)

    con.close()
    return montages


def collate_cell_instances(
        cell_ids,
        group_dimension,
        group_values,
        n_instances_per_group,
        output_fname,
        image_config,
        image_cell_height=470,
        image_cell_width=21,
        verbose=False):

    # check inputs
    output_path = os.path.basename(output_fname)
    if not os.path.exists(output_path):
        if verbose:
            print(f"Output path '{output_path}' does not exist, creating...")
        os.makedirs(output_path)

    if group_dimension not in cell_ids.columns:
        print(f"Group dimension {group_dimension} is not a column of the cell_ids: [{cell_ids.columns.join(', ')}].")

    workbook = xlsxwriter.Workbook(output_fname)
    image_worksheet = workbook.add_worksheet("Cell Instances")
    cell_info_worksheet = workbook.add_worksheet("Cell Info")

    # write image worksheet row labels
    image_worksheet.write(0, 0, group_dimension)
    for group_index, group_value in enumerate(group_values):
        row = group_index + 1
        image_worksheet.set_row(row, image_cell_height)
        image_worksheet.write(row, 0, f"{group_value}")

    # write image worksheet column titles
    for instance_index in range(n_instances_per_group):
        column = instance_index + 1
        image_worksheet.write(0, column, f"Instance {instance_index}")
    image_worksheet.set_column(0, n_instances_per_group, image_cell_width)

    # write cell info column labels
    for column, column_name in enumerate(cell_ids.columns):
        cell_info_worksheet.write(0, column, column_name)

    # insert montaged images for cell instances for each group
    for group_index, group_value in enumerate(group_values):
        if verbose:
            print(f"Getting example images for {group_dimension}={group_value}...")
        cell_instances = cell_ids[cell_ids[group_dimension] == group_value] \
            .sample(n_instances_per_group)

        cell_images = cell_images = view_cells(cell_ids=cell_instances, **image_config)
        for instance_index in range(n_instances_per_group):
            image_data = io.BytesIO()
            cell_images[instance_index].save(image_data, format='PNG')
            image_name = f"{group_dimension}={group_value}_instance={instance_index}"
            image_worksheet.insert_image(
                row=group_index + 1,
                col=instance_index + 1,
                filename=image_name,
                options={'image_data':image_data})

            cell_instance = cell_instances.iloc[instance_index]
            for column_index, value in enumerate(cell_instance):
                row_index = 1 + group_index * n_instances_per_group + instance_index
                try:
                    cell_info_worksheet.write(
                        row_index,
                        column_index,
                        value)
                except Exception as exception:
                    print(f"ERROR Writing value '{value}' to cell_info table in cell row={row_index} column={column_index}")
                    print("  " + exception)
    workbook.close()


