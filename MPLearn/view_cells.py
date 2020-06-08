# -*- tab-width:4;indent-tabs-mode:nil;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:

import os
import PIL
import PIL.ImageDraw
import PIL.ImageOps
import boto3
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib as mpl





def retrieve_cell_coordinates_from_db(
        con,
        cell_ids,
        dyes,
        verbose=False):
    """
    cell_ids is a pd.DataFrame with rows representing cells and columns
        Plate_Name
        Image_Metadata_WellID
        Image_Metadata_FieldID
        Cells_Number_Object_Number

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
        Cells_Number_Object_Number
        Cells_AreaShape_Center_X,
        Cells_AreaShape_Center_Y

    """
    required_columns = ['Plate_Name', 'Image_Metadata_WellID', 'Image_Metadata_FieldID']
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
            print(f"   Cells_Number_Object_Number: '{cell_params['Cells_Number_Object_Number']}'")

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
                cell.Cells_AreaShape_Center_X,
                cell.Cells_AreaShape_Center_Y
            FROM
                {f"{cell_params['Plate_Name']}_Per_Image"} AS image,
                {f"{cell_params['Plate_Name']}_Per_Cells"} AS cell
            WHERE
                Image_Metadata_WellID = '{cell_params['Image_Metadata_WellID']}' AND
                Image_Metadata_FieldID = '{cell_params['Image_Metadata_FieldID']}' AND
                cell.ImageNumber = image.ImageNumber AND
                cell.Cells_Number_Object_Number = '{cell_params['Cells_Number_Object_Number']}';
        """
        cursor.execute(query)
        values = cursor.fetchone()

        for dye_index, dye in enumerate(dyes):
            cell_coordinates.append(dict(
                cell_params.to_dict(), **{
				"Image_Metadata_PlateID" : values[0],
				"ImageNumber" : values[1],
				"Dye" : dye,
				"Image_FileName" : values[2 + dye_index],
				"Image_Width" : values[2 + len(dyes) + dye_index],
				"Image_Height" : values[2 + 2*len(dyes) + dye_index],
				"Cells_AreaShape_Center_X" : values[2 + 3*len(dyes)],
				"Cells_AreaShape_Center_Y" : values[2 + 3*len(dyes) + 1]}))
    cursor.close()
    cell_coordinates = pd.DataFrame(cell_coordinates)
    return cell_coordinates


def retrieve_image_from_S3(
        S3_region,
        S3_bucket,
        S3_key,
        verbose):
    """
    Retrieve an image from S3://<bucket>.S3.<region>.amazonaws.com/<S3_key>
    and load using PIL Image library
    """
    if verbose:
        print(f"Retriving image from url: url S3://{S3_bucket}.s3.{S3_region}.amazonaws.com/{S3_key}")
    try:
        S3_resource = boto3.resource('s3', region_name=S3_region)
        S3_bucket = S3_resource.Bucket(S3_bucket)
        S3_object = S3_bucket.Object(S3_key)
        response = S3_object.get()
        image = PIL.Image.open(response['Body'], mode='r')
    except:
        raise Exception(f"Unable to locate image in S3 at url S3://{S3_bucket}.s3.{S3_region}.amazonaws.com/{S3_key}")
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
        con,
        S3_region,
        S3_bucket,
        S3_key_template,
        cell_ids,
        dyes,
        saturations,
        color_maps,
        width,
        height,
        verbose=False):
    """
    Retrieve images and make them easy what's going to look at
    """

    if verbose:
        print(f"Retriving information about the images for {len(cell_ids)} cells from the database...")
    cell_coordinates = retrieve_cell_coordinates_from_db(
        con=con,
        cell_ids=cell_ids,
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
                center_x=coords['Cells_AreaShape_Center_X'],
                center_y=coords['Cells_AreaShape_Center_Y'],
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
    return montages
