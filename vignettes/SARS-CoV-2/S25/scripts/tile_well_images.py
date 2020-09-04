
import math
import PIL
import PIL.ImageDraw
import PIL.ImageOps

frame_fnames = [
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0001T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0001T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0002T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0003T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0004T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0005T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0006T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0007T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0008T0001Z000C1.tif",
    "intermediate_data/CQ1_projection_images/20200427T024129_1999B/Projection/W0001F0009T0001Z000C1.tif"]

# just tile the images as is
def tile_well_images_CQ1(
        frame_fnames,
        output_fname,
        frame_width=2560,
        frame_height=2160):
    well_image = PIL.Image.new('I;16', (frame_width*3, frame_height*3))
    well_image.info = {'compression': 'raw', 'dpi': (96, 96)}
    for frame_index, frame_fname in enumerate(frame_fnames):
        frame_image = PIL.Image.open(frame_fname, mode='r')
        assert frame_image.width == frame_width
        assert frame_image.height == frame_height
        left = (frame_index % 3) * frame_width
        top = math.floor(frame_index / 3) * frame_height
        well_image.paste(frame_image, (left, top))
        well_image.save(output_fname)

plate_id = "20200427T024129_1999B"
local_path = f"intermediate_data/CQ1_projection_images/{plate_id}"
well_id = "0001"

for color_index in range(1, 5):
    frame_fnames = [f"{local_path}/Projection/W{well_id}F000{frame_index}T0001Z000C{color_index}.tif" for frame_index in range(1, 10)]
    output_fname = f"{local_path}/Projection/W{well_id}T0001Z000C{color_index}.tif"
    tile_well_images_CQ1(frame_fnames, output_fname)




    
