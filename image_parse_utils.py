#!/usr/bin/env python
#
# file: image_parse_utils.py
#
# revision history:
# 20241215 (YQ): first version
#
# description:
#  This script implements image parsing functions. It uses nedc_image_tools to
#  read an image once, generates subwindow coordinates, and slices the windows
#  accordingly.
#

#!/usr/bin/env python

import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import nedc_image_tools

__FILE__ = os.path.basename(__file__)

GRID_FRAME_SIZE = (128, 128)
BLOCK_WINDOW_SIZE = (256, 256)
SHIFT_VALUE = (256 // 2) - (128 // 2)

def generate_subwin_coords(img_height: int, img_width: int, step_size: tuple) -> list:
    """
    function: generate_top_left_coordinates

    arguments:
     height: the height of the image
     width: the width of the image
     frame_size: a tuple specifying the stepping size in x and y directions

    return:
     return_list: a list of (top_left_x, top_left_y) tuples

    description:
     This function creates a list of top-left pixel coordinates for subwindows
     extracted from the image. Each subwindow has size WINDOW_SIZE (256, 256).
     The difference (DIFFERENCE) ensures subwindows are properly aligned.
    """
    return_list = []
    for coord_x in range(SHIFT_VALUE, img_width - BLOCK_WINDOW_SIZE[0] + SHIFT_VALUE, step_size[0]):
        for coord_y in range(SHIFT_VALUE, img_height - BLOCK_WINDOW_SIZE[1] + SHIFT_VALUE, step_size[1]):
            return_list.append((coord_y - SHIFT_VALUE, coord_x - SHIFT_VALUE))
    return return_list

def crop_section(arg_tuple):
    """
    function: crop_image

    arguments:
     input_tuple: a tuple (shared_array, (top_left_x, top_left_y)) used to slice

    return:
     a NumPy array representing the cropped subwindow

    description:
     This function extracts a subwindow from shared_array based on coordinate
     (top_left_x, top_left_y). The resulting subwindow dimension is WINDOW_SIZE
     (256, 256).
    """
    arr_shared, coords = arg_tuple
    px, py = coords
    return arr_shared[px : px + BLOCK_WINDOW_SIZE[0],
                      py : py + BLOCK_WINDOW_SIZE[1]].copy()

def get_frames(image_path, output_list, index_key, batch_size=4000):
    """
    function: get_frames

    arguments:
     image_path: a string specifying the path to the image
     output: a shared list (or array) storing the output windows
     image_index: index into the shared output structure

    return:
     none

    description:
     This function loads an image using nedc_image_tools, generates top-left
     coordinates, and reads subwindows in parallel via read_data_multithread.
     The subwindows are stored into output[image_index].
    """
    img_reader = nedc_image_tools.Nil()
    img_reader.open(image_path)
    w_val, h_val = img_reader.get_dimension()
    coord_bank = generate_subwin_coords(h_val, w_val, GRID_FRAME_SIZE)

    def chunk_coords():
        for start_idx in range(0, len(coord_bank), batch_size):
            yield coord_bank[start_idx : start_idx + batch_size]

    collected_pieces = []
    for chunk_set in chunk_coords():
        subwin_batch = img_reader.read_data_multithread(
            chunk_set,
            npixy=BLOCK_WINDOW_SIZE[1],
            npixx=BLOCK_WINDOW_SIZE[0],
            color_mode="RGB",
            num_threads=96
        )
        collected_pieces.extend(subwin_batch)

    output_list[index_key] = np.array(collected_pieces)

#
# end of file
#
