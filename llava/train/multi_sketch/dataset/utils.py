
import os
import random
import numpy as np
from PIL import Image


def autocrop(pil_img, matrix_HW_pct_range=[0.4, 0.95]):
    """
    random crop from an input image
    """
    img_focus = pil_img
    x_focus, y_focus = img_focus.size


    matrix_x_pct = random.uniform(matrix_HW_pct_range[0], matrix_HW_pct_range[1])
    matrix_x = round(matrix_x_pct*x_focus)
    matrix_y_pct = random.uniform(matrix_HW_pct_range[0], matrix_HW_pct_range[1])
    matrix_y = round(matrix_y_pct*y_focus)

    if matrix_x < 10 or matrix_y < 10:
        return pil_img

    x1 = random.randrange(0, x_focus - matrix_x)
    y1 = random.randrange(0, y_focus - matrix_y)
    cropped_img = img_focus.crop((x1, y1, x1 + matrix_x, y1 + matrix_y))
    
    return cropped_img

def convert_rgba_to_rgb_with_white_bg(image_path):
    """Convert an RGBA image to an RGB image with a white background."""
    if not os.path.exists(image_path):
        image_path = image_path[:-4] + '.gif'

    # Open the image
    image = Image.open(image_path)

    # Check if the image is in RGBA mode
    if image.mode == 'RGBA':
        # Create a white background image of the same size as the original
        white_background = Image.new("RGB", image.size, (255, 255, 255))
        
        # Paste the RGBA image onto the white background, using the alpha channel as the mask
        white_background.paste(image, mask=image.split()[3])  # Split to get the alpha channel
        image = white_background
    else:
        # If not RGBA, save it directly as an RGB image
        image = image.convert("RGB")
    
    return image

def generate_all_float_labels(all_floats, indices):
    if isinstance(all_floats[0], list) or len(indices) > 1:
        if not isinstance(all_floats[0], list):
            all_floats = [all_floats]
        if len(indices) == 2 and len(all_floats) == 1:
            # debug
            assert len(all_floats[0]) == len(indices[0]) + len(indices[1]), (len(all_floats[0]), len(indices[0]), len(indices[1]))
            all_floats_new = [
                [all_floats[0][i] for i in range(len(indices[0]))],
                [all_floats[0][i] for i in range(len(indices[0]), len(indices[0]) + len(indices[1]) )]
            ]
            all_floats = all_floats_new
        all_floats_padded = []
        all_floats_weight_padded = []
        for i, item in enumerate(all_floats):
            indices_i = indices[i]
            assert len(item) == len(indices_i), (len(item), len(indices_i), i, all_floats)
            all_floats_padded_tmp = np.zeros(76)
            all_floats_padded_tmp[indices_i] = np.array(item)
            all_floats_padded.append(all_floats_padded_tmp)

            all_floats_padded_flag = np.zeros(76)
            all_floats_padded_flag[indices_i] = 1
            all_floats_weight_padded.append(all_floats_padded_flag)
            
        all_floats = np.array(all_floats_padded).reshape(-1)
        all_floats_weight = np.array(all_floats_weight_padded).reshape(-1)
    
    else:
        all_floats_padded = np.zeros(76)
        indices_i = indices[0]
        assert len(indices_i) == len(all_floats), (len(all_floats), len(indices_i),  all_floats)
        all_floats_padded[indices_i] = np.array(all_floats)
        all_floats_padded_flag = np.zeros(76)
        all_floats_padded_flag[indices_i] = 1

        all_floats = np.array(all_floats_padded).reshape(-1)
        all_floats_weight = np.array(all_floats_padded_flag).reshape(-1)
    
    return all_floats, all_floats_weight