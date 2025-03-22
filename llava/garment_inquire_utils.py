import os
import sys

import random
import cv2
import numpy as np
from tqdm import tqdm
import json
import pickle as pkl
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from llava.json_fixer import repair_json

skirt_configs =  {
    'SkirtCircle': 'flare-skirt',
    'AsymmSkirtCircle': 'flare-skirt',
    'GodetSkirt': 'godet-skirt',
    'Pants': 'pants',
    'Skirt2': 'skirt',
    'SkirtManyPanels': 'flare-skirt',
    'PencilSkirt': 'pencil-skirt',
    'SkirtLevels': 'levels-skirt',
}

all_skirt_names = ['SkirtCircle', 'AsymmSkirtCircle', 'GodetSkirt', 'Skirt2', 'SkirtManyPanels', 'PencilSkirt', 'SkirtLevels']

changed_descriptions = {
    'shirt': 'upper garment body',
    'collar': 'collar',
    'sleeve': 'sleeves',
    'sleeve_cuff': 'sleeve cuffs',

    'skirt': 'skirt',
    
    'pants': 'pants',
    'pants_cuff': 'pants cuffs',
    'overall': 'garment body',
}

all_possible_changes = list(changed_descriptions.keys())


donot_descriptions = {
    'shirt': 'sleeves, necklines, collars',
    'collar': 'sleeves, shirt torsos',
    'sleeve': 'collar, shirt torsos',
    'sleeve_cuff': 'collar, shirt torsos',

    'skirt': 'waistbands, upperbody garments',
    
    'pants': 'waistbands, upperbody garments',
    'pants_cuff': 'waistbands, upperbody garments',
    'overall': 'sleeves, necklines, collars',
}


def get_extra_config(change_name):
    if change_name in ['shirt', 'collar', 'sleeve']:
        require_dict = {
            'meta.upper': ['Shirt', 'FittedShirt'],
        }

    if change_name in ['sleeve_cuff']:
        require_dict = {
            'meta.upper': ['Shirt', 'FittedShirt'],
            'sleeve.sleeveless': [False]
        }

    if change_name in ['skirt']:
        require_dict = {
            'meta.bottom': all_skirt_names,
        }
    
    if change_name in ['pants', 'pants_cuff']:
        require_dict = {
            'meta.bottom': ['Pants'],
        }
    
    if change_name in ['overall']:
        require_dict = {
            'meta.upper': ['Shirt', 'FittedShirt'],
            'meta.wb': ['StraightWB', 'FittedWB', None],
            'meta.bottom': all_skirt_names,
        }

    return require_dict



def proces_gpt_labels(gpt_labels_str, changed_name):
    gpt_labels = repair_json(gpt_labels_str, return_objects=True)
    if not isinstance(gpt_labels, list):
        return None, False

    label_dict = {changed_name: gpt_labels}
    return label_dict, True


prompt_init = open("docs/prompt_garment_part_inference.txt", "r").read()
def ask_part_question(change_name, client, base64_image):
    change_name_desc = changed_descriptions[change_name]
    donot_name = donot_descriptions[change_name]

    prompt = prompt_init[:].replace("[PART]", change_name_desc)
    prompt = prompt.replace("[DONOT]", donot_name)

    # try:
    if True:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {   
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        result = response.choices[0].message.content
        result, flag = proces_gpt_labels(result, changed_descriptions[change_name])

    # except Exception as e:
    #     print(e)
    #     result = ''
    #     flag = False
    
    return result, flag

def get_value_from_dict(data, path):
    keys = path.split(".")
    value = data
    for key in keys:
        value = value[key]
    return value

def check_satisfy_requirement(config, require_dict):
    for k, vs in require_dict.items():
        value = get_value_from_dict(config, k)
        # print('value ?', require_dict, k, value, vs)
        if value not in vs:
            return False
    
    return True


def ask_all_parts_questions(garment_sewing, client, base64_image, is_upperbody=False):
    answer_dict = {}
    for change_name in all_possible_changes:
        if is_upperbody:
            if change_name in ['skirt', 'pants', 'pants_cuff', 'overall']:
                continue
        extra_config = get_extra_config(change_name)
        
        change_name_flag = check_satisfy_requirement(garment_sewing, extra_config)
        if change_name_flag:
            # print(change_name, extra_config, change_name_flag)
            result, flag = ask_part_question(change_name, client, base64_image)
            print(result)
            if not flag:
                break

            answer_dict.update(result)
    
    text_description = json.dumps(answer_dict)
    text_description = text_description.replace('"', "'")

    return text_description


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == "__main__":
    from openai import OpenAI
    client = OpenAI()

    img_path = "runs/valid_garment/gt_image.png"

    json_str = "{'upperbody_garment': {'meta': {'upper': 'FittedShirt', 'wb': None, 'bottom': None}, 'collar': {'f_collar': 'Bezier2NeckHalf', 'b_collar': 'CircleNeckHalf', 'width': 0.388671875, 'fc_depth': 0.0859375, 'bc_depth': 0.1015625, 'fc_angle': 109, 'bc_angle': 90, 'f_bezier_x': 0.3046875, 'f_bezier_y': 0.486328125, 'b_bezier_x': 0.1708984375, 'b_bezier_y': 0.1474609375, 'f_flip_curve': False, 'b_flip_curve': False, 'component': {'style': 'Turtle', 'depth': 5, 'lapel_standing': False, 'hood_depth': -0.003753662109375, 'hood_length': 0.1279296875}}, 'left': {'enable_asym': False}, 'shirt': {'strapless': False, 'length': 0.224609375, 'width': 0.32421875, 'flare': 0.439453125}, 'sleeve': {'sleeveless': False, 'armhole_shape': 'ArmholeCurve', 'length': 0.66015625, 'connecting_width': 0.1123046875, 'end_width': 0.4140625, 'sleeve_angle': 14, 'opening_dir_mix': 0.57421875, 'standing_shoulder': False, 'standing_shoulder_len': 0.73828125, 'connect_ruffle': -0.0286865234375, 'smoothing_coeff': 0.4921875, 'cuff': {'type': 'CuffSkirt', 'top_ruffle': 0.419921875, 'cuff_len': 0.06787109375, 'skirt_fraction': 0.490234375, 'skirt_flare': 0.62890625, 'skirt_ruffle': 0.10986328125}}}, 'lowerbody_garment': {'meta': {'upper': None, 'wb': 'StraightWB', 'bottom': 'PencilSkirt'}, 'pencil-skirt': {'length': 0.58203125, 'rise': 0.8125, 'flare': 0.46875, 'low_angle': 0, 'front_slit': -0.06298828125, 'back_slit': 0.02099609375, 'left_slit': -0.007537841796875, 'right_slit': -0.0040283203125, 'style_side_cut': None, 'style_side_file': 'assets\\img\\test_shape.svg'}, 'waistband': {'waist': 0.003509521484375, 'width': 0.1337890625}}}"
    json_output = repair_json(json_str, return_objects=True)['upperbody_garment']
    base64_image = encode_image(img_path)
    
    text_description = ask_all_parts_questions(json_output, client, base64_image)
    print(text_description)

