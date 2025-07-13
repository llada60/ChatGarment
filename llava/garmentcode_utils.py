from datetime import datetime
from pathlib import Path
import yaml
import sys
import shutil 
import random
import pickle
import os
import json
import yaml
import copy
import numpy as np
import re

def randomly_replace_extra(text):
    # Define a function to randomly replace with empty list
    def replace_match(match):
        return "'extra': []" if random.choice([True, False]) else match.group(0)

    # Use regex to find all occurrences of "'extra': [xxx]"
    result = re.sub(r"'extra': \[.*?\]", replace_match, text)
    return result


def change_prompt(prompt):
    if 'Can you estimate ' in prompt:
        possible_changes_1 = [
            "Could you tell ",
            "Can you identify ",
            "Would you estimate ",
            "What is "
        ]

        possible_changes_2 = [
            "Please estimate ",
            "Please identify ",
            "Please tell ",
            "Provide ",
            "Return ",
        ]

        randnnum = random.random()
        if randnnum < 0.4:
            prompt = prompt.replace('Can you estimate ', random.choice(possible_changes_1))
        elif randnnum < 0.8:
            prompt = prompt.replace('Can you estimate ', random.choice(possible_changes_2))
            prompt = prompt.replace('?', '.')

    if 'outfit sewing pattern code' in prompt:
        possible_changes_1 = [
            "sewing pattern codes for the garments in the image",
            "outfit sewing pattern codes for the garments",
            "sewing pattern codes for the outfit",
        ]
        if random.random() < 0.5:
            prompt = prompt.replace('outfit sewing pattern code', random.choice(possible_changes_1))

            prompt = prompt + '\nAnswer in the format: {\'upperbody_garment\': (upper garment sewing pattern), \'lowerbody_garment\': (lower garment sewing pattern)} or {\'wholebody_garment\': (wholebody garment sewing pattern)}.'
    
    
    if 'the Json format garment geometry description' in prompt:
        possible_changes_1 = [
            "the json-format garment geometry descriptions",
            "the garment geometry descriptions in Json format",
        ]
        if random.random() < 0.5:
            prompt = prompt.replace('the Json format garment geometry description', random.choice(possible_changes_1))
    
    
    if "Adjust the old sewing pattern according to the text descriptions." in prompt:
        pattern = r"The old garment sewing pattern is: \n(.*?)\. \nAnd the text descriptions are: \n(.+)"

        # Use re.search to find matches
        match = re.search(pattern, prompt, re.DOTALL)

        if match:
            answer_data_old_wfloats = match.group(1)  # Extracts {answer_data_old_wfloats}
            used_text_description = match.group(2)  # Extracts {used_text_description}
            possible_prompts = [
                f"Adjust the sewing pattern according to the text descriptions. Leave all unspecified parts unchanged. The old garment sewing pattern is: \n{answer_data_old_wfloats}. \nThe text descriptions are: \n{used_text_description}.",
                f"Adapt the old sewing pattern based on the descriptions, with old garment sewing pattern to be \n{answer_data_old_wfloats}. \nand the descriptions to be: \n{used_text_description}. Keep the untouched sections as they are.",
                f"Here is an old garment sewing pattern: \n{answer_data_old_wfloats}. \nPlease modify the pattern to align with the text descriptions: \n{used_text_description}. And let all unaddressed elements remain as is.",
                f'Update the old sewing pattern following the text descriptions: \n{used_text_description}. The old garment sewing pattern is: \n{answer_data_old_wfloats}.',
                f"Could you adjust the mentioned part of the old sewing pattern? The old garment sewing pattern is: \n{answer_data_old_wfloats}. \nAnd the target garment part is described as: \n{used_text_description}."
            ]

            if random.random() < 0.8:
                prompt = random.choice(possible_prompts)
        else:
            print("Pattern not found in the text.")
            

    if "Can you describe the geometry features of the garments worn by the model in the Json format?" in prompt:
        possible_changes_1 = [
            "Could you provide the geometric features of the model's garments in JSON format?",
            "Please describe the shape and structure of the outfits on the model using Json format",
            "Can you outline the geometric details of the garments the subject is wearing in JSON format?",
            "Could you specify the geometry of the clothing on the model in a JSON structure?",
            "Provide a Json-format description of the geometric features of the model's oufits."
        ]

        if random.random() < 0.8:
            prompt = prompt.replace('Can you describe the geometry features of the garments worn by the model in the Json format?', random.choice(possible_changes_1))
    
    if 'based on the image' in prompt:
        possible_changes_1 = [
            "according to the image",
            "shown in the image"
        ]
        if random.random() < 0.3:
            prompt = prompt.replace('based on the image', random.choice(possible_changes_1))

    if random.random() < 0.5:
        prompt = randomly_replace_extra(prompt)
        
    return prompt



def recursive_simplify_params(cfg, all_float_paths_dict, parent_path='design'):
    """
    return dict cfg: (key: part name, value: subconfig (deepest one is [SEG]-idx, which is the param idx in all_float_paths__dict [idx<76]))
    """
    cfg_new = {}
    if isinstance(cfg, dict):
        for subpattern_n, subpattern_cfg in cfg.items():
            subconfig = recursive_simplify_params(
                subpattern_cfg, all_float_paths_dict, parent_path=parent_path + '.' + subpattern_n)
            
            cfg_new[subpattern_n] = subconfig
    
    else:
        if cfg == "[SEG]" and parent_path not in all_float_paths_dict:
            print(parent_path)
            
        if cfg == "[SEG]" and parent_path in all_float_paths_dict:
            idx = all_float_paths_dict[parent_path]
            cfg_new = f"[SEG]-{idx:03d}"
        else:
            cfg_new = cfg

    return cfg_new


def extract_seg_indices(text):
    # Find all occurrences of "[SEG]-{idx:03d}" and extract the index as integers
    indices = [int(match.group(1)) for match in re.finditer(r"\[SEG\]-(\d{3})", text)]
    return indices


def change_answer(original_str, all_float_paths_dict, update_seg=False):
    # original_str: {'length': [SEG], 'width': [SEG], 'height': [SEG]}
    # Return: It is [STARTS]{'length': -1, 'width': -1, 'height': -1}[SEG].
    original_str_cp = original_str[:]
    original_str = original_str.replace("'", '"')
    original_str = original_str.replace('[SEG]', '"[SEG]"')
    original_json = eval(original_str.strip().replace('null', 'None').replace('false', 'False').replace('true', 'True'))

    sub_json = []
    names = []

    if "upperbody_garment" in original_json:
        sub_json.append(original_json["upperbody_garment"])
        names.append("upperbody_garment")
    
        sub_json.append(original_json["lowerbody_garment"])
        names.append("lowerbody_garment")

        original_str_cp = original_str_cp.replace("'upperbody_garment': ", "'upperbody_garment': [STARTS]")
        original_str_cp = original_str_cp.replace(", 'lowerbody_garment': ", "[END], 'lowerbody_garment': [STARTS]")
        original_str_cp = original_str_cp[:-1] + "[END]}"
    
    elif "wholebody_garment" in original_json:
        sub_json.append(original_json["wholebody_garment"])
        names.append("wholebody_garment")

        original_str_cp = original_str_cp.replace("'wholebody_garment': ", "'wholebody_garment': [STARTS]")
        original_str_cp = original_str_cp[:-1] + "[END]}"
    
    else:
        sub_json.append(original_json)
        names.append("garment")

        original_str_cp = '[STARTS]' + original_str_cp
        original_str_cp = original_str_cp + "[END]"
    
    json_upds = []
    for name, json_now in zip(names, sub_json):
        json_upd = recursive_simplify_params(json_now, all_float_paths_dict)
        json_upds.append(json_upd)
    
    if "upperbody_garment" in original_str:
        str_upds = json.dumps(json_upds[0])
        indices1 = extract_seg_indices(str_upds)

        str_upds = json.dumps(json_upds[1])
        indices2 = extract_seg_indices(str_upds)

        indices = [indices1, indices2]
    
    elif "wholebody_garment" in original_str:
        str_upds = {
            "wholebody_garment": json_upds[0]
        }
        str_upds = json.dumps(str_upds)
        str_upds = str_upds.replace('"', "'")

        indices = extract_seg_indices(str_upds)
        indices = [indices]
    
    else:
        str_upds = json_upds[0]
        str_upds = json.dumps(str_upds)
        str_upds = str_upds.replace('"', "'")

        indices = extract_seg_indices(str_upds)
        indices = [indices]
    
    possible_starts = [
        "It is ",
        "The sewing pattern is ",
        "Sure, it is "
        "The estimated sewing pattern is ",
        "The sewing pattern config is ",
    ]
    if update_seg:
        original_str_cp = original_str_cp.replace("[SEG]", "-1").replace('[END]', '[SEG]')
    if random.random() < 0.5:
        original_str_cp = random.choice(possible_starts) + original_str_cp

    return original_str_cp, indices

    