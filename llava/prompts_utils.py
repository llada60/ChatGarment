import json
import re
from llava.json_fixer import repair_json


def get_text_labels(gpt4o_results):
    result_dict = repair_json(gpt4o_results, return_objects=True)

    used_config_new = {}
    used_config_text = []
    
    if "upper garment" in result_dict:
        used_config_now = {
            'garment_name': result_dict["upper garment"][0],
            'geometry_styles': result_dict["upper garment"][1],
        }
        used_config_new['upperbody_garment'] = used_config_now
        used_config_text.append(
           result_dict["upper garment"][1] + ', ' + result_dict["upper garment"][0]
        )
    
    if "lower garment" in result_dict:
        used_config_now = {
            'garment_name': result_dict["lower garment"][0],
            'geometry_styles': result_dict["lower garment"][1],
        }
        used_config_new['lowerbody_garment'] = used_config_now
        used_config_text.append(
           result_dict["lower garment"][1] + ', ' + result_dict["lower garment"][0]
        )
    
    if "wholebody garment" in result_dict:
        used_config_now = {
            'garment_name': result_dict["wholebody garment"][0],
            'geometry_styles': result_dict["wholebody garment"][1],
        }
        used_config_new['wholebody_garment'] = used_config_now
        used_config_text.append(
            result_dict["wholebody garment"][1] + ', ' + result_dict["wholebody garment"][0]
        )

    return used_config_new, used_config_text



def get_text_labels_detailed(gpt4o_results):
    gpt4o_results = gpt4o_results.strip()
    if "```" in gpt4o_results:
        gpt4o_results = gpt4o_results.split("```")[1]
        gpt4o_results = gpt4o_results.strip()
        if gpt4o_results.startswith('json') or gpt4o_results.startswith('Json') or gpt4o_results.startswith('JSON'):
            gpt4o_results = gpt4o_results[4:].strip()
    
    results = repair_json(gpt4o_results, return_objects=True)
    if len(results) < 2:
        return None

    if isinstance(results[1], str):
        try:
            results[1] = eval(results[1])
        except:
            print('????')
            return None

    if len(results) > 2:
        extra_styles = results[2].split(',')
        results[1]['extra'] = [item.strip() for item in extra_styles]
    else:
        results[1]['extra'] = []
        
    used_config_now = {
        'garment_name': results[0],
        'geometry_styles': results[1],
    }

    return used_config_now


def get_text_labels_foredit(gpt4o_results):
    gpt4o_results = gpt4o_results.strip()
    if "```" in gpt4o_results:
        gpt4o_results = gpt4o_results.split("```")[1]
        gpt4o_results = gpt4o_results.strip()
        if gpt4o_results.startswith('json') or gpt4o_results.startswith('Json') or gpt4o_results.startswith('JSON'):
            gpt4o_results = gpt4o_results[4:].strip()
    
    results = repair_json(gpt4o_results, return_objects=True)
    return results


def get_gpt4o_textgen_prompt(garment_name, garment_description):
    txtgen_prompt_path = 'docs/detailed_textbased_description.txt'
    with open(txtgen_prompt_path, 'r') as f:
        txtgen_prompt = f.read()
    
    txtgen_prompt = txtgen_prompt.replace('[TYPE]', garment_name)
    txtgen_prompt = txtgen_prompt.replace('[DESCRIPTION]', garment_description)
    return txtgen_prompt
    

def get_gpt4o_edit_prompt(garment_name, prompt):
    edit_prompt_path = 'docs/prompt_garment_editing.txt'
    with open(edit_prompt_path, 'r') as f:
        edit_prompt = f.read()
    
    edit_prompt = edit_prompt.replace('[TYPE]', garment_name)
    edit_prompt = edit_prompt.replace('[DESCRIPTION]', prompt)
    return edit_prompt

