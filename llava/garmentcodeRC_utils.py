import sys
import os
import yaml
from pathlib import Path
from collections import OrderedDict
import pickle as pkl
import argparse
import json
import re
import copy
import torch
import numpy as np

wb_config_name = 'waistband'
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
all_skirt_configs = ['skirt', 'flare-skirt', 'godet-skirt', 'pencil-skirt', 'levels-skirt', 'pants']



def ordered(d, desired_key_order):
    return OrderedDict([(key, d[key]) for key in desired_key_order])


def recursive_simplify_params(cfg, is_used=True, unused_configs=[], parent_path='design', device='cpu'):
    # change float to 4 decimal places
    if cfg is None:
        print(parent_path)

    cfg_new = {}
    if ('type' not in cfg) or not isinstance(cfg['type'], str):

        if 'enable_asym' in cfg: ############################################
            enable_asym = bool(cfg['enable_asym']['v'])
            if not enable_asym:
                cfg_new['enable_asym'] = cfg['enable_asym']['v']
                return cfg_new

        if parent_path == 'design.sleeve.cuff' and cfg['type']['v'] is None:
            return {'type': None}

        if parent_path == 'design.left.sleeve.cuff' and cfg['type']['v'] is None:
            return {'type': None}
        
        if parent_path == 'design.pants.cuff' and cfg['type']['v'] is None:
            return {'type': None}
        
        # if parent_path == 'design.sleeve' and cfg['sleeveless']['v']:
        #     return {'type': None}
        
        # if parent_path == 'design.sleeve'
        
        for subpattern_n, subpattern_cfg in cfg.items():
            if (subpattern_n in unused_configs) and ('meta' in cfg):
                continue
            else:
                subconfig = recursive_simplify_params(subpattern_cfg, is_used=is_used, parent_path=parent_path + '.' + subpattern_n, device=device)
            
            cfg_new[subpattern_n] = subconfig
    
    else:
        type_now = cfg['type']
        if type_now == 'float':
            lower_bd = float(cfg['range'][0])
            upper_bd = float(cfg['range'][1])

            float_val = cfg['v']
            float_val_normed = (float_val - lower_bd) / (upper_bd - lower_bd)
            cfg_new = torch.tensor([float_val_normed]).float().to(device)
        
        else:
            cfg_new = cfg['v']

    return cfg_new


def GarmentCodeRC_simplify_params(new_config, device='cpu'):
    if 'design' in new_config:
        new_config = new_config['design']

    ################ get unused_configs
    unused_configs = []
    ub_garment = new_config['meta']['upper']['v']
    if ub_garment is None:
        unused_configs += ['shirt', 'collar', 'sleeve', 'left']
    
    wb_garment = new_config['meta']['wb']['v']
    if not wb_garment:
        unused_configs.append(wb_config_name)
    
    lower_garment = new_config['meta']['bottom']['v']
    assert lower_garment != 'null', (lower_garment)
    if lower_garment is None:
        unused_configs += all_skirt_configs
    else:
        unused_configs += copy.deepcopy(all_skirt_configs)
        unused_configs.remove(skirt_configs[lower_garment])

        if 'base' in new_config[skirt_configs[lower_garment]]:
            base_garment = new_config[skirt_configs[lower_garment]]['base']['v']
            unused_configs.remove(skirt_configs[base_garment])

    new_config = recursive_simplify_params(new_config, is_used=True, unused_configs=unused_configs, device=device)
    
    return new_config


def update_design_ranges():
    return