import os
import sys
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm

from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene
from pytorch3d.io import IO, save_obj, load_ply
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
import copy

sys.path.insert(0, '/home/ids/liliu/projects/ChatGarment/ContourCraft-CG/')

from utils.close_utils import get_seged_points
from utils.smplx_garment_conversion import deform_garments
from runners.smplx.body_models import SMPLXLayer
import subprocess

argparser = argparse.ArgumentParser()
argparser.add_argument('--garment_path', type=str, default='/home/ids/liliu/projects/ChatGarment/runs/ckpt_model_epoch32_07-07_07_03/close_eva_imgs_img_recon/vis_new/valid_garment_10001_1938', help='Path to the folder containing the results')
argparser.add_argument('--close_dataset', type=str, default='/home/ids/liliu/data/close/CloSe-Di', help='Path to the close dataset')

args = argparser.parse_args()


smplx_layer = SMPLXLayer(
    '/home/ids/liliu/data/body_models/models/smplx/SMPLX_NEUTRAL.pkl',
    ext='pkl',
    num_betas=300
).cuda()



def get_meshes(folder_path):
    mesh_dict = {}
    for subfolder in os.listdir(folder_path):
        garment_path = os.path.join(folder_path, subfolder, subfolder, f'{subfolder}_sim.obj') # mesh_file ended in _sim
        if not os.path.exists(garment_path):
            continue
        mesh = IO().load_mesh(garment_path, load_textures=False)
        mesh_dict[subfolder] = mesh
    meshes_all = list(mesh_dict.values())
    if len(meshes_all) == 0:
        return None
    garment_combined = join_meshes_as_scene(meshes_all)
    mesh_dict['combined'] = garment_combined
    mesh_dict['']

    smplx_params_path = '/home/ids/liliu/projects/ChatGarment/assets/eval/aaa_mesh_registrarion/registered_params.pkl'
    with open(smplx_params_path, 'rb') as f:
        smplx_params = pickle.load(f)
    
    smplx_dict = {
        'betas': torch.tensor(smplx_params['pred_shape'], dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['pred_pose'], dtype=torch.float32).reshape(1, 165).cuda(),
        'transl': torch.tensor(smplx_params['pred_transl'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    return garment_combined.cuda(), smplx_dict

def convert_garments_close(pred_garment_mesh, garment_id, smplx_params_raw, save_folder):
    target_npz_path = os.path.join(
        args.close_dataset, f'{garment_id}.npz'
    )
    target_npz = np.load(target_npz_path)
    pkl_path = '/home/ids/liliu/projects/ChatGarment/assets/eval/smplxn_params.pkl'
    with open(pkl_path, 'rb') as f:
        smplx_data = pickle.load(f)
    smplx_params = smplx_data[garment_id]

    betas = np.zeros(300)
    betas[:16] = smplx_params['betas']
    smplx_params_new = {
        'betas': torch.tensor(betas, dtype=torch.float32).reshape(1, 300).cuda(),
        'poses': torch.tensor(smplx_params['poses'], dtype=torch.float32).reshape(1, 55, 3).cuda(),
        'transl': torch.tensor(smplx_params['trans'], dtype=torch.float32).reshape(1, 3).cuda(),
    }
    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )
    deformed_garment_mesh = Meshes(verts=[deformed_garment_verts], faces=[pred_garment_mesh.faces_packed()])
    IO().save_mesh(deformed_garment_mesh, os.path.join(saved_folder, f'{garment_id}_converted.obj'))
    #Note: how to render it on smplx

if __name__ == '__main__':
    garment_combined, smplx_dict = get_meshes(args.garment_path)
    garment_id = os.path.basename(args.garment_path)[len('valid_garment_'):]
    print("loading garment ", garment_id)
    save_folder = args.garment_path

    convert_garments_close(garment_combined, garment_id, smplx_dict, save_folder)
    