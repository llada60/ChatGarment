import numpy as np
import torch
import os
import copy
import subprocess
import cv2
import pickle
import sys
import yaml
import argparse
import logging
from easydict import EasyDict as edict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from llava.garment_utils_v2 import try_generate_garments
from llava.garment_lbs_utils import deform_garments
from llava.pytorch3d_render_utils import TexturedIUVRenderer
from llava.model.smplx.body_models import SMPL, SMPLX
from llava.garmentcodeRC_utils import GarmentCodeRC_simplify_params
from pytorch3d.io import IO


DEVICE = 'cuda'
smplx_layer = SMPL(
    '../Pose_estimation/TokenHMR/data/body_models/smpl/SMPL_NEUTRAL.pkl',
    ext='pkl',
).to(DEVICE) 
IMG_LENGTH = 512
Renderer = TexturedIUVRenderer(device=DEVICE, img_wh=IMG_LENGTH)

to_opt_names = [
    'shirt.length',
    'shirt.width',
    'sleeve.length',
    'sleeve.end_width',
    'skirt.length',
    'flare-skirt.length',
    'pencil-skirt.length',
    'pants.length',
    'pants.flare',
]

possible_garment_names = {
    'shirt': 'upper',
    'dress': 'wholebody',
    'skirt': 'lower',
    "pants": 'lower',
    'coat': 'upper',
}

logger = logging.getLogger('evaluation_logger')
logger.setLevel(logging.INFO)

def search_path(parent_path, target_name, suffix):
    all_files = os.listdir(parent_path)
    for file in all_files:
        # if file.endswith(suffix) and target_name in file:
        #     return os.path.join(parent_path, file)
        if file == target_name + suffix:
            return os.path.join(parent_path, file)
    
    print(f'Warning: {target_name} not found in {parent_path}')
    return None

def parse_sam_results(skirt_seg_path, person_seg_path, is_upper_garment=False):
    garment_seg = cv2.imread(skirt_seg_path, cv2.IMREAD_GRAYSCALE)
    person_seg = cv2.imread(person_seg_path, cv2.IMREAD_GRAYSCALE)

    pant_seg = None
    if is_upper_garment:
        pant_seg_path = person_seg_path.replace('person', 'pants')
        if os.path.exists(pant_seg_path):
            pant_seg = cv2.imread(pant_seg_path, cv2.IMREAD_GRAYSCALE)
        
        skirt_seg_path = person_seg_path.replace('person', 'skirt')
        if os.path.exists(skirt_seg_path):
            pant_seg = cv2.imread(skirt_seg_path, cv2.IMREAD_GRAYSCALE)
        
    return garment_seg, person_seg, pant_seg



def evaluate_wrapper_base(opt_params, ori_params_all, saved_dir, name='upd',
                     smplx_params_raw=None, smplx_params_new=None):
    # update new_params
    ori_params_all = merge_opt_params(ori_params_all, opt_params)
    
    # 2d sewing pattern generation
    try_generate_garments(
        None, ori_params_all, name, saved_dir, invnorm_float=True, float_dict=None
    )
    
    json_path = os.path.join(saved_dir, f'valid_garment_{name}', f'valid_garment_{name}_specification.json')
    command = f'python run_garmentcode_sim.py --json_spec_file "{json_path}"'
    subprocess.run(command, shell=True, text=True)
    
    # LBS re-pose
    mesh_path = os.path.join(saved_dir, f'valid_garment_{name}', f'valid_garment_{name}', f'valid_garment_{name}_sim.obj')
    pred_garment_mesh = IO().load_mesh(mesh_path).to(DEVICE)
    deformed_garment_verts = deform_garments(
        smplx_layer, smplx_params_raw, smplx_params_new, pred_garment_mesh, smplx_layer.lbs_weights
    )
    deformed_garment_verts = deformed_garment_verts.unsqueeze(0)
    faces = pred_garment_mesh.faces_padded()

    # rendering, pytorch3d    
    iuv_image = Renderer.forward(
        deformed_garment_verts, faces, cam_t=None, focal_length=smplx_params_new['focal_length'].cpu().numpy()
    )
    
    segmentation = iuv_image[0, :, :, 3]
    segmentation = (segmentation > 1e-6).float()

    return segmentation


def criterion(garment_seg, person_seg, pred_seg_init, pred_seg):
    assert garment_seg.shape == person_seg.shape, (garment_seg.shape, person_seg.shape)
    assert garment_seg.shape == pred_seg.shape, (garment_seg.shape, pred_seg.shape)

    certain_mask = (person_seg < 0.5) | (garment_seg > 0.5)
    certain_mask = certain_mask.float()

    loss1 = torch.square(pred_seg - garment_seg) * certain_mask
    loss1 = loss1.mean()
    loss_reg = torch.square(pred_seg - pred_seg_init.detach()).mean() * 1e-2
    loss = loss1 + loss_reg
    return loss


def flatten_json(y):
    out = {}
 
    def flatten(x, name=''):

        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
 
        else:
            out[name[:-1]] = x
 
    flatten(y)
    return out
 
    

def get_opt_params(pred_dict):
    flatten_json_dict = flatten_json(pred_dict)
    opt_dict = {}
    for name in to_opt_names:
        if name in flatten_json_dict:
            opt_dict[name] = flatten_json_dict[name]
    
    if 'meta.upper' in flatten_json_dict and flatten_json_dict['meta.upper'] == 'FittedShirt':
        opt_dict.pop('shirt.length')
        opt_dict.pop('shirt.width')
    
    if 'sleeve.sleeveless' in flatten_json_dict and flatten_json_dict['sleeve.sleeveless']:
        opt_dict.pop('sleeve.length')
        opt_dict.pop('sleeve.end_width')
    
    if 'shirt.strapless' in flatten_json_dict and flatten_json_dict['shirt.strapless']:
        opt_dict.pop('sleeve.length')
        opt_dict.pop('sleeve.end_width')

    return opt_dict


def merge_opt_params(pred_dict, opt_dict):
    pred_dict_new = copy.deepcopy(pred_dict)
    for k, v in opt_dict.items():
        ks = k.split('.')
        pred_dict_new[ks[0]][ks[1]] = v[0].detach().cpu().numpy()
        
    return pred_dict_new
    
def parse_args():
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgname', type=str, default='000')
    parser.add_argument('--img_dir', type=str, default='/home/ids/liliu/data/ChatGarment/v2/1046/motion_0/imgs/150/sketch')
    parser.add_argument('--inp_pose_params_dir', type=str, default='runs/tokenhmr_output/sketch')
    parser.add_argument('--garmentcode_dir', type=str, default='/home/ids/liliu/projects/ChatGarment/runs/ckpt_model_epoch32_07-07_07_03/sketch_recon/vis_new')
    parser.add_argument('--template_smpl_pkl', type=str, default='/home/ids/liliu/projects/ChatGarment/assets/eval/aaa_mesh_registrarion/registered_params_garmentgenerator_SMPL.pkl')
    parser.add_argument('--saved_dir', type=str, default='runs/postprocess/')
    parser.add_argument('--garment_seg_dir', type=str, default='/home/ids/liliu/projects/ChatGarment/runs/example_eva_SAM/mask')
    args = parser.parse_args()
    
    new_args = edict(
        inp_img=search_path(args.img_dir, args.imgname, '.png'),
        inp_pose_params_path=search_path(args.inp_pose_params_dir, args.imgname, '_0.pkl'),
        template_smpl_pkl=args.template_smpl_pkl,
        person_seg_path=search_path(args.garment_seg_dir, args.imgname, '_person.png'),
    )
    
    garmentcode_dir = os.path.join(args.garmentcode_dir, f'valid_garment_{args.imgname}')
    all_seg_names = os.listdir(args.garment_seg_dir)
    
    segname_list = []
    new_args_list = []
    for filename in all_seg_names:
        img_name = filename.split('_')[:-1]
        img_name = '_'.join(img_name)
        if img_name != args.imgname:
            continue
        
        if 'person' in filename:
            continue
        
        segname = filename.split('_')[-1]
        segname = segname.split('.')[0]
        if segname in possible_garment_names:
            garment_type = possible_garment_names[segname]
            if not os.path.exists(os.path.join(garmentcode_dir, f'valid_garment_{garment_type}')):
                print(os.path.join(garmentcode_dir, f'valid_garment_{garment_type}'))
                continue
            this_args = copy.deepcopy(new_args)

            this_args['garmentcode_path'] = os.path.join(
                garmentcode_dir, f'valid_garment_{garment_type}', 'design.yaml'
            )
            this_args['skirt_seg_path'] = os.path.join(args.garment_seg_dir, f'{args.imgname}_{segname}.png')
            saved_path = os.path.join(args.saved_dir, args.imgname, f'valid_garment_{garment_type}')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
            this_args['saved_path'] = saved_path
            new_args_list.append(this_args)
            segname_list.append(segname)
            
        else:
            print(f'Warning: garment type {segname} not in possible_garment_names')
            continue
    
    print('new_args_list', new_args_list)
    if 'shirt' in segname_list and 'coat' in segname_list:
        print('Warning: shirt and coat both exist, removing shirt')
        shirt_idx = segname_list.index('shirt')
        new_args_list.pop(shirt_idx)
        segname_list.pop(shirt_idx)
    
    if 'pants' in segname_list and 'skirt' in segname_list:
        print('Warning: pants and skirt both exist, removing pants')
        pants_idx = segname_list.index('pants')
        new_args_list.pop(pants_idx)
        segname_list.pop(pants_idx)
    
    os.makedirs(os.path.join(args.saved_dir, args.imgname), exist_ok=True)
    
    file_handler = logging.FileHandler(
        os.path.join(args.saved_dir, args.imgname, 'evaluation_log.txt')
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - Iteration %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return new_args_list

def parse_sam_results(skirt_seg_path, person_seg_path, is_upper_garment=False):
    garment_seg = cv2.imread(skirt_seg_path, cv2.IMREAD_GRAYSCALE)
    person_seg = cv2.imread(person_seg_path, cv2.IMREAD_GRAYSCALE)

    pant_seg = None
    if is_upper_garment:
        pant_seg_path = person_seg_path.replace('person', 'pants')
        if os.path.exists(pant_seg_path):
            pant_seg = cv2.imread(pant_seg_path, cv2.IMREAD_GRAYSCALE)
        
        skirt_seg_path = person_seg_path.replace('person', 'skirt')
        if os.path.exists(skirt_seg_path):
            pant_seg = cv2.imread(skirt_seg_path, cv2.IMREAD_GRAYSCALE)
        
    return garment_seg, person_seg, pant_seg

def FDM(args):
    img = cv2.imread(args.inp_img)
    img = cv2.resize(img, (IMG_LENGTH, IMG_LENGTH))
    img_size = img.shape[:2]

    is_upper_garment = 'valid_garment_upper' in args.garmentcode_path
    garment_seg, person_seg, pant_seg = parse_sam_results(args.skirt_seg_path, args.person_seg_path, is_upper_garment=is_upper_garment)
    garment_seg = cv2.resize(garment_seg, img_size)
    person_seg = cv2.resize(person_seg, img_size)
    
    gt_seg_vis = img * 0.5 + garment_seg.reshape(img_size[0], img_size[1], 1) * 0.5
    cv2.imwrite(os.path.join(args.saved_path, 'gt_seg_vis.png'), gt_seg_vis)
    
    garment_seg = torch.from_numpy(np.array(garment_seg)).squeeze(-1).float().to(DEVICE) / 255.0
    person_seg = torch.from_numpy(np.array(person_seg)).squeeze(-1).float().to(DEVICE) / 255.0
    if pant_seg is not None:
        pant_seg = cv2.resize(pant_seg, img_size)
        pant_seg = torch.from_numpy(np.array(pant_seg)).squeeze(-1).float().to(DEVICE) / 255.0
        
        person_seg = (person_seg > 0.5) & (pant_seg < 0.5)
        person_seg = person_seg.float()
        
    with open(args.inp_pose_params_path, 'rb') as f:
        inp_smpl_params_origin = pickle.load(f)
    
    with open(args.template_smpl_pkl, 'rb') as f:
        template_smpl_params_origin = pickle.load(f)

    inp_smpl_params = {}
    inp_smpl_params['betas'] = torch.from_numpy(inp_smpl_params_origin['pred_smpl_params']['betas']).float().reshape(1, 10).to(DEVICE)
    inp_smpl_params['poses'] = np.concatenate(
        [inp_smpl_params_origin['pred_smpl_params']['global_orient'], inp_smpl_params_origin['pred_smpl_params']['body_pose']], axis=1)
    inp_smpl_params['poses'] = torch.from_numpy(inp_smpl_params['poses']).float().to(DEVICE)
    inp_smpl_params['transl'] = torch.from_numpy(inp_smpl_params_origin['pred_cam_t_full']).float().reshape(1, 3).to(DEVICE)
    inp_smpl_params['focal_length'] = inp_smpl_params_origin['scaled_focal_length'] / 4096 * IMG_LENGTH
    
    template_smpl_params = {}
    template_smpl_params['betas'] = torch.from_numpy(template_smpl_params_origin['pred_shape']).float().reshape(1, 10).to(DEVICE)
    template_smpl_params['poses'] = torch.from_numpy(template_smpl_params_origin['pred_pose']).float().reshape(1, 24, 3).to(DEVICE)
    template_smpl_params['transl'] = torch.from_numpy(template_smpl_params_origin['pred_transl']).float().reshape(1, 3).to(DEVICE)

    evaluate_wrapper = lambda opt_params, configRC, saved_dir, name: evaluate_wrapper_base(opt_params, configRC, saved_dir, name, template_smpl_params, inp_smpl_params)
    
    with open(args.garmentcode_path, 'r') as f:
        config_raw = yaml.safe_load(f)
    
    configRC = GarmentCodeRC_simplify_params(config_raw, device=DEVICE)
    print('configRC', configRC)
    
    opt_params = get_opt_params(configRC)
    print('opt_params', opt_params)
    max_iterations = 30
    
    optimizer = torch.optim.Adam([item for item in opt_params.values()], lr=0.025, betas=(0.5, 0.9))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations, eta_min=0.01)
    
    losses = []
    pred_seg_init = None
    best_loss = 1e10
    for i in range(max_iterations):
        pred_seg = evaluate_wrapper(opt_params, configRC, args.saved_path, 'base')
        if pred_seg_init is None:
            pred_seg_init = pred_seg.detach().clone()
            img_overlap = img * 0.5 + pred_seg.detach().cpu().numpy().reshape(img_size[0], img_size[1], 1) * 0.5 * 255
            cv2.imwrite(os.path.join(args.saved_path, 'img_overlap_init.png'), img_overlap)   
        
        loss_ori = criterion(garment_seg, person_seg, pred_seg_init, pred_seg)
        
        print(f'iter {i}, loss: {loss_ori.detach().cpu().numpy()}')
        img_overlap = img * 0.5 + pred_seg.detach().cpu().numpy().reshape(img_size[0], img_size[1], 1) * 0.5 * 255
        cv2.imwrite(os.path.join(args.saved_path, f'img_overlap_{i}.png'), img_overlap)   
        
        if loss_ori < best_loss:
            best_loss = loss_ori
            best_param = {
                name: param.detach().clone() for name, param in opt_params.items()
            }
        
        optimizer.zero_grad()
        
        for name, param in opt_params.items():
            delta_param = 0.06 if name in ['pants.flare', 'shirt.width', 'sleeve.end_width'] else 0.02
            opt_params_tmp = copy.deepcopy(opt_params)
            opt_params_tmp[name] = param + delta_param
            pred_seg_pert = evaluate_wrapper(opt_params_tmp, configRC, args.saved_path, name)
            loss_pert = criterion(garment_seg, person_seg, pred_seg_init, pred_seg_pert)
            
            gradient = (loss_pert - loss_ori) / delta_param
            print('gradient', gradient, opt_params[name])
            gradient = torch.tensor([gradient], device=DEVICE)
            opt_params[name].grad = gradient
        
        optimizer.step()
        lr_scheduler.step()
        
        for name, param in opt_params.items():
            opt_params[name] = param.clamp_(0, 1)
        
        saved_dict = {k: f'{v.detach().cpu().item():.5f}' for k, v in opt_params.items()}
        logger.info(f'{i}: Result = {saved_dict}')
        
        losses.append(loss_ori)
        
    print(f'Best loss: {best_loss}', 'Best param:', best_param)
    logger.info(f'Best loss: {best_loss}')
    logger.info(f'Best param: {best_param}')
    evaluate_wrapper(best_param, configRC, args.saved_path, 'best')

if __name__ == '__main__':
    args_list = parse_args()
    for args in args_list:
        FDM(args)
    