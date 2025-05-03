# convert the garment to the target pose & shape import random
import random
import pickle as pk
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather
from pytorch3d.ops.laplacian_matrices import cot_laplacian

def compute_face_normals(meshes):
    verts_packed = meshes.verts_padded()
    faces_packed = meshes.faces_packed()
    verts_packed = verts_packed[faces_packed]
    face_normals = torch.cross(
        verts_packed[:, 1] - verts_packed[:, 0],
        verts_packed[:, 2] - verts_packed[:, 0],
        dim=1,
    )
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    return face_normals

        
def get_high_conf_indices(mesh_garment, mesh_smpl, max_distance=0.05, max_angle=25.0):
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)

    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    garment_data = {
        "position": verts_garment,
        "normal": garment_normals.reshape(-1, 3)
    }
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3),
        "normal": closest_normal.reshape(-1, 3),
    }
    high_confidence_flag = filter_high_confidence_matches_torch(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    return high_confidence_flag


def get_high_conf_indices_knn(mesh_garment, mesh_smpl, max_distance=0.05, max_angle=25.0):
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)

    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    garment_data = {
        "position": verts_garment,
        "normal": garment_normals.reshape(-1, 3)
    }
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3),
        "normal": closest_normal.reshape(-1, 3),
    }
    high_confidence_flag = filter_high_confidence_matches_torch(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    return high_confidence_flag, (dists, indices, closest_pos)



def get_best_skinning_weights(mesh_garment, mesh_smpl, lbs_weights, max_distance=0.05, max_angle=25.0):
    # single-mesh operation, batch = 1
    garment_normals = compute_vertex_normals(mesh_garment)
    smpl_normals = compute_vertex_normals(mesh_smpl)
    
    verts_garment = mesh_garment.verts_packed()
    verts_smpl = mesh_smpl.verts_packed()
    num_verts = verts_garment.shape[0]
        
    dists, indices, closest_pos = knn_points(verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), K=1, return_nn=True)
    
    # indices: 1 x N x 1
    closest_normal = knn_gather(smpl_normals.unsqueeze(0), indices) # 1 x N x 1 x 3
    
    smpl_based_lbs = diffuse_lbs_to_space2(
        verts_garment.unsqueeze(0), verts_smpl.unsqueeze(0), lbs_weights.unsqueeze(0)
    )
    
    threshold_distance = calculate_threshold_distance(verts_garment, max_distance)
    threshold_distance = threshold_distance.cpu().numpy()
    
    L, inv_area = cot_laplacian(verts_garment, mesh_garment.faces_packed())
    L_sum = torch.sparse.sum(L, dim=1).to_dense().cpu().numpy()

    L = L.coalesce().cpu()
    L_indices = L.indices().numpy()
    L_values = L.values().numpy()
    L_size = L.size()

    L = sp.coo_matrix((L_values, (L_indices[0], L_indices[1])), shape=L_size).tocsr()
    L = L - sp.diags(L_sum, offsets=0)
    inv_area = inv_area.reshape(-1).cpu().numpy()
    
    ################################################################################################
    garment_data = {
        "position": verts_garment.cpu().numpy(),
        "normal": garment_normals.reshape(-1, 3).cpu().numpy(),
    }
    
    closest_points_data = {
        "position": closest_pos.reshape(-1, 3).cpu().numpy(),
        "normal": closest_normal.reshape(-1, 3).cpu().numpy(),
    }
    high_confidence_indices = filter_high_confidence_matches(
        garment_data, closest_points_data, threshold_distance, max_angle
    )
    
    low_confidence_indices = list(
        set(range(num_verts)) - set(high_confidence_indices)
    )
    
    smpl_based_lbs = torch.clamp(smpl_based_lbs, 0.0, 1.0)
    smpl_based_lbs = smpl_based_lbs / (smpl_based_lbs.sum(dim=-1, keepdim=True) + 1e-6)
    
    smpl_based_lbs_np = smpl_based_lbs[0].cpu().numpy() # N x lbs_weight

    final_lbs_weight = np.zeros((num_verts, smpl_based_lbs_np.shape[1]))
    final_lbs_weight[high_confidence_indices] = smpl_based_lbs_np[high_confidence_indices]

    try:
        final_lbs_weight = do_inpainting(
            high_confidence_indices, low_confidence_indices, final_lbs_weight, L, inv_area)
    except:
        final_lbs_weight = smpl_based_lbs_np
    
    return final_lbs_weight, smpl_based_lbs_np


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces_packed[:, 0], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 1], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 2], faces_normals)
    
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )


def calculate_threshold_distance(verts_garment, threadhold_ratio=0.05):
    """Returns dbox * 0.05

    dbox is the target mesh bounding box diagonal length.
    """
    
    length = verts_garment.max(dim=0)[0] - verts_garment.min(dim=0)[0]
    length = torch.norm(length)

    threshold_distance = length * threadhold_ratio

    return threshold_distance


def filter_high_confidence_matches(target_vertex_data, closest_points_data, max_distance, max_angle):
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = np.linalg.norm(source_positions - target_positions, axis=1)

    # Calculate angles between normals (vectorized)
    cos_angles = np.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles /= np.linalg.norm(source_normals, axis=1) * np.linalg.norm(target_normals, axis=1)
    cos_angles = np.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_indices = np.where((distances <= max_distance) & (angles <= max_angle))[0]

    return high_confidence_indices.tolist()



def filter_high_confidence_matches_torch(target_vertex_data, closest_points_data, max_distance, max_angle):
    """filter high confidence matches using structured arrays."""

    target_positions = target_vertex_data["position"]
    target_normals = target_vertex_data["normal"]
    source_positions = closest_points_data["position"]
    source_normals = closest_points_data["normal"]

    # Calculate distances (vectorized)
    distances = torch.norm(source_positions - target_positions, dim=-1)

    # Calculate angles between normals (vectorized)
    cos_angles = torch.einsum("ij,ij->i", source_normals, target_normals)
    cos_angles = cos_angles / (torch.norm(source_normals, dim=-1) * torch.norm(target_normals, dim=-1))
    cos_angles = torch.abs(cos_angles)  # Consider opposite normals by taking absolute value
    angles = torch.arccos(torch.clamp(cos_angles, min=-1, max=1)) * 180 / np.pi

    # Apply thresholds (vectorized)
    high_confidence_flag = ((distances <= max_distance) & (angles <= max_angle))

    return high_confidence_flag


def do_inpainting(known_indices, unknown_indices, all_weights, L, inv_area):
    num_bones = all_weights.shape[1] ######################## sparse ?
    W = all_weights
    
    Q = -L + L @ sp.diags(inv_area) @ L

    S_match = np.array(known_indices)
    S_nomatch = np.array(unknown_indices)

    Q_UU = sp.csr_matrix(Q[np.ix_(S_nomatch, S_nomatch)])
    Q_UI = sp.csr_matrix(Q[np.ix_(S_nomatch, S_match)])

    W_I = W[S_match, :] # match_num x num_bones
    W_U = W[S_nomatch, :] # nomatch_num x num_bones
    W_I = np.clip(W_I, 0.0, 1.0) 
    # print('W_I', W_I.min(), W_I.max())

    for bone_idx in range(num_bones):
        if W_I[:, bone_idx].max() < 1e-3:
            continue
        
        b = -Q_UI @ W_I[:, bone_idx]
        W_U[:, bone_idx] = splinalg.spsolve(Q_UU, b)
        
    # print('W_U', W_U.max(), W_U.min())
    W[S_nomatch, :] = W_U

    # apply constraints,
    # each element is between 0 and 1
    W = np.clip(W, 0.0, 1.0)

    # normalize each row to sum to 1
    W_sum = W.sum(axis=1, keepdims=True)
    
    W = W / W.sum(axis=1, keepdims=True)

    return W


def transformation_inv(transforms):
    transforms_shape = transforms.shape
    transforms = transforms.reshape(-1, 4, 4)
    # transforms: batch x 4 x 4
    batch_size = transforms.shape[0]
    rotmat = transforms[:, :3, :3]
    transl = transforms[:, :3, 3]
    rotmat_inv = rotmat.transpose(1, 2)
    
    transl_inv = torch.einsum('bij,bj->bi', rotmat_inv, -transl)
    transforms_inv = torch.cat([rotmat_inv, transl_inv.unsqueeze(-1)], dim=-1)
    transforms_inv = torch.cat(
        [transforms_inv, torch.tensor([[[0, 0, 0, 1]]], device=transforms.device).repeat(batch_size, 1, 1)], dim=1)
    return transforms_inv.reshape(transforms_shape)
    

def do_transformation(V, transforms, R=None, t=None):
    # V: b x N x 3, transforms: b x 4 x 4
    V_shape = V.shape
    V = V.reshape(-1, 3)
    V = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1).reshape(V_shape[0], V_shape[1], 4)
    if transforms is None:
        transforms = torch.eye(4, device=V.device).reshape(1, 4, 4).expand(V_shape[0], 4, 4).clone()
        transforms[:, :3, :3] = R
        transforms[:, :3, 3] = t
        
    V = torch.einsum('bij,bnj->bni', transforms, V)
    return V[:, :, :3]


def do_transformation2(V, transforms, R=None, t=None):
    # V: b x N x 3, transforms: b x L x 4 x 4
    V_shape = V.shape
    V = V.reshape(-1, 3)
    V = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1).reshape(V_shape[0], V_shape[1], 4)
    if transforms is None:
        batch_size, seq_len, _, _ = R.shape
        transforms = torch.eye(4, device=V.device).reshape(1, 1, 4, 4).expand(batch_size, seq_len, 4, 4).clone()
        transforms[:, :, :3, :3] = R
        transforms[:, :, :3, 3] = t
        
    V = torch.einsum('blij,bnj->blni', transforms, V)
    return V[:, :, :, :3]



def get_modified_garment2(vertices0, smpl_verts0, A_smpl0, A_smpl1, smpl_based_lbs, extra_disp):
    # vertices0: n x 3, 
    # A_smpl0, A_smpl1: batch x K x 4 x 4, 
    # smpl_based_lbs: n x K
    batch_size, K, _, _ = A_smpl0.shape
    A_smpl0_inv = transformation_inv(A_smpl0.reshape(-1, 4, 4)).reshape(batch_size, K, 4, 4)

    dists, idx, _ = knn_points(vertices0.unsqueeze(0), smpl_verts0.unsqueeze(0), K=16)
    score = (1 / torch.sqrt(dists + 1e-8))
    score = score / score.sum(dim=2, keepdim=True) # b x num_p x K

    disp_v = knn_gather(extra_disp, idx)
    disp_v = (disp_v * score.unsqueeze(-1)).sum(dim=-2)

    vertices0_homo = torch.cat(
        [vertices0, torch.ones_like(vertices0[:, :1])], dim=-1
    )
    
    transformed_v1 = torch.einsum('bkij,nj->bnki', A_smpl0_inv, vertices0_homo)
    transformed_v1 = (transformed_v1 * smpl_based_lbs.unsqueeze(-1)).sum(dim=2)

    transformed_v1[:, :, :3] = transformed_v1[:, :, :3] + disp_v

    transformed_v2 = torch.einsum('bkij,bnj->bnki', A_smpl1, transformed_v1)
    transformed_v2 = (transformed_v2 * smpl_based_lbs.unsqueeze(-1)).sum(dim=2)

    transformed_v2 = transformed_v2[:, :, :3] / transformed_v2[:, :, [3]]
    return transformed_v2


def linear_blending(vertices_rest, R, t, lbs_weight, transform_matrix=None):
    # vertices_rest: N x 3
    
    if transform_matrix is None:
        batch_size = R.shape[0]
        K = R.shape[1]
        
        transform_matrix = torch.zeros(batch_size, K, 4, 4, device=vertices_rest.device)
        transform_matrix[:, :, :3, :3] = R
        transform_matrix[:, :, :3, 3] = t
        transform_matrix[:, :, 3, 3] = 1

    if len(vertices_rest.shape) == 2:
        vertices_homo = torch.cat([vertices_rest, torch.ones(vertices_rest.shape[0], 1, device=vertices_rest.device)], dim=-1)
        vertices_homo = torch.einsum('bkij,nj->bnki', transform_matrix, vertices_homo)
    else:
        vertices_homo = torch.cat([
            vertices_rest, torch.ones(vertices_rest.shape[0], vertices_rest.shape[1], 1, device=vertices_rest.device)
        ], dim=-1)
        vertices_homo = torch.einsum('bkij,bnj->bnki', transform_matrix, vertices_homo)

    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(-1)).sum(dim=2)
    vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
    
    return vertices_new


def linear_blending_batch(vertices_rest, R, t, lbs_weight, transform_matrix=None):
    # vertices_rest: b x N x 3, each batch use on R and T
    batch_size = vertices_rest.shape[0]
    if transform_matrix is None:
        K = R.shape[1]
        
        transform_matrix = torch.zeros(batch_size, K, 4, 4, device=vertices_rest.device)
        transform_matrix[:, :, :3, :3] = R
        transform_matrix[:, :, :3, 3] = t
        transform_matrix[:, :, 3, 3] = 1

    vertices_homo = torch.cat([vertices_rest, torch.ones(batch_size, vertices_rest.shape[1], 1, device=vertices_rest.device)], dim=-1)
    vertices_homo = torch.einsum('bkij,bnj->bnki', transform_matrix, vertices_homo)
    vertices_homo = (vertices_homo * lbs_weight.unsqueeze(-1)).sum(dim=2)
    
    vertices_new = vertices_homo[:, :, :3] / vertices_homo[:, :, [3]]
    
    return vertices_new



def put_V_back_by_A0(V, verts_mean_posed, max_length, A0_inv=None, transl=0):
    scale = max_length / 0.9
    V = V * scale + verts_mean_posed - transl
    
    if A0_inv is not None:
        V_last = torch.ones_like(V[..., :1])
        V_homo = torch.cat([V, V_last], dim=-1)
        V = torch.einsum('bij,blj->bli', A0_inv, V_homo)
    
    return V[..., :3]


def put_V_back_by_A0_list(V_list, verts_mean_posed, max_length):
    scale = max_length / 0.9
    batch_size = len(V_list)

    V = [V_list[i] * scale[i] + verts_mean_posed[i] for i in range(batch_size)]
    
    return V


def lbs_weight_nofoot(lbs_weight):
    assert len(lbs_weight.shape) == 3
    assert ((lbs_weight.shape[-1] == 55) or (lbs_weight.shape[-1] == 21)) or (lbs_weight.shape[-1] == 22)
    lbs_weight_new  = lbs_weight.clone()
    lbs_weight_new[:, :, 4] += lbs_weight_new[:, :, 10]
    lbs_weight_new[:, :, 5] += lbs_weight_new[:, :, 11]
    lbs_weight_new[:, :, 4] += lbs_weight_new[:, :, 7]
    lbs_weight_new[:, :, 5] += lbs_weight_new[:, :, 8]

    lbs_weight_new[:, :, 7] = 0
    lbs_weight_new[:, :, 8] = 0
    lbs_weight_new[:, :, 10] = 0
    lbs_weight_new[:, :, 11] = 0

    lbs_weight_new[:, :, 13] += (lbs_weight_new[:, :, 12] + lbs_weight_new[:, :, 15]) * 0.5
    lbs_weight_new[:, :, 14] += (lbs_weight_new[:, :, 12] + lbs_weight_new[:, :, 15]) * 0.5
    lbs_weight_new[:, :, 15] = 0
    lbs_weight_new[:, :, 12] = 0
    return lbs_weight_new


def diffuse_lbs_to_space2(points, vertices_rest, lbs_weight, K=16, no_foot=True, return_dist=False):
    # print('shapes', points.shape, vertices_rest.shape, faces.shape, lbs_weight.shape)
    
    dists, idx, _ = knn_points(points, vertices_rest, K=K)
    score = (1 / torch.sqrt(dists + 1e-8))
    score = score / score.sum(dim=2, keepdim=True) # b x num_p x K
    
    if len(lbs_weight.shape) == 2:
        lbs_weight = lbs_weight.unsqueeze(0).expand(len(points), -1, -1)

    if no_foot:
        lbs_weight_new = torch.zeros_like(lbs_weight)
        flags = torch.ones(lbs_weight.shape[-1], dtype=torch.bool)
        lbs_weight_new[:, :, 4] = (lbs_weight[:, :, 4] + lbs_weight[:, :, 10] + lbs_weight[:, :, 7])
        lbs_weight_new[:, :, 5] = (lbs_weight[:, :, 5] + lbs_weight[:, :, 11] + lbs_weight[:, :, 8])
        lbs_weight_new[:, :, 13] = lbs_weight[:, :, 13] + (lbs_weight[:, :, 12] + lbs_weight[:, :, 15]) * 0.5
        lbs_weight_new[:, :, 14] = lbs_weight[:, :, 14] + (lbs_weight[:, :, 12] + lbs_weight[:, :, 15]) * 0.5

        flags[[4, 5, 7, 8, 10, 11, 12, 13, 14, 15]] = False
        lbs_weight_new[:, :, flags] = lbs_weight[:, :, flags]

        lbs_weight = lbs_weight_new
        
    lbs_weight_gathered = knn_gather(lbs_weight, idx)
    lbs_weight_gathered = (lbs_weight_gathered * score.unsqueeze(-1)).sum(dim=-2)

    if return_dist:
        return lbs_weight_gathered, dists[:, :, 0] # (N, P1)
    
    return lbs_weight_gathered


def get_one_hot_np(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_one_hot(targets, nb_classes, device):
    res = torch.eye(nb_classes, device=device)[
        targets.reshape(-1)
    ]
    return res.reshape(list(targets.shape)+[nb_classes])



def deform_garments(smplx_layer, params_old, params_new, garment_mesh, lbs_weights, scale=True, return_lbs=False):

    garment_verts = garment_mesh.verts_padded()
    if (garment_verts.max(dim=1)[0] - garment_verts.min(dim=1)[0]).max() > 10:
        scale = True
        garment_verts = garment_verts * 0.01
    
    pose2rot = False if (params_old['poses'].ndim == 4 or params_old['poses'].shape[-1] == 9) else True
    if 'scale' not in params_old:
        garment_verts = garment_verts - params_old['transl'].unsqueeze(1)
        smplx_out_old = smplx_layer.forward_simple(
            betas=params_old['betas'],
            full_pose=params_old['poses'],
            transl=params_old['transl'] * 0.0,
            pose2rot=pose2rot
        )
    
    else:
        smplx_out_old = smplx_layer.forward_simple(
            betas=params_old['betas'],
            full_pose=params_old['poses'],
            transl=torch.zeros(1, 3, device=params_old['betas'].device),
            pose2rot=pose2rot,
            scale=params_old['scale']
        )

        body_verts_0 = params_old['body_vs']
        body_verts_1 = smplx_out_old.vertices

        transl = (body_verts_0.max(dim=1)[0] + body_verts_0.min(dim=1)[0] - body_verts_1.max(dim=1)[0] - body_verts_1.min(dim=1)[0]) * 0.5
        garment_verts = garment_verts - transl.unsqueeze(1)
        params_old['transl'] = transl

    A_old = smplx_out_old['A']
    smplx_verts = smplx_out_old.vertices

    garment_mesh = Meshes(verts=garment_verts, faces=garment_mesh.faces_padded())
    smplx_mesh = Meshes(verts=smplx_verts, faces=smplx_layer.faces_tensor.unsqueeze(0))

    final_lbs_weight, smplx_lbs_weight = get_best_skinning_weights(
        garment_mesh, smplx_mesh, lbs_weights, max_distance=0.05, max_angle=25.0
    )
    final_lbs_weight = torch.from_numpy(final_lbs_weight).float().cuda()
    if torch.isnan(final_lbs_weight).any() or torch.isinf(final_lbs_weight).any():
        final_lbs_weight = torch.from_numpy(smplx_lbs_weight).float().cuda()

    pose2rot = False if (params_new['poses'].ndim == 4 or params_new['poses'].shape[-1] == 9) else True
    smplx_out_new = smplx_layer.forward_simple(
        betas=params_new['betas'],
        full_pose=params_new['poses'],
        transl=params_new['transl'] * 0.0,
        pose2rot=pose2rot
    )
    A_new = smplx_out_new['A']
        
    extra_disp = smplx_layer.get_disp(
        params_new['betas']-params_old['betas'],
        params_old['poses'], params_new['poses'],
    )
    
    vertices_new_bid = get_modified_garment2(
                garment_verts[0], smplx_verts[0], A_old, A_new, final_lbs_weight, extra_disp=extra_disp[0])[0]
    
    vertices_new_bid = vertices_new_bid + params_new['transl']
    if return_lbs:
        return vertices_new_bid, final_lbs_weight
    
    return vertices_new_bid
