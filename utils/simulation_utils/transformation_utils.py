import torch
import numpy as np
from utils.simulation_utils.camera_view_utils import *


def transform2origin(position_tensor):  
    min_pos = torch.min(position_tensor, 0)[0]  
    max_pos = torch.max(position_tensor, 0)[0]  
    max_diff = torch.max(max_pos - min_pos) 
    original_mean_pos = (min_pos + max_pos) / 2.0   
    scale = 1.0 / max_diff   
    # scale = scale * 0.5
    original_mean_pos = original_mean_pos.to(device="cuda")
    scale = scale.to(device="cuda")
    
    
    new_position_tensor = (position_tensor - original_mean_pos) * scale     

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):    
    return original_mean_pos + position_tensor / scale


def transform2normal(pos):
    pos_max = pos.max()
    pos_min = pos.min()
    scale = (pos_max - pos_min) * 1.8
    shift = -pos_min + (pos_max - pos_min) * 0.25
    print("scale, shift", scale, shift)
    transformed_pos = (pos + shift) / scale
    sim_aabb = torch.stack(
        [torch.min(transformed_pos, dim=0)[0], torch.max(transformed_pos, dim=0)[0]], dim=0
    )
    # This AABB is used to constraint the material fields and velocity fields.
    sim_aabb = (
        sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
    ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)
    print("simulation aabb: ", sim_aabb)
    
    return transformed_pos, scale, shift

def undotransform2normal(transformed_pos, scale, shift):
    pos = (transformed_pos * scale) - shift
    return pos
    


def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)   
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)   
    if axis == 0:   
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:  
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2: 
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)  

    matrices = []

    for i in range(len(degrees)):   
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)  
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)): 
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_uppder_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")    
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")    
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_uppder_from_mat(cov_tensor)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )   
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )   
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace 
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )   
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )   
    
    observant_coordinates = np.column_stack((h1, h2, vertical)) 

    return viewpoint_center_worldspace, observant_coordinates

def get_camera_info(xyz, rotation_matrices):
    xyz = apply_rotations(xyz, rotation_matrices)
    new_xyz, scale_origin, original_mean_pos = transform2origin(xyz)
    return {
        "scale_origin": scale_origin,
        "original_mean_pos": original_mean_pos,
    }

def get_camera_info_normal(xyz, rotation_matrices):
    xyz = apply_rotations(xyz, rotation_matrices)
    new_xyz, scale, shift = transform2normal(xyz)
    return {
        "scale_origin": scale,
        "original_mean_pos": shift,
    }
