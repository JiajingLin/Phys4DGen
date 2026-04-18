

import os
import json
import torch
import numpy as np
import open3d as o3d
import matplotlib as mpl

from core.gpt_inference import parse_material_list
from utils.project_utils import load_ns_point_cloud
from core.arguments import get_args
from utils.voxel_grid_utils import *
from utils.simulation_utils.binding_utils import *
from utils.load_utils import *


@torch.no_grad()
def get_interpolated_values(source_pts, source_vals, inner_pts, batch_size=8192, k=1):
    n_inner = len(inner_pts)
    inner_vals = torch.zeros(n_inner, dtype=torch.int32, device=inner_pts.device)
    for batch_start in range(0, n_inner, batch_size):
        curr_batch_size = min(batch_size, n_inner - batch_start)
        curr_inner_pts = inner_pts[batch_start:batch_start + curr_batch_size]

        dists = torch.cdist(curr_inner_pts, source_pts)
        _, idxs = torch.topk(dists, k=k, dim=1, largest=False)
        

        curr_inner_vals = source_vals[idxs].mode(1)[0]

        inner_vals[batch_start:batch_start + curr_batch_size] = curr_inner_vals
    return inner_vals
        

@torch.no_grad()
def get_particles_material_group(args):
    pcd_file = os.path.join(args.scene_dir, 'point_cloud.ply')
    binding_path = os.path.join(args.scene_dir, 'bindings.pt')
    save_pts_file = os.path.join(args.scene_dir, 'particle.ply')

    gaussians = load_checkpoint(pcd_file, sh_degree=args.sh_degree)
    gs_pts = gaussians.get_xyz
    gs_material_group = torch.load(os.path.join(args.scene_dir, 'features', 'gs_material_group.pt'))
    
    carved_pts, grid_density, is_surface, grid_cell_size, xyz1 = construct_grid_structure(args)
    grid_material_group = torch.full_like(is_surface, 0, dtype=torch.int32, device=args.device)    
    
    if args.particle_binding:
        weight_mat = gaussian_binding_with_surface_define(gaussians, carved_pts, is_surface, grid_cell_size, xyz1)

        save_sparse_chunks(weight_mat, binding_path)
    else:

        gaussian_binding_with_surface_define_noweightmat(gaussians, carved_pts, is_surface, grid_cell_size, xyz1)

    surface_pts = (torch.nonzero(is_surface) * grid_cell_size + xyz1)
    surface_pts_material_group = get_interpolated_values(gs_pts, gs_material_group, surface_pts)
    
    surface_cells = torch.nonzero(is_surface)
    grid_material_group[surface_cells[:, 0], surface_cells[:, 1], surface_cells[:, 2]] = surface_pts_material_group

    is_internal =  (grid_density > 0) & (is_surface == False)
    internal_cells = torch.nonzero(is_internal)

    grid_material_group, valid_cells = assign_internal_cells_material_group(grid_material_group, internal_cells)
    is_internal = valid_cells

    valid_material_pts = (torch.nonzero(grid_material_group != 0) * grid_cell_size + xyz1)
    valid_material_pts_material_group = grid_material_group[grid_material_group != 0]
    pts_material_group = valid_material_pts_material_group
    valid_cells = ((valid_material_pts - xyz1) / grid_cell_size).round().to(torch.int32)
    
    grid_material_group[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]] = pts_material_group 
    
    grid_info = {
        "grid_material_group": grid_material_group,
        "is_surface": is_surface,
        "is_internal": is_internal,
        "grid_cell_size": grid_cell_size,
        "xyz1": xyz1
    }
    torch.save(grid_info, os.path.join(args.scene_dir, 'features', 'grid_info.pt'))
    torch.save(pts_material_group, os.path.join(args.scene_dir, 'features', 'pts_material_group.pt'))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_material_pts.cpu().numpy())
    o3d.io.write_point_cloud(save_pts_file, pcd)
    
    return pts_material_group, is_surface, is_internal

if __name__ == '__main__':
    args = get_args()
    get_particles_material_group(args)