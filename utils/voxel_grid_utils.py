import taichi as ti
import torch
from utils.carving_utils import get_carved_and_voxels
import open3d as o3d
import os
ti.init(arch=ti.cuda, device_memory_GB=8.0)

@ti.kernel
def compute_grid_gradient(grid_density: ti.template(), 
    grid_gradient: ti.template()):
    N = grid_density.shape[0]
    M = grid_density.shape[1]
    K = grid_density.shape[2]
    print(N, M, K)
    for i, j, k in grid_density:  
        
        
        #     gx = (voxels_data[i+1, j, k] - voxels_data[i-1, j, k]) * 0.5
        #     gy = (voxels_data[i, j+1, k] - voxels_data[i, j-1, k]) * 0.5
        #     gz = (voxels_data[i, j, k+1] - voxels_data[i, j, k-1]) * 0.5
        
        # print(i, j, k)
        gx, gy, gz = 0.0, 0.0, 0.0  
        
        if 1 <= i < N-1 :
            gx = (grid_density[i+1, j, k] - grid_density[i-1, j, k]) * 0.5
        else:
            if 1 > i:
                gx = (grid_density[i+1, j, k] - 0) * 0.5
            elif i >= N-1:
                gx = (0 - grid_density[i-1, j, k]) * 0.5
        
        
        if 1 <= j < M-1:
            gy = (grid_density[i, j+1, k] - grid_density[i, j-1, k]) * 0.5
        else:
            if 1 > j:
                gy = (grid_density[i, j+1, k] - 0) * 0.5
            elif j >= M-1:
                gy = (0 - grid_density[i, j-1, k]) * 0.5
                
        
        if 1 <= k < K-1:
            gz = (grid_density[i, j, k+1] - grid_density[i, j, k-1]) * 0.5
        else:
            if 1 > k:
                gz = (grid_density[i, j, k+1] - 0) * 0.5
            elif k >= K-1:
                gz = (0 - grid_density[i, j, k-1]) * 0.5
            
            
        grid_gradient[i, j, k] = ti.Vector([gx, gy, gz])
        
        
@ti.kernel
def surface_filling(surface_voxels_indices: ti.template(), 
    voxels_gradient: ti.template(), 
    surface_thickness: ti.template(), 
    is_surface: ti.template(), 
    is_surface_copy: ti.template()):
    for v in range(surface_voxels_indices.shape[0]):
        i, j, k = surface_voxels_indices[v]
        
        dx = (voxels_gradient[i, j, k][0] > 0) * 1 + (voxels_gradient[i, j, k][0] < 0) * -1
        dy = (voxels_gradient[i, j, k][1] > 0) * 1 + (voxels_gradient[i, j, k][1] < 0) * -1
        dz = (voxels_gradient[i, j, k][2] > 0) * 1 + (voxels_gradient[i, j, k][2] < 0) * -1
        # print(dx, dy, dz)
        for t in range(1, surface_thickness[i, j, k]+1):
            new_i = i + dx * t
            
            if 0 <= new_i < is_surface.shape[0] and is_surface_copy[new_i, j, k] == 0:
            # if 0 <= new_i < is_surface.shape[0]:
                is_surface[new_i, j, k] = 1
                # print("hello world!")
            else:
                break
        
        for t in range(1, surface_thickness[i, j, k]+1):
            new_j = j + dy * t
            if 0 <= new_j < is_surface.shape[1] and is_surface_copy[i, new_j, k] == 0:
            # if 0 <= new_j < is_surface.shape[1]:
                is_surface[i, new_j, k] = 1
                # print(i, new_j, k)
            else:
                break
        
        for t in range(1, surface_thickness[i, j, k]+1):
            new_k = k + dz * t
            if 0 <= new_k < is_surface.shape[2] and is_surface_copy[i, j, new_k] == 0:
            # if 0 <= new_k < is_surface.shape[2]:
                is_surface[i, j, new_k] = 1
                # print(i, j, new_k)
            else:
                break

@ti.func
def collision_search(
    grid: ti.template(), index, dir_type
):
    dir = ti.Vector([0, 0, 0])
    
    if dir_type == 0:   
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = 0  
    index += dir 
    i, j, k = index
    
    while 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and 0 <= k < grid.shape[2]:  
        if grid[index] != 0:     
            flag = grid[index]     
            break
        index += dir    
        i, j, k = index

    return flag

@ti.kernel
def internal_matrial_search(voxels_mat_vectors: ti.template(), voxels_mat: ti.template(), internal_voxels: ti.template()):
    for v in range(internal_voxels.shape[0]):
        i, j, k = internal_voxels[v]
        
        # mat_vector = ti.Vector([-1, -1, -1, -1, -1, -1], dtype=ti.int32)
        for dir_type in ti.static(range(6)):    
            mat_idx = collision_search(
                grid=voxels_mat,
                index=ti.Vector([i, j, k]),
                dir_type=dir_type,
            )
            # mat_vector[dir_type] = mat_idx
            voxels_mat_vectors[i, j, k][dir_type] = mat_idx
        
        
def assign_internal_cells_material_group(grid_material_group, internal_cells):
    
    ti_grid_material_group = ti.field(dtype=ti.int32, shape=grid_material_group.shape)
    ti_grid_material_group.from_torch(grid_material_group)
    ti_internal_cells = ti.Vector.field(n=3, dtype=ti.int32, shape=internal_cells.shape[0])
    ti_internal_cells.from_torch(internal_cells)
    cell_mat_vectors = torch.full([grid_material_group.shape[0], grid_material_group.shape[1], grid_material_group.shape[2], 6],
                                    0, dtype=torch.int32, device=grid_material_group.device)
    ti_cell_mat_vectors = ti.Vector.field(n=6, dtype=ti.int32, shape=grid_material_group.shape)
    ti_cell_mat_vectors.from_torch(cell_mat_vectors)

    internal_matrial_search(ti_cell_mat_vectors, ti_grid_material_group, ti_internal_cells)
    
    
    cell_mat_vectors = ti_cell_mat_vectors.to_torch().to('cuda')
    
    most_common_mat = torch.mode(cell_mat_vectors, dim=-1).values
    # print(most_common_mat.max())
    
    valid_cells = (torch.sum(cell_mat_vectors != 0, dim=-1) == 6)
    print(torch.nonzero(valid_cells).shape)    
    
    
    grid_material_group[valid_cells] = most_common_mat[valid_cells]
    
    return grid_material_group, valid_cells
        

def construct_grid_structure(args):
    carved_pts, grid_density, grid_cell_size, xyz1 = get_carved_and_voxels(args.scene_dir, grid_cell_size_ns=args.grid_cell_size_ns, dist_thr_ns=args.dist_thr_ns, mask_only=args.mask_only)
    xyz1 = torch.from_numpy(xyz1).to(args.device).float()
    
    
    # voxels_data = voxels_data.float()
    
    ti_grid_density = ti.field(dtype=ti.float32, shape=grid_density.shape)
    ti_grid_density.from_torch(grid_density)
    ti_grid_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=grid_density.shape)
    
    compute_grid_gradient(ti_grid_density, ti_grid_gradient)
    
    
    grid_gradient = ti_grid_gradient.to_torch()
    
    grid_gradient_norm = torch.norm(grid_gradient, dim=-1).to('cuda')
    
    is_surface = (grid_gradient_norm > 0) & (grid_density > 0)
    
    # print(surface_thickness.max())
    surface_cell_indices = torch.nonzero(is_surface)  
    
    is_internal = (grid_density > 0) & (is_surface == False) 
    
    surface_or_internal_cell = torch.zeros(grid_density.shape, dtype=torch.int32, device=grid_density.device)
    surface_or_internal_cell[is_surface] = 1
    surface_or_internal_cell[is_internal] = 2
    
    
    surface_pts = torch.nonzero(surface_or_internal_cell == 1) * grid_cell_size + xyz1[0]
    internal_pts = torch.nonzero(surface_or_internal_cell == 2) * grid_cell_size + xyz1[0]
    
    surface_pts = surface_pts.cpu().numpy()
    internal_pts = internal_pts.cpu().numpy()
    
    return carved_pts, grid_density, is_surface, grid_cell_size, xyz1

if __name__ == "__main__":
    from arguments import *
    args = get_args()
    # construct_voxels_structure(args)
    
    