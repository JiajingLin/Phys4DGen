import imp
import os
import numpy as np
import open3d as o3d
import time
import torch
from PIL import Image
from utils.project_utils import *


def get_bounding_box(pts, percentile=1.0, buffer=0.1):
    """Get the bounding box of a point cloud."""
    xyz1 = np.percentile(pts, percentile, axis=0)
    xyz2 = np.percentile(pts, 100 - percentile, axis=0)
    lwh = xyz2 - xyz1
    xyz1 -= buffer * lwh
    xyz2 += buffer * lwh
    return xyz1, xyz2       


def get_grid_points(xyz1, xyz2, grid_cell_size):
    """Get grid points."""
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    x = np.arange(x1, x2, grid_cell_size)
    y = np.arange(y1, y2, grid_cell_size)
    z = np.arange(z1, z2, grid_cell_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return grid_pts



def project_3d_to_2d(pts, w2c, K, return_dists=False):
    """Project 3D points to 2D (nerfstudio format)."""
    pts = np.array(pts)
    K = np.hstack([K, np.zeros((3, 1))])
    pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts = np.dot(pts, w2c.T)
    pts[:, [1, 2]] *= -1
    if return_dists:
        dists = np.linalg.norm(pts[:, :3], axis=-1)
    pts = np.dot(pts, K.T)
    pts_2d = pts[:, :2] / pts[:, 2:]
    if return_dists:
        return pts_2d, dists
    return pts_2d

def project_3d_to_2d_torch(pts, w2c, K, return_dists=False):
    """Project 3D points to 2D (nerfstudio format)."""
    device = pts.device
    K = torch.cat([K, torch.zeros((3, 1), device=device)], 1)
    pts = torch.cat([pts, torch.ones((pts.shape[0], 1), device=device)], 1)
    pts = torch.matmul(pts, w2c.t())
    pts[:, [1, 2]] *= -1
    if return_dists:
        dists = torch.norm(pts[:, :3], dim=-1)
    pts = torch.matmul(pts, K.t())
    pts_2d = pts[:, :2] / pts[:, 2:]
    if return_dists:
        return pts_2d, dists
    return pts_2d


def depth_to_distance(depth, K):
    """Convert depth map to distance from camera."""
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    pts = np.stack([x, y, np.ones_like(x)], axis=1)
    pts = np.dot(pts, np.linalg.inv(K).T)
    pts *= depth[:, None]
    dists = np.linalg.norm(pts, axis=1)
    dists = dists.reshape(h, w)
    return dists


def depth_to_distance_torch(depth, K):
    """Convert depth map to distance from camera."""
    h, w = depth.shape
    x, y = torch.meshgrid(torch.arange(w), torch.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    pts = torch.stack([x, y, torch.ones_like(x)], dim=1).float().to(depth.device)
    pts = torch.matmul(pts, torch.inverse(K).t())
    pts *= depth[:, None]
    dists = torch.norm(pts, dim=1)
    dists = dists.reshape(h, w)
    return dists

def transform2origin(position_tensor):   
    min_pos = np.min(position_tensor, 0)[0]  
    
    max_pos = np.max(position_tensor, 0)[0]  
    max_diff = np.max(max_pos - min_pos) 
    
    original_mean_pos = (min_pos + max_pos) / 2.0   
    scale = 1.0 / max_diff   
    # original_mean_pos = original_mean_pos.to(device="cuda")
    # scale = scale.to(device="cuda")
    
    

    
    # return new_position_tensor, scale, original_mean_pos
    return scale

def carve_numpy(pts, masks, depths, w2cs, K, dist_thr):
    n_imgs = len(masks)

    for i in range(n_imgs):
        h, w = masks[i].shape
        pts_2d, dists = project_3d_to_2d(pts, w2cs[i], K, return_dists=True)
        pts_2d = np.round(pts_2d).astype(np.int32)
        pts_2d = np.clip(pts_2d, 0, [w - 1, h - 1])

        observed_dists = depths[i] 

        is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
        is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
        pts = pts[is_in_mask & is_behind_depth]

    return pts       


def carve_torch(pts, masks, depths, w2cs, K, dist_thr, mask_only=False):
    n_imgs = len(masks)

    with torch.no_grad():
        mask_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        depth_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        for i in range(n_imgs):
            h, w = masks[i].shape
            pts_2d, dists = project_3d_to_2d_torch(pts, w2cs[i], K, return_dists=True)
            pts_2d = torch.round(pts_2d).long().to(pts.device)
            
            pts_2d[:, 0] = torch.clamp(pts_2d[:, 0], 0, w - 1)
            pts_2d[:, 1] = torch.clamp(pts_2d[:, 1], 0, h - 1)
            
            observed_dists = depths[i] 
            
            is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
            
            
            is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
            mask_votes[is_in_mask] += 1
            depth_votes[is_behind_depth] += 1
        
        if mask_only:
            
            pts = pts[mask_votes == n_imgs]
        else:
            
            pts = pts[(mask_votes == n_imgs) & (depth_votes == n_imgs)]

    return pts
   

def carve_torch_v1(pts, grid_shape, masks, depths, w2cs, K, dist_thr, mask_only=False):
    n_imgs = len(masks)
    
    x_dim, y_dim, z_dim = grid_shape    
    
    with torch.no_grad():
        mask_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        depth_votes = torch.zeros(len(pts), device=pts.device, dtype=torch.int32)
        for i in range(n_imgs):
            h, w = masks[i].shape
            pts_2d, dists = project_3d_to_2d_torch(pts, w2cs[i], K, return_dists=True)
            pts_2d = torch.round(pts_2d).long().to(pts.device)
            
            pts_2d[:, 0] = torch.clamp(pts_2d[:, 0], 0, w - 1)
            pts_2d[:, 1] = torch.clamp(pts_2d[:, 1], 0, h - 1)
            
            observed_dists = depths[i] 
            
            is_in_mask = masks[i][pts_2d[:, 1], pts_2d[:, 0]]
            
            is_behind_depth = dists > observed_dists[pts_2d[:, 1], pts_2d[:, 0]] - dist_thr
            mask_votes[is_in_mask] += 1
            depth_votes[is_behind_depth] += 1
        
        if mask_only:
            
            valid_mask = mask_votes == n_imgs
            pts = pts[valid_mask]
            grid_density = valid_mask.reshape(x_dim, y_dim, z_dim)
        else:
            
            valid_mask = (mask_votes == n_imgs) & (depth_votes == n_imgs)
            pts = pts[valid_mask]
            grid_density = valid_mask.reshape(x_dim, y_dim, z_dim)

    return pts, grid_density



def get_carved_pts(scene_dir, grid_cell_size_ns=1/32, dist_thr_ns=0.1, verbose=False, device='cuda'):
    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
    t_file = os.path.join(scene_dir, 'transforms.json')
    img_dir = os.path.join(scene_dir, 'render_images')
    depth_dir = os.path.join(scene_dir, 'render_depths')

    

    pts = load_ns_point_cloud(pcd_file)
    w2cs, K, _ = parse_transforms_json(t_file, return_w2c=True)
    imgs, masks = load_images(img_dir, return_masks=True)
    depths = load_depths(depth_dir, Ks=None)
    
    scale = transform2origin(pts)

    xyz1, xyz2 = get_bounding_box(pts)
    grid_cell_size = grid_cell_size_ns / scale    
    
    grid_pts = get_grid_points(xyz1, xyz2, grid_cell_size)
    dist_thr = dist_thr_ns / scale

    grid_pts = torch.from_numpy(grid_pts).float().to(device)
    masks = [torch.from_numpy(mask).to(device) for mask in masks]
    depths = [torch.from_numpy(depth).to(device) for depth in depths]
    w2cs = [torch.from_numpy(w2c).float().to(device) for w2c in w2cs]
    K = torch.from_numpy(K).float().to(device)
    
    carved = carve_torch(grid_pts, masks, depths, w2cs, K, dist_thr, mask_only=True)
    
    if verbose:
        print('scene: %s, num. surface points: %d, num. carved points: %d, scale: %.4f' % 
              (scene_name, len(pts), len(carved), scale))
    
    return carved, grid_cell_size

# def get_carved_and_voxels(scene_dir, grid_cell_size_ns=1/32, dist_thr_ns=0.05, verbose=False, device='cuda'):
#     """

#     """
#     scene_name = os.path.basename(scene_dir)
#     pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
#     t_file = os.path.join(scene_dir, 'transforms.json')
#     img_dir = os.path.join(scene_dir, 'render_images')
#     depth_dir = os.path.join(scene_dir, 'render_depths')

#     pts = load_ns_point_cloud(pcd_file)
#     w2cs, K, _ = parse_transforms_json(t_file, return_w2c=True)
#     imgs, masks = load_images(img_dir, return_masks=True)
#     depths = load_depths(depth_dir, Ks=None)
    


#     xyz1, xyz2 = get_bounding_box(pts)
#     grid_cell_size = grid_cell_size_ns / scale
#     dist_thr = dist_thr_ns / scale
    

#     # grid_pts = get_grid_points(xyz1, xyz2, grid_cell_size)
#     """Get grid points."""
#     x1, y1, z1 = xyz1
#     x2, y2, z2 = xyz2
#     x = np.arange(x1, x2, grid_cell_size)
#     y = np.arange(y1, y2, grid_cell_size)
#     z = np.arange(z1, z2, grid_cell_size)




#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#     grid_pts = np.stack([xx, yy, zz], axis=-1)
#     grid_pts = torch.from_numpy(grid_pts).float().to(device)
#     grid_shape = grid_pts.shape[:-1]
#     # grid_density = torch.zeros_like(grid_pts, device=device)
    
#     pts = grid_pts.reshape(-1, 3)
    
#     masks = [torch.from_numpy(mask).to(device) for mask in masks]
#     depths = [torch.from_numpy(depth).to(device) for depth in depths]
#     w2cs = [torch.from_numpy(w2c).float().to(device) for w2c in w2cs]
#     K = torch.from_numpy(K).float().to(device)
        

#     carved_pts, grid_density = carve_torch(pts, grid_shape, masks, depths, w2cs, K, dist_thr, mask_only=True)
    

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(carved_pts.cpu().numpy())
#     o3d.io.write_point_cloud(os.path.join("log", 'carved_pts.ply'), pcd)
        
#     return carved_pts, grid_density, grid_cell_size, xyz1


def get_carved_and_voxels(scene_dir, grid_cell_size_ns=1/32, dist_thr_ns=0.1, mask_only=False, device='cuda'):
    """
    获取雕刻后的点云和体素数据，体素点云中存储是否为对象体素，True为对象体素，False为背景体素
    """
    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
    t_file = os.path.join(scene_dir, 'transforms.json')
    img_dir = os.path.join(scene_dir, 'render_images')
    depth_dir = os.path.join(scene_dir, 'render_depths')

    pts = load_ns_point_cloud(pcd_file)
    w2cs, K, _ = parse_transforms_json(t_file, return_w2c=True)
    imgs, masks = load_images(img_dir, return_masks=True)
    depths = load_depths(depth_dir, Ks=None)
    
    scale = transform2origin(pts) 

    xyz1, xyz2 = get_bounding_box(pts)
    grid_cell_size = grid_cell_size_ns / scale
    dist_thr = dist_thr_ns / scale
    
    
    # grid_pts = get_grid_points(xyz1, xyz2, grid_cell_size)
    """Get grid points."""
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    x = np.arange(x1, x2, grid_cell_size)
    y = np.arange(y1, y2, grid_cell_size)
    z = np.arange(z1, z2, grid_cell_size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # xx, yy, zz = np.meshgrid(x, y, z)
    grid_pts = np.stack([xx, yy, zz], axis=-1)
    grid_pts = torch.from_numpy(grid_pts).float().to(device)
    grid_shape = grid_pts.shape[:-1]
    # grid_density = torch.zeros_like(grid_pts, device=device)
    
    pts = grid_pts.reshape(-1, 3)
    
    masks = [torch.from_numpy(mask).to(device) for mask in masks]
    depths = [torch.from_numpy(depth).to(device) for depth in depths]
    w2cs = [torch.from_numpy(w2c).float().to(device) for w2c in w2cs]
    K = torch.from_numpy(K).float().to(device)
        
    
    carved_pts, grid_density = carve_torch_v1(pts, grid_shape, masks, depths, w2cs, K, dist_thr, mask_only=True)
        
    return carved_pts, grid_density, grid_cell_size, xyz1


if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    scene_dir = args.scene_dir
    carved_pts, grid_density, grid_cell_size, xyz1 = get_carved_and_voxels(scene_dir)
    carved_pts = carved_pts.cpu().numpy()
    grid_density = grid_density.cpu().numpy()
    
    non_zero_indices = np.nonzero(grid_density)
    
    non_zero_coords = np.column_stack(non_zero_indices)
    
    grid_pts = non_zero_coords * grid_cell_size + xyz1
    print("carved point nums", carved_pts.shape[0])
    print("grid point nums", grid_pts.shape[0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(carved_pts)
    o3d.io.write_point_cloud(os.path.join("log", 'carved_pts.ply'), pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_pts)
    o3d.io.write_point_cloud(os.path.join("log", 'grid_pts.ply'), pcd)
    
    

# if __name__ == '__main__':
#     args = get_args()

    
#     carved, grid_cell_size = get_carved_pts(args.scene_dir)
#     is_visible = get_surface_pts(args.scene_dir, carved, 0.01)
#     pts = carved[is_visible].cpu().numpy()
    
#     # carved_pts, grid_cell_size, voxels_data = get_carved_and_voxels(args.scene_dir)
#     # valid_voxels = torch.nonzero(voxels_data).cpu().numpy()
#     # pts = valid_voxels * grid_cell_size
    
#     # pts = carved_pts.cpu().numpy()
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     o3d.visualization.draw_geometries([pcd])

