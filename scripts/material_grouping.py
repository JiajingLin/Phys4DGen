import os
import numpy as np
import torch
from PIL import Image

from utils.project_utils import *
from core.arguments import get_args

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
def embed_index_convolution_batch(query_pts, query_pts_embed_index, kernel_size=10, batch_size=8192):
    """
    Perform convolution and smoothing operations on embedding indices
    Args:
        pts_embed_index: Embedding indices shape (n_pts,) 
        kernel_size: Convolution kernel size, the nearest kernel_size points
    Returns:
        smoothed_pts_embed_index: Convolved embedding indices shape (n_pts,)
    """
    n_pts = len(query_pts)

    smoothed_pts_embed_index = torch.zeros(n_pts, dtype=torch.long, device=query_pts.device)
    for batch_start in range(0, n_pts, batch_size):
        curr_batch_size = min(batch_size, n_pts - batch_start)
        curr_query_pts = query_pts[batch_start:batch_start + curr_batch_size]
        dists = torch.cdist(curr_query_pts, query_pts)
        _, idxs = torch.topk(dists, k=kernel_size, dim=1, largest=False)
        
        smoothed_pts_embed_index[batch_start:batch_start + curr_batch_size] = query_pts_embed_index[idxs].mode(1)[0]
    return smoothed_pts_embed_index


def projection_and_aggregation(args, scene_dir):
    pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
    feature_dir = os.path.join(scene_dir, 'features')
    frame_result_dir = os.path.join(scene_dir, 'sam_result')
    canonical_view_result_file = os.path.join(frame_result_dir, 'canonical_view_index.npy')

    os.makedirs(feature_dir, exist_ok=True)
    
    canonical_view_maskmap = np.load(canonical_view_result_file)
    render_infos = torch.load(os.path.join(feature_dir, 'render_infos.pt'))
    pts = load_ns_point_cloud(pcd_file)
    n_imgs = len(render_infos)
    n_pts = len(pts)
    n_canonical_maskmap_index = len(np.unique(canonical_view_maskmap))
    pts_material_group_map = torch.zeros(n_imgs, n_pts, n_canonical_maskmap_index, dtype=torch.bool, device=args.device, requires_grad=False)
    is_visible = torch.zeros(n_imgs, n_pts, device=args.device, dtype=torch.bool, requires_grad=False)
    
    depth_viz_dir= os.path.join(scene_dir, 'viz', 'depth_viz')
    os.makedirs(depth_viz_dir, exist_ok=True)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(n_imgs):
            maskmap = torch.tensor(render_infos[i]['aligned_maskmap'], device=args.device, dtype=torch.int32)
            h, w = maskmap.shape
            curr_K = render_infos[i]['K']
            w2c = render_infos[i]['w2c']
            depth = render_infos[i]['depth']    

            pts_2d, dists = project_3d_to_2d(pts, w2c, curr_K, return_dists=True)
            pts_2d = np.round(pts_2d).astype(np.int32)

            pts_2d = torch.tensor(pts_2d, device=args.device, dtype=torch.int32)
            dists = torch.tensor(dists, device=args.device, dtype=torch.float32)

            observed_dists = depth
            observed_dists = torch.tensor(observed_dists, device=args.device, dtype=torch.float32)
            image_is_visible = torch.zeros(h, w, device=args.device, dtype=torch.bool)
        
            x,y = pts_2d.T   
            x = x.clamp(0, w-1)
            y = y.clamp(0, h-1)
            is_not_occluded = dists <= observed_dists[y, x] + args.occ_thr
            is_valid = is_not_occluded
            valid_pts = pts_2d[is_valid]
            valid_pts_embed_index = maskmap[valid_pts[:, 1], valid_pts[:, 0]]
            # unique_vals, counts = torch.unique(valid_pts_embed_index, return_counts=True)
            pts_material_group_map[i, is_valid, valid_pts_embed_index] = True
            is_visible[i, is_valid] = True
            image_is_visible[valid_pts[:, 1], valid_pts[:, 0]] = True
            image_is_visible = image_is_visible.to(torch.float32) * 255
            image_is_visible = image_is_visible.to(torch.uint8)
            image_is_visible = Image.fromarray(image_is_visible.cpu().numpy())
            image_is_visible.save(os.path.join(depth_viz_dir, 'is_visible_%s_%d.png' % ('flower', i)))
            
        views_count = pts_material_group_map.sum(dim=0)
        pts_material_group = torch.argmax(views_count, dim=1).to(torch.int32)

    return pts, pts_material_group, is_visible
   
   
def material_grouping(args, scene_dir):
    feature_dir = os.path.join(scene_dir, 'features')
    gs_pts, gs_pts_material_group, is_visible = projection_and_aggregation(args, scene_dir)
    gs_pts = torch.tensor(gs_pts, device=args.device, dtype=torch.float32)
    
    n_visible = is_visible.sum(0)
    is_visible = n_visible > 0
    is_valid = gs_pts_material_group != 0
    is_valid = is_visible & is_valid

    valid_gs_pts_material_group = gs_pts_material_group[is_valid]
    valid_gs_pts = gs_pts[is_valid]
    query_gs_pts = gs_pts.clone()
    
    query_gs_pts_material_group = get_interpolated_values(valid_gs_pts, valid_gs_pts_material_group, query_gs_pts, batch_size=2048, k=1)
    
    kernel_size = args.kernel_size

    # Smooth Material Grouping Results
    while kernel_size >= 4:
        query_gs_pts_material_group = embed_index_convolution_batch(query_gs_pts, query_gs_pts_material_group, kernel_size=kernel_size)
        kernel_size //= 2

    torch.save(query_gs_pts_material_group, os.path.join(feature_dir, 'gs_material_group.pt'))
    
    return query_gs_pts_material_group

if __name__ == '__main__':   

    args = get_args()

    material_grouping(args, args.scene_dir)
    
    print("[INFO] Feature fusion completed.")