import os
import json
import torch
import open_clip
import numpy as np
import open3d as o3d
import matplotlib as mpl

from core.gpt_inference import parse_material_list
from utils.project_utils import load_ns_point_cloud
from core.arguments import get_args
from utils.voxel_grid_utils import *
from utils.simulation_utils.binding_utils import *
from core.phyobj import PhyObj
from utils.load_utils import *

def predict_physical_property_query(args):
    perperty_info_file = os.path.join(args.scene_dir, 'info_new.json')
    gs_file = os.path.join(args.scene_dir, 'point_cloud.ply')
    pts_file = os.path.join(args.scene_dir, 'particle.ply')
    grid_info_file = os.path.join(args.scene_dir, 'features', 'grid_info.pt')

    phyobj = PhyObj(perperty_info_file)    
    grid_info = torch.load(grid_info_file)
    
    gaussians = load_checkpoint(gs_file, sh_degree=args.sh_degree)
    gs_material_group = torch.load(os.path.join(args.scene_dir, 'features', 'gs_material_group.pt'))
    gs = gaussians.get_xyz  
    pts = torch.tensor(load_ns_point_cloud(pts_file), device=args.device)
    
    grid_material_group = grid_info["grid_material_group"]
    is_surface = grid_info["is_surface"]
    is_internal = grid_info["is_internal"]
    grid_cell_size = grid_info["grid_cell_size"]
    xyz1 = grid_info["xyz1"]
    
    result = {"Density": {}, "Young's modulus": {}, "Poisson's ratio": {}}
    valid_cells = ((pts - xyz1) / grid_cell_size).round().to(torch.int32)
    
    for property_name in result.keys():
        surface_property_vals = torch.tensor(phyobj.get_partobjs_surface_property(property_name), dtype=torch.float32, device=args.device)
        internal_property_vals = torch.tensor(phyobj.get_partobjs_internal_property(property_name), dtype=torch.float32, device=args.device)
        
        grid_property_vals = torch.zeros(grid_material_group.shape, dtype=surface_property_vals.dtype, device=args.device)
        
        grid_surface_property_vals = torch.take(surface_property_vals, grid_material_group.to(torch.long))
        grid_internal_property_vals = torch.take(internal_property_vals, grid_material_group.to(torch.long))

        grid_property_vals[is_surface] = grid_surface_property_vals[is_surface]
        grid_property_vals[is_internal] = grid_internal_property_vals[is_internal]
        
        pts_property_vals = grid_property_vals[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]]
        
        gs_property_vals = surface_property_vals[gs_material_group]
        
        gs_pts_property_vals = torch.cat([gs_property_vals, pts_property_vals], dim=0)
        result[property_name] = gs_pts_property_vals.cpu().numpy()
    
    result['captions'] = phyobj.get_partobjs_caption()
    gs_pts_material_group = torch.cat([gs_material_group, grid_material_group[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]]], dim=0)
    result['seg_index'] = gs_pts_material_group
    
    predict_property_save_path = os.path.join(args.scene_dir, 'predict_property.json')
    
    save_result = {}
    save_result["Density"] = result["Density"].tolist()
    save_result["Young's modulus"] = result["Young's modulus"].tolist()
    save_result["Poisson's ratio"] = result["Poisson's ratio"].tolist()
    save_result["seg_index"] = result["seg_index"].tolist()
    save_result["captions"] = result["captions"]

    

    with open(predict_property_save_path, 'w') as f:
        json.dump(save_result, f, indent=4)

    gs_pts = torch.cat([gs, pts], dim=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gs_pts.cpu().detach().numpy())
    o3d.io.write_point_cloud(os.path.join(args.scene_dir, 'gs_pts.ply'), pcd)
    
    return result

def assign_physical_property(args):

    perperty_info_file = os.path.join(args.scene_dir, 'info_new.json')
    gs_file = os.path.join(args.scene_dir, 'point_cloud.ply')
    pts_file = os.path.join(args.scene_dir, 'particle.ply')
    grid_info_file = os.path.join(args.scene_dir, 'features', 'grid_info.pt')

    phyobj = PhyObj(perperty_info_file)    
    grid_info = torch.load(grid_info_file)
    
    gaussians = load_checkpoint(gs_file, sh_degree=args.sh_degree)
    gs_material_group = torch.load(os.path.join(args.scene_dir, 'features', 'gs_material_group.pt'))
    gs = gaussians.get_xyz
    
    pts = torch.tensor(load_ns_point_cloud(pts_file), device=args.device)
    
    grid_material_group = grid_info["grid_material_group"]
    is_surface = grid_info["is_surface"]
    is_internal = grid_info["is_internal"]
    grid_cell_size = grid_info["grid_cell_size"]
    xyz1 = grid_info["xyz1"]
    
    result = {"Density": {}, "Young's modulus": {}, "Poisson's ratio": {}}
    

    valid_cells_1 = torch.nonzero(grid_material_group != 0)
    valid_cells = ((pts - xyz1) / grid_cell_size).round().to(torch.int32)

    for property_name in result.keys():
        surface_property_vals = torch.tensor(phyobj.get_partobjs_surface_property(property_name), dtype=torch.float32, device=args.device)
        internal_property_vals = torch.tensor(phyobj.get_partobjs_internal_property(property_name), dtype=torch.float32, device=args.device)
        
        grid_property_vals = torch.zeros(grid_material_group.shape, dtype=surface_property_vals.dtype, device=args.device)

        grid_surface_property_vals = torch.take(surface_property_vals, grid_material_group.to(torch.long))

        grid_internal_property_vals = torch.take(internal_property_vals, grid_material_group.to(torch.long))
        grid_property_vals[is_surface] = grid_surface_property_vals[is_surface]
        # print(torch.unique(grid_property_vals[is_surface]))
        grid_property_vals[is_internal] = grid_internal_property_vals[is_internal]
        # print(torch.unique(grid_property_vals[is_internal]))

        pts_property_vals = grid_property_vals[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]]
        # print(torch.unique(pts_property_vals))
        
        gs_property_vals = surface_property_vals[gs_material_group]
        

        result[property_name]["pts"] = pts_property_vals.cpu().numpy()

        result[property_name]["gs"] = gs_property_vals.cpu().numpy()
    
    result['captions'] = phyobj.get_partobjs_caption()
    result['material_group'] = {}
    result['material_group']['pts'] = grid_material_group[valid_cells[:, 0], valid_cells[:, 1], valid_cells[:, 2]]
    result['material_group']['gs'] = gs_material_group
    
    predict_property_save_path = os.path.join(args.scene_dir, 'predict_property_new.json')

    save_result = {
        "Density": {
            "pts": result["Density"]["pts"].tolist(),
            "gs": result["Density"]["gs"].tolist()
        },
        "Young's modulus": {
            "pts": result["Young's modulus"]["pts"].tolist(),
            "gs": result["Young's modulus"]["gs"].tolist()
        },
        "Poisson's ratio": {
            "pts": result["Poisson's ratio"]["pts"].tolist(),
            "gs": result["Poisson's ratio"]["gs"].tolist()
        },
        "material_group": {
            "pts": result["material_group"]["pts"].tolist(),
            "gs": result["material_group"]["gs"].tolist()
        },
        "captions": result["captions"]
    }
    

    with open(predict_property_save_path, 'w') as f:
        json.dump(save_result, f, indent=4)
            
    return result

  
@torch.no_grad()
def predict_physical_property_query_viz(args):
    scene_dir = args.scene_dir
    pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
    perperty_info_file = os.path.join(scene_dir, 'info_new.json')
    
    pts = load_ns_point_cloud(pcd_file)
    gs_material_group = torch.load(os.path.join(scene_dir, 'features', 'gs_material_group.pt'))
    
    phyobj = PhyObj(perperty_info_file)   
    
    result = {"Density": {}, "Young's modulus": {}, "Poisson's ratio": {}}
    

    for property_name in result.keys():


        surface_property_vals = torch.tensor(phyobj.get_partobjs_surface_property(property_name), dtype=torch.float32, device=args.device)
        internal_property_vals = torch.tensor(phyobj.get_partobjs_internal_property(property_name), dtype=torch.float32, device=args.device)
        
        gs_surface_property_vals = surface_property_vals[gs_material_group]
        gs_internal_property_vals = internal_property_vals[gs_material_group]
        

        result[property_name]["surface"] = gs_surface_property_vals.cpu().numpy()
        result[property_name]["internal"] = gs_internal_property_vals.cpu().numpy()
        

    result['captions'] = phyobj.get_partobjs_caption()

    
    return result

if __name__ == '__main__':

    args = get_args()
    result = assign_physical_property(args)
        
    print("[INFO] Predict physical property completed.")
    
