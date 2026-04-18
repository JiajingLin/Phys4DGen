import os
import time
import open3d as o3d
import cv2
import torch
import numpy as np
import json
from tqdm import tqdm
import taichi as ti
from core.arguments import get_args

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Utils
from mpm_solver_warp.mpm_utils import *
from utils.load_utils import *
from utils.simulation_utils.decode_param import *
from utils.simulation_utils.transformation_utils import *
from utils.simulation_utils.camera_view_utils import *
from utils.simulation_utils.render_utils import *
from utils.simulation_utils.export_utils import *
from utils.simulation_utils.filling_utils import *
from utils.simulation_utils.binding_utils import *

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time

if __name__ == "__main__":
    args = get_args()
    args.output_name = os.path.basename(args.scene_dir)
    args.predict_property_path = os.path.join(args.scene_dir, "predict_property_new.json")
    args.model_dir = os.path.join(args.scene_dir, "model")

    args.viz_dir = os.path.join(args.scene_dir, "viz")
    args.config = os.path.join("config", args.output_name + "_config.json")
    args.gs_load = os.path.join(args.scene_dir, "point_cloud.ply")
    args.particle_load = os.path.join(args.scene_dir, "particle.ply")
    args.binding_path = os.path.join(args.scene_dir, "bindings.pt")

    if not os.path.exists(args.gs_load):
        AssertionError("GS Model does not exist!")
    if not os.path.exists(args.particle_load):
        AssertionError("Particle Model does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
        
    timer = Timer()
    timer.start()

    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    print(f"Loading gaussians from {args.gs_load}...")
    gaussians = load_checkpoint(args.gs_load, sh_degree=args.sh_degree)
    gs_num = gaussians.get_xyz.shape[0]
    
    """GS-particle binding"""
    if args.particle_binding:
        if not os.path.exists(args.binding_path):
            AssertionError("Binding path does not exist!")
        weight_mat = load_sparse_chunks(args.binding_path).to(device="cuda")
        
    with open(args.predict_property_path, 'r') as f:
        predict_property = json.load(f)
        
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") 
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)   
    initspace_gs_pos = params["pos"]
    initspace_gs_cov = params["cov3D_precomp"]
    initspace_gs_screen_points = params["screen_points"]
    initspace_gs_opacity = params["opacity"]
    initspace_gs_shs = params["shs"]
    
    initspace_gs_rotation = gaussians.get_rotation
    initspace_gs_scaling = gaussians.get_scaling
    
    # select a sim area and save params of unslected particles
    gs_material_group = predict_property['material_group']['gs']
    gs_material_group = torch.tensor(gs_material_group, dtype=torch.int32, device="cuda")
    
    unselected_gs_pos, unselected_gs_cov, unselected_gs_opacity, unselected_gs_shs = (
        None,
        None,
        None,
        None,
    )
    if material_params["sim_object"] is not None:
        sim_object_captions = material_params["sim_object"]
        captions = predict_property['captions']
        
        sim_object_material_group = []
        gs_mask = torch.zeros(gs_num, dtype=torch.bool).to(device="cuda")
        for object_caption in sim_object_captions:
            if object_caption in captions:
                sim_object_material_group.append(captions.index(object_caption))
            else:
                
                raise ValueError(f"Object caption '{object_caption}' not found in captions.")
        for index in sim_object_material_group: 
            gs_mask = torch.logical_or(gs_mask, gs_material_group == index) 
        
        unselected_gs_pos = initspace_gs_pos[~gs_mask, :]
        unselected_gs_cov = initspace_gs_cov[~gs_mask, :]
        unselected_gs_opacity = initspace_gs_opacity[~gs_mask, :]
        unselected_gs_shs = initspace_gs_shs[~gs_mask, :]
        
        
        initspace_gs_pos = initspace_gs_pos[gs_mask, :]
        initspace_gs_cov = initspace_gs_cov[gs_mask, :]
        initspace_gs_opacity = initspace_gs_opacity[gs_mask, :]
        initspace_gs_shs = initspace_gs_shs[gs_mask, :]   
        
        gs_material_group = gs_material_group[gs_mask]  
    
    else:
        gs_mask = torch.ones(gs_num, dtype=torch.bool).to(device="cuda")
        
    if args.debug:
        particle_position_tensor_to_ply(
            initspace_gs_pos,
            "./log/initspace_gs_pos.ply",
        )   
    
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    ) 
    rotated_gs_pos = apply_rotations(initspace_gs_pos, rotation_matrices)  
        
    if args.debug:
        particle_position_tensor_to_ply(rotated_gs_pos, "./log/rotated_gs_pos.ply") 
    
    transformed_gs_pos, scale_origin, original_mean_pos = transform2origin(rotated_gs_pos)    
    transformed_gs_pos = shift2center111(transformed_gs_pos)  
    
    mpmspace_gs_pos = transformed_gs_pos
    mpmspace_gs_cov = apply_cov_rotations(initspace_gs_cov, rotation_matrices) 
    mpmspace_gs_cov = scale_origin * scale_origin * mpmspace_gs_cov   


    if args.debug:
        test_pos = mpmspace_gs_pos.clone()
        particle_position_tensor_to_ply(
            test_pos,
            "./log/mpmspace_gs_pos.ply",
        )
        
    gs_num = mpmspace_gs_pos.shape[0]
    device = "cuda"
    
    if material_params["phyfilling"]:
        
        args.particle_load = os.path.join(args.scene_dir, "particle.ply")
        print(f"Loading particle from {args.particle_load}...")
        if not os.path.exists(args.particle_load):
            AssertionError("Particle Model does not exist!")
        initspace_particle_pos = load_particle(args.particle_load) 
        
        pts_material_group = predict_property['material_group']['pts']
        
        pts_material_group = torch.tensor(pts_material_group, dtype=torch.int32, device="cuda")
        if material_params["sim_object"] is not None:
            
            pts_mask = torch.zeros(initspace_particle_pos.shape[0], dtype=torch.bool).to(device="cuda")
            for index in sim_object_material_group: 
                pts_mask = torch.logical_or(pts_mask, pts_material_group == index)
            initspace_particle_pos = initspace_particle_pos[pts_mask, :] 
            
            pts_material_group = pts_material_group[pts_mask]
            
            if args.particle_binding == True:  
                weight_mat = gaussian_rebinding(initspace_gs_pos, initspace_gs_cov, initspace_particle_pos) 
        else:
            pts_mask = torch.ones(initspace_particle_pos.shape[0], dtype=torch.bool).to(device="cuda")
        
        
        rotated_particle_pos = apply_rotations(initspace_particle_pos, rotation_matrices)
        transformed_particle_pos = (rotated_particle_pos - original_mean_pos) * scale_origin
        transformed_particle_pos = shift2center111(transformed_particle_pos)
        
        mpmspace_particle_pos = transformed_particle_pos
        
        mpm_init_pos = torch.cat([mpmspace_gs_pos, mpmspace_particle_pos], dim=0)
        
        mpm_material_group = torch.cat([gs_material_group, pts_material_group], dim=0)
        
        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply") 
    else:
        
        mpm_init_pos =  mpmspace_gs_pos.to(device=device)    
        
        mpm_material_group = gs_material_group

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(     
        mpm_init_pos,
        material_params["n_grid"],              
        material_params["grid_lim"] / material_params["n_grid"],        
        unifrom=material_params["material"] == "sand",      
    ).to(device=device)     

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(args, 10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        None,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )   
    mpm_solver.set_parameters_dict(material_params, predict_property_path=args.predict_property_path, gs_mask=gs_mask, pts_mask=pts_mask if material_params["phyfilling"] else None) 
    mpm_solver.set_material_info(predict_property['captions'], mpm_material_group)
    
    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(args, mpm_solver, bc_params, time_params) 

    mpm_solver.finalize_mu_lam()    

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (  
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    
    camera_info = get_camera_info(gaussians.get_xyz, rotation_matrices)
    
    (
        viewpoint_center_worldspace,    
        observant_coordinates,          
    ) = get_center_view_worldspace_and_observant_coordinate(    
        mpm_space_viewpoint_center,     
        mpm_space_vertical_upward_axis,  
        rotation_matrices,  
        camera_info["scale_origin"],       
        camera_info["original_mean_pos"],
    )

    # run the simulation
    substep_dt = time_params["substep_dt"]  
    frame_dt = time_params["frame_dt"]  
    frame_num = time_params["frame_num"]    
    step_per_frame = int(frame_dt / substep_dt) 
    height = None
    width = None
    
    opacity_render = initspace_gs_opacity
    shs_render = initspace_gs_shs
    
    cur_mpmsapce_particle_pos = mpm_solver.export_particle_x_to_torch().to(device)
    cur_mpmspace_gs_pos = mpmspace_gs_pos
    t0_mpmspace_gs_cov3D = mpmspace_gs_cov

    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            scene_dir = args.scene_dir,   
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )   
        
        rasterize = initialize_resterize(
            current_camera, gaussians, pipeline, background
        )
        
        pre_mpmspace_particle_pos = cur_mpmsapce_particle_pos.clone()
        pre_mpmspace_gs_pos = cur_mpmspace_gs_pos.clone()

        for step in range(step_per_frame):  
            mpm_solver.p2g2p(frame, substep_dt, device=device) 
        
        if args.render_img:
            if args.particle_binding == True and material_params["phyfilling"] == True:      
                cur_mpmspace_particle_pos = mpm_solver.export_particle_x_to_torch().to(device)   
                
                cur_mpmspace_particle_F = mpm_solver.export_particle_F_to_torch()   
                cur_mpmspace_particle_R = mpm_solver.export_particle_R_to_torch()   
                
                cur_mpmspace_gs_pos = compute_bindings_xyz(cur_mpmspace_particle_pos, pre_mpmspace_particle_pos, pre_mpmspace_gs_pos, weight_mat).contiguous()
                cur_mpmspace_gs_deform_grad = compute_bindings_F(cur_mpmspace_particle_F, weight_mat).contiguous()   
                
                cur_mpmspace_gs_cov3D = update_cov_by_F(t0_mpmspace_gs_cov3D, cur_mpmspace_gs_deform_grad, device)
                cur_mpmspace_gs_cov3D = cur_mpmspace_gs_cov3D.view(-1, 6)
            else:
                cur_mpmspace_particle_pos = mpm_solver.export_particle_x_to_torch().to(device)   
                
                cur_mpmspace_particle_F = mpm_solver.export_particle_F_to_torch()   
                cur_mpmspace_particle_R = mpm_solver.export_particle_R_to_torch()   
                cur_mpmspace_gs_pos = cur_mpmspace_particle_pos[:gs_num] 
                cur_mpmspace_gs_deform_grad = cur_mpmspace_particle_F[:gs_num]
                
                cur_mpmspace_gs_cov3D = update_cov_by_F(t0_mpmspace_gs_cov3D, cur_mpmspace_gs_deform_grad, device)
                cur_mpmspace_gs_cov3D = cur_mpmspace_gs_cov3D.view(-1, 6)
            
            """Transform the current space's Gaussian kernels to the original space"""
            
            cur_initspace_gs_pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(cur_mpmspace_gs_pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )   
            cur_mpmspace_gs_cov3D = cur_mpmspace_gs_cov3D / (scale_origin * scale_origin)   
            cur_initspace_gs_cov3D = apply_inverse_cov_rotations(cur_mpmspace_gs_cov3D, rotation_matrices)   
            cur_initspace_opacity = opacity_render
            cur_initspace_shs = shs_render
            
            if material_params["sim_object"] is not None:
                cur_initspace_gs_pos = torch.cat([cur_initspace_gs_pos, unselected_gs_pos], dim=0)
                cur_initspace_gs_cov3D = torch.cat([cur_initspace_gs_cov3D, unselected_gs_cov], dim=0)
                cur_initspace_opacity = torch.cat([cur_initspace_opacity, unselected_gs_opacity], dim=0)
                cur_initspace_shs = torch.cat([cur_initspace_shs, unselected_gs_shs], dim=0)
                
            colors_precomp = convert_SH(cur_initspace_shs, current_camera, gaussians, cur_initspace_gs_pos, None)   
            rendering, raddi, rendered_depth, rendered_alpha = rasterize(
                means3D=cur_initspace_gs_pos,
                means2D=initspace_gs_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=cur_initspace_opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cur_initspace_gs_cov3D,    
            )
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            
            os.makedirs(os.path.join(args.viz_dir, "dynamic_viz"), exist_ok=True)
            cv2.imwrite(
                os.path.join(os.path.join(args.viz_dir, "dynamic_viz"), f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )
            
            if args.save_simulation_model:
                
                path = os.path.join(args.model_dir, "dynamic")
                os.makedirs(path, exist_ok=True)
                rotation, scaling = build_rotation_scaling_from_covariance(cur_initspace_gs_cov3D)    
                
                gaussians.save_frame_ply(path, frame, cur_initspace_gs_pos, rotation, scaling, cur_initspace_opacity, cur_initspace_shs)  
        else:
            particle_pos = mpm_solver.export_particle_x_to_torch().to(device)
            particle_position_tensor_to_ply(particle_pos, f"./log/{frame}.ply")
            
    print(f"Total time: {timer.get_elapsed_time()}s")

    if args.render_img and args.compile_video:
        if args.fps is not None:
            fps = args.fps
        else:
            fps = 7
        os.system(
            f"ffmpeg -framerate {fps} -i {os.path.join(args.viz_dir, 'dynamic_viz')}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {os.path.join(args.viz_dir, 'dynamic_viz')}/output.mp4"
        )