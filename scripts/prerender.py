import argparse
import gzip
from utils.cam_utils import orbit_camera, OrbitCamera
from core.gs_renderer import Renderer, MiniCam
import os
import cv2
import time
import numpy as np

import torch
import torch.nn.functional as F
import json
from core.arguments import get_args
from matplotlib import pyplot as plt
from utils.prerender_utils import orbit_cameras
import subprocess

def prerender_normal(args):


    model_path = os.path.join(args.scene_dir, 'point_cloud.ply')
    render_images_dir = os.path.join(args.scene_dir, 'render_images')
    render_depths_dir = os.path.join(args.scene_dir, 'render_depths')
    sam_frames_dir = os.path.join(args.scene_dir, 'sam_frames')
    transforms_path = os.path.join(args.scene_dir, 'transforms.json')

    os.makedirs(render_images_dir, exist_ok=True)
    os.makedirs(render_depths_dir, exist_ok=True)
    os.makedirs(sam_frames_dir, exist_ok=True)
    
    cam = OrbitCamera(args.image_width, args.image_height, r=args.radius, fovy=args.fovy)
    renderer = Renderer(sh_degree=args.sh_degree)
    transforms = {}
    
    fovy_rad = np.deg2rad(args.fovy)
    fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2) * args.image_width / args.image_height)
    transforms['camera_angle_x'] = fovx_rad
    transforms['frames'] = []
    elevations = np.rad2deg(np.arcsin(np.linspace(0, -np.sqrt(1)/2, args.n_elevation)))
    views_count = 0

    for i, elevation in enumerate(elevations):
        azimuths = np.rad2deg(np.linspace(0, 2 * np.pi, (args.n_azimuth)//(i+1), endpoint=False))
        
        for azimuth in azimuths:
            pose = orbit_camera(elevation, azimuth, args.radius)
            renderer.initialize(model_path)
            cur_cam = MiniCam(pose, args.image_width, args.image_height, cam.fovy, cam.fovx, cam.near, cam.far)
            outputs = renderer.render(cur_cam)
            render_image = outputs["image"].cpu().detach().numpy().astype(np.float32)

            render_image = np.transpose(render_image, (1, 2, 0))
            render_image = np.uint8(render_image*255)
            cv2.imwrite(os.path.join(sam_frames_dir, f'{views_count:05d}.jpg'), cv2.cvtColor(render_image, cv2.COLOR_RGBA2BGRA))
            
            background_mask = ~((render_image[:, :, 0] == 255) & (render_image[:, :, 1] == 255) & (render_image[:, :, 2] == 255))
            alpha_channel = np.uint8(background_mask * 255)
            alpha_channel = np.expand_dims(alpha_channel, axis=2)
            render_image = np.concatenate((render_image, alpha_channel), axis=2)
            render_depth = outputs["depth"].cpu().detach().numpy().astype(np.float32)
            render_depth = np.transpose(render_depth, (1, 2, 0))

            cv2.imwrite(os.path.join(render_images_dir, f'{views_count:05d}.png'), cv2.cvtColor(render_image, cv2.COLOR_RGBA2BGRA))  
            np.save(os.path.join(render_depths_dir, f'{views_count:05d}.npy'), render_depth)

            with open(os.path.join(render_depths_dir, f'{views_count:05d}.npy'), 'rb') as f:
                with gzip.open(os.path.join(render_depths_dir, f'{views_count:05d}.npy.gz'), 'wb') as gz:
                    gz.writelines(f)

            os.remove(os.path.join(render_depths_dir, f'{views_count:05d}.npy'))

            transforms['frames'].append({
                "file_path": os.path.join(render_images_dir, f'{views_count:05d}.png'),
                "transform_matrix": pose.tolist()
            })     

            with open(transforms_path, 'w') as f:
                json.dump(transforms, f, indent=4)
            views_count += 1

    video_path = os.path.join(args.scene_dir, 'render_video.mp4')

    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '10',
        '-i', os.path.join(render_images_dir, '%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ])

    print("[INFO] Pre-rendering completed.")
        


def prerender_random(args):

    
    model_path = os.path.join(args.scene_dir, 'point_cloud.ply')
    render_images_dir = os.path.join(args.scene_dir, 'render_images')
    render_depths_dir = os.path.join(args.scene_dir, 'render_depths')
    render_depths_viz_dir = os.path.join(args.scene_dir, 'viz','render_depths_viz')
    sam_frames_dir = os.path.join(args.scene_dir, 'sam_frames')
    transforms_path = os.path.join(args.scene_dir, 'transforms.json')

    os.makedirs(render_images_dir, exist_ok=True)
    os.makedirs(render_depths_dir, exist_ok=True)
    os.makedirs(sam_frames_dir, exist_ok=True)
    os.makedirs(render_depths_viz_dir, exist_ok=True)

    cam = OrbitCamera(args.image_width, args.image_height, r=args.radius, fovy=args.fovy)
    renderer = Renderer(sh_degree=args.sh_degree)
    transforms = {}
    fovy_rad = np.deg2rad(args.fovy)
    fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2) * args.image_width / args.image_height)
    transforms['camera_angle_x'] = fovx_rad
    transforms['frames'] = []

    canonical_poses = [
        orbit_camera(0, 0, args.radius),
        orbit_camera(0, 180, args.radius),
        orbit_camera(0, 90, args.radius),
        orbit_camera(0, -90, args.radius),
        orbit_camera(89, 0, args.radius),
        orbit_camera(-89, 0, args.radius)
    ]
    poses = orbit_cameras(args.radius, args.seed, num_views=args.prerender_view_num) 
    poses = canonical_poses + poses
    renderer.initialize(model_path)    
    views_count = 0
    
    for pose in poses:
        cur_cam = MiniCam(pose, args.image_width, args.image_height, cam.fovy, cam.fovx, cam.near, cam.far)
        outputs = renderer.render(cur_cam)
        render_image = outputs["image"].cpu().detach().numpy().astype(np.float32)
        render_image = np.transpose(render_image, (1, 2, 0))
        render_image = np.uint8(render_image*255)
        cv2.imwrite(os.path.join(sam_frames_dir, f'{views_count:05d}.jpg'), cv2.cvtColor(render_image, cv2.COLOR_RGBA2BGRA))
        
        background_mask = ~((render_image[:, :, 0] == 255) & (render_image[:, :, 1] == 255) & (render_image[:, :, 2] == 255))
        alpha_channel = np.uint8(background_mask * 255)
        alpha_channel = np.expand_dims(alpha_channel, axis=2)
        render_image = np.concatenate((render_image, alpha_channel), axis=2)
        cv2.imwrite(os.path.join(render_images_dir, f'{views_count:05d}.png'), cv2.cvtColor(render_image, cv2.COLOR_RGBA2BGRA)) 

        render_depth = outputs["depth"].cpu().detach().numpy().astype(np.float32)
        render_depth = np.transpose(render_depth, (1, 2, 0))
         
        np.save(os.path.join(render_depths_dir, f'{views_count:05d}.npy'), render_depth)

        with open(os.path.join(render_depths_dir, f'{views_count:05d}.npy'), 'rb') as f:
            with gzip.open(os.path.join(render_depths_dir, f'{views_count:05d}.npy.gz'), 'wb') as gz:
                gz.writelines(f)

        os.remove(os.path.join(render_depths_dir, f'{views_count:05d}.npy'))
        depth_min = render_depth.min()
        depth_max = render_depth.max()
        render_depth_normalized = (render_depth - depth_min) / (depth_max - depth_min)
        render_depth_gray = (render_depth_normalized * 255).astype(np.uint8)
        render_depth_gray = render_depth_gray.squeeze()

        plt.imsave(os.path.join(render_depths_viz_dir, f'{views_count:05d}.png'), render_depth_gray, cmap='gray')

        transforms['frames'].append({
            "file_path": os.path.join(render_images_dir, f'{views_count:05d}.png'),
            "transform_matrix": pose.tolist()
        })     

        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=4)
        views_count += 1
        
    video_path = os.path.join(args.scene_dir, 'render_video.mp4')

    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', '10',
        '-i', os.path.join(render_images_dir, '%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ])
    
    print("[INFO] Pre-rendering completed.") 
    
if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    if args.prerender_mode == "normal":
        prerender_normal(args)
    elif args.prerender_mode == "random":
        prerender_random(args)
    else:
        raise ValueError(f"Invalid prerender mode: {args.prerender_mode}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")