import os
import json
import numpy as np
import torch
from utils.simulation_utils.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov, fov2focal


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector) 
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:    
        horizontal_1 = np.array([0.72, 0.37, -0.67])    
    
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)
    
    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates   


def get_camera_position_and_rotation(
    azimuth, elevation, radius, view_center, observant_coordinates
):   
    # get camera position
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )   
    # get rotation matrix
    R = generate_camera_rotation_matrix(    
        view_center - position, -observant_coordinates[:, 2]
    )   
    return position, R


def get_current_radius_azimuth_and_elevation(
    camera_position, view_center, observesant_coordinates
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observesant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observesant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observesant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observesant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observesant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observesant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation

def get_camera_view(
    scene_dir,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(scene_dir, "cameras.json")
    if os.path.exists(cam_path):
        with open(cam_path) as f:
            data = json.load(f)
            
            height, width = data['h'], data['w']
            fovx = data["camera_angle_x"]

            if default_camera_index > -1:
                frame = data['frames'][default_camera_index]

            else:  
                frame = data['frames'][0]  # get data to be modified

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fovy = focal2fov(fov2focal(fovx, width), height)

            return GSCamera(
                colmap_id=0,
                R=R,
                T=T,
                FoVx=fovx,
                FoVy=fovy,
                image=torch.zeros((3, height, width)),  # fake
                gt_alpha_mask=None,
                image_name="fake",
                uid=0,
            )
    else:
        # print("Use default camera view")
        
        assert init_azimuthm is not None
        assert init_elevation is not None
        assert init_radius is not None
        if move_camera: 
            assert delta_a is not None
            assert delta_e is not None
            assert delta_r is not None
            position, R = get_camera_position_and_rotation(
                init_azimuthm + current_frame * delta_a,
                init_elevation + current_frame * delta_e,
                init_radius + current_frame * delta_r,
                center_view_world_space,
                observant_coordinates,
            )
        else:
            position, R = get_camera_position_and_rotation(
                init_azimuthm,
                init_elevation,
                init_radius,
                center_view_world_space,
                observant_coordinates,
            )   
        
        tmp = np.zeros((4, 4))
        tmp[:3, :3] = R
        tmp[:3, 3] = position
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)    
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]
        
        
        width = 512
        height = 512
        fovx = np.deg2rad(49.1)   
        fovy = np.deg2rad(49.1)  

        return GSCamera(
            colmap_id=0,
            R=R,    
            T=T,    
            FoVx=fovx,  
            FoVy=fovy,  
            image=torch.zeros((3, height, width)),  # fake
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        )