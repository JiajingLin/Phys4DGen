import os
import shutil
import random
import math
import numpy as np
import torch


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera_from_sphere(radius, seed=1234, frame=100, upper_views=False, target=None, opengl=True):
    # radius: scalar
    # return: [4, 4], camera pose matrix
    
    
    
    

    unit_x, unit_y, unit_z = sample_point_from_unit_sphere(seed, frame, upper_views)
    
    x, y, z = unit_x * radius, unit_y * radius, unit_z * radius
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

# elevation & azimuth to pose (cam2world) matrix
def orbit_cameras(radius, seed=1234, num_views=100, upper_views=False, target=None, opengl=True):
    # radius: scalar
    # return: [4, 4], camera pose matrix
    
    poses = []
    for frame in range(num_views):
        poses.append(orbit_camera_from_sphere(radius, seed, frame, upper_views, target, opengl))
    return poses
    


def sample_on_unit_sphere(num_views, upper_views=False):
    points = []
    for _ in range(num_views):
        u = random.random()  # [0,1)
        v = random.random()  # [0,1)
        
        theta = 2 * np.pi * u
        phi = np.arccos(1 - 2*v)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        if upper_views:
            
            
            
            if z < 0:
                z = -z

        points.append([x,y,z])
    return np.array(points)


# non uniform sampling when stretched or squeezed sphere
def sample_point_from_unit_sphere(seed, frame_current, upper_views=False):
    seed = (2654435761 * (seed + 1)) ^ (805459861 * (frame_current + 1))
    rng = random.Random(seed) # random number generator

    # sample random angles
    theta = rng.random() * 2 * math.pi
    phi = math.acos(1 - 2 * rng.random()) # ensure uniform sampling from unit sphere

    # uniform sample from unit sphere, given theta and phi
    unit_x = math.cos(theta) * math.sin(phi)
    unit_y = math.sin(theta) * math.sin(phi)
    unit_z = abs( math.cos(phi) ) if upper_views else math.cos(phi)

    return unit_x, unit_y, unit_z
