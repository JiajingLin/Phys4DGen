
import os
import numpy as np
import json
import open3d as o3d
import gzip
from PIL import Image


def project_3d_to_2d(pts, w2c, K, return_dists=False):
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

def parse_transforms_json(t_file, return_w2c=False, image_width=512, image_height=512):
    with open(t_file, 'r') as f:
        contents = json.load(f)
    
    # Extract camera_angle_x from JSON
    fovx = contents["camera_angle_x"]
    
    # Calculate focal length
    f_x = f_y = image_width / (2 * np.tan(fovx / 2))
    c_x = image_width / 2
    c_y = image_height / 2
    
    # Construct the intrinsic matrix K
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    
    frames = contents["frames"]
    c2ws = []
    img_paths = []
    for idx, frame in enumerate(frames):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        
        c2ws.append(c2w)
        img_paths.append(frame["file_path"])
    if return_w2c:
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        return w2cs, K, img_paths
    
    return c2ws, K, img_paths



def parse_dataparser_transforms_json(dt_file):
    with open(dt_file, "r") as fr:
        dataparser_transforms = json.load(fr)

    ns_transform = np.asarray(dataparser_transforms["transform"])  
    scale = dataparser_transforms["scale"]  
    return ns_transform, scale


def load_ns_point_cloud(pcd_file, viz=False):
    pcd = o3d.io.read_point_cloud(pcd_file)     

    pts = np.asarray(pcd.points)  

    if viz: 
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([cf, pcd])
    return pts

def load_image(img_file, bg_change=255):
    img = np.array(Image.open(img_file))  
    bg = np.array([bg_change, bg_change, bg_change])/255.0
    norm_data = img / 255.0
    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    img = np.array(img)   
    return img


def load_image_RGB(img_file):
    img = np.array(Image.open(img_file))  
    return img


def load_images(img_dir, bg_change=255, return_masks=False):
    img_files = os.listdir(img_dir)
    img_files.sort()  
    imgs = []
    masks = []
    for img_file in img_files:
        # load RGBA image
        img = np.array(Image.open(os.path.join(img_dir, img_file)))  
        if return_masks or bg_change is not None:
            mask = img[:, :, 3] > 0  
            if bg_change is not None: 
                img[~mask] = bg_change  
            masks.append(mask)  
        imgs.append(img[:, :, :3])  
        
    if return_masks:
        return imgs, masks
    return imgs


def load_depths(depth_dir, Ks):
    depth_files = os.listdir(depth_dir)  
    depth_files.sort()  
    depths = []
    for i, depth_file in enumerate(depth_files):
        # load npy.gz depth file
        with gzip.open(os.path.join(depth_dir, depth_file), 'rb') as f: 
            dist = np.load(f)[:, :, 0]  
        if Ks is not None:
            depth = distance_to_depth(dist, Ks[i]) 
        else:
            depth = dist
        depths.append(depth)  
    return depths  


def load_depth(depth_file, Ks):
    with gzip.open(depth_file, 'rb') as f: 
        dist = np.load(f)[:, :, 0]  
    if Ks is not None:
        depth = distance_to_depth(dist, Ks) 
    else:
        depth = dist
    return depth


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


def distance_to_depth(dists, K):
    """Convert distance map to depth map."""
    h, w = dists.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    pts = np.stack([x, y, np.ones_like(x)], axis=1)
    pts = np.dot(pts, np.linalg.inv(K).T)
    divisor = np.linalg.norm(pts, axis=1)
    divisor = divisor.reshape(h, w)
    depth = dists / divisor
    return depth


def get_last_file_in_folder(folder):
    files = os.listdir(folder)
    return os.path.join(folder, sorted(files, reverse=True)[0])


def get_scenes_list(args):
    if args.split != 'all':
        with open(os.path.join(args.data_dir, 'splits.json'), 'r') as f:
            splits = json.load(f)
        if args.split == 'train+val':
            scenes = splits['train'] + splits['val']
        else:
            scenes = splits[args.split]
    else:
        scenes = sorted(os.listdir(os.path.join(args.data_dir, 'scenes')))

    if args.end_idx != -1:
        scenes = scenes[args.start_idx:args.end_idx]
    else:
        scenes = scenes[args.start_idx:]
    return scenes


def unproject_point(pt_2d, depth, c2w, K):
    """Unproject a single point from 2D to 3D (nerfstudio format)."""
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = (pt_2d[0] - cx) / fx
    y = (pt_2d[1] - cy) / fy
    pt_3d = np.array([x, -y, -1])
    pt_3d *= depth[pt_2d[1], pt_2d[0]]
    pt_3d = np.concatenate([pt_3d, np.ones((1,))], axis=0)
    pt_3d = np.dot(c2w, pt_3d)
    pt_3d = pt_3d[:3]
    return pt_3d
