import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as iio
from torch import nn

import matplotlib.pyplot as plt
from core.arguments import get_args
import open_clip

from utils.project_utils import *
from utils.sam_utils import *
from utils.seed_utils import *
from utils.visualization_utils import colorize_mask

CLIP_BACKBONE = 'ViT-B-16'
CLIP_CHECKPOINT = 'datacomp_xl_s13b_b90k'
CLIP_INPUT_SIZE = 224
CLIP_OUTPUT_SIZE = 512

def get_seg_img_clip_embeding(args, img_idx, img, maskmap, clip_model, clip_preprocess):
    seg_img_viz_dir = os.path.join(args.scene_dir, 'viz', 'seg_img_viz', f'img_{img_idx}')
    os.makedirs(seg_img_viz_dir, exist_ok=True)
    
    materail_group_list = np.unique(maskmap)

    
    tiles = []
    for materail_group in materail_group_list:

        mask = maskmap == materail_group

        seg_img = get_seg_img(mask, img)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))

        cv2.imwrite(os.path.join(seg_img_viz_dir, f'seg_img_{materail_group}.png'), pad_seg_img)
        tiles.append(pad_seg_img)
        
    tiles = [clip_preprocess(Image.fromarray(tile)) for tile in tiles]
    tiles = torch.stack(tiles).to("cuda")
    with torch.no_grad():
        clip_embeding = clip_model.encode_image(tiles)
    clip_embeding /= clip_embeding.norm(dim=-1, keepdim=True)

    clip_embeding = clip_embeding.detach().cpu()

    return clip_embeding


def align_maskmap(args, img_idx, img, maskmap, canonical_view_clip_embeding, clip_model, clip_preprocess):
    img_clip_embeding = get_seg_img_clip_embeding(args, img_idx, img, maskmap, clip_model, clip_preprocess)

    maskmap = torch.tensor(maskmap, device=args.device)
    materail_group_list = torch.unique(maskmap)
    align_maskmap = torch.full_like(maskmap, -1)
    similarity = canonical_view_clip_embeding @ img_clip_embeding.T
    max_indices = torch.argmax(similarity, dim=0).to(torch.int32)


    for j, index in zip(materail_group_list, max_indices):
        align_maskmap[maskmap == j] = index

    return align_maskmap


def clip_fusion(args, scene_dir):
    scene_name = os.path.basename(scene_dir)
    args.scene_dir = scene_dir
    pcd_file = os.path.join(scene_dir, 'point_cloud.ply')
    t_file = os.path.join(scene_dir, 'transforms.json')
    canonical_view_path = os.path.join(scene_dir, 'canonical_view.png')
    img_dir = os.path.join(scene_dir, 'render_images')
    depth_dir = os.path.join(scene_dir, 'render_depths')
    feature_dir = os.path.join(scene_dir, 'features')
    viz_mask_dir = os.path.join(scene_dir, 'viz', 'aligh_mask_viz')
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(viz_mask_dir, exist_ok=True)
    frame_result_dir = os.path.join(scene_dir, 'sam_result')
    frame_result_files = os.listdir(frame_result_dir)
    
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)

    canonical_view_result_file = [file for file in frame_result_files if file.startswith('canonical_view') and file.endswith('.npy')][0]
    render_result_files = [file for file in frame_result_files if not file.startswith('canonical_view') and file.endswith('.npy')]
    render_result_files.sort()

    pts = load_ns_point_cloud(pcd_file)

    w2cs, K, img_paths = parse_transforms_json(t_file, return_w2c=True, image_width=args.image_width, image_height=args.image_height)
    print('scene: %s, points: %d' % (scene_name, len(pts)))
    

    canonical_view_img = load_image(canonical_view_path)
    canonical_view_maskmap = np.load(os.path.join(frame_result_dir, canonical_view_result_file))
    blended, color_mask = colorize_mask(canonical_view_img, canonical_view_maskmap)
    iio.imwrite(os.path.join(viz_mask_dir, 'canonical_view_mask.png'), blended)
    canonical_view_clip_embeding = get_seg_img_clip_embeding(args, -1, canonical_view_img, canonical_view_maskmap, clip_model, clip_preprocess)
    
    render_infos = []
    for i in range(len(img_paths)):
        render_info = {}
        suffix = img_paths[i].split('/')[-1].split('.')[0]
        maskmap_file = os.path.join(frame_result_dir, f'{suffix}.npy')
        maskmap = np.load(maskmap_file)
        depth_file = os.path.join(depth_dir, f'{suffix}.npy.gz')
        depth = load_depth(depth_file, Ks=None)

        img = load_image(img_paths[i])
        if args.clip_align:
            aligned_maskmap = align_maskmap(args, i, img, maskmap, canonical_view_clip_embeding, clip_model, clip_preprocess)
            maskmap = aligned_maskmap.cpu().numpy()
        
        blended, color_mask = colorize_mask(img, maskmap)
        iio.imwrite(os.path.join(viz_mask_dir, f'{suffix}.png'), blended)
        
        render_info['file_suffix'] = suffix
        render_info['aligned_maskmap'] = maskmap
        render_info['w2c'] = w2cs[i]
        render_info['K'] = K
        render_info['depth'] = depth
        render_infos.append(render_info)
        
    torch.save(render_infos, os.path.join(feature_dir, 'render_infos.pt'))


if __name__ == "__main__":
    args = get_args()
       
    # clip_fusion(args, args.scene_dir, clip_model, clip_preprocess)
    clip_fusion(args, args.scene_dir)
    
    print("[INFO] CLIP Fusion completed.")