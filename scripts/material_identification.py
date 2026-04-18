import os
import time
import json
import cv2
import openai
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from core.gpt_inference import parse_json_response, gpt4v_multi_images_predict
from utils.project_utils import load_images
from core.arguments import get_args
from my_api_key import OPENAI_API_KEY
from utils.sam_utils import get_seg_img, pad_img
from core.assign_property import assign_physical_property


BASE_SEED = 12345


def gpt_wrapper(gpt_fn, parse_fn, max_tries=0, sleep_time=3):
    """Wrap gpt_fn with error handling and retrying."""
    tries = 0

    time.sleep(sleep_time)
    try:
        gpt_response = gpt_fn(BASE_SEED + tries)
        result = parse_fn(gpt_response)
    except Exception as error:
        print('error:', error)
        result = None
    while result is None and tries < max_tries:
        tries += 1
        time.sleep(sleep_time)
        print('retrying...')
        try:
            gpt_response = gpt_fn(BASE_SEED + tries)
            result = parse_fn(gpt_response)
        except:
            result = None
    return result


def show_img_to_caption(scene_dir, idx_to_caption):
    img_dir = os.path.join(scene_dir, 'images')
    imgs = load_images(img_dir, bg_change=None, return_masks=False)
    img_to_caption = imgs[idx_to_caption]
    plt.imshow(img_to_caption)
    plt.show()
    plt.close()
    return

def predict_object_info_gpt4v_multi_images(args, scene_dir, model_name='gpt-4o-mini'):
    origin_img_path = os.path.join(scene_dir, 'canonical_view.png')

    origin_img = np.array(Image.open(origin_img_path))
    origin_img = origin_img[:, :, :3]
    
    seg_img_save_dir = os.path.join(scene_dir, 'canonical_view_seg')
    os.makedirs(seg_img_save_dir, exist_ok=True)

    canonical_view_maskmap = np.load(os.path.join(scene_dir, 'sam_result', 'canonical_view_index.npy'))
    n_materail_group = len(np.unique(canonical_view_maskmap))
    
    infos = {}
    background_image_paths = None
    partial_image_paths = []
    with torch.no_grad():
        for i in range(n_materail_group):
            mask = canonical_view_maskmap == i
            seg_img = get_seg_img(mask, origin_img)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))

            seg_img_save_path = os.path.join(seg_img_save_dir, f'{i}.png')

            cv2.imwrite(seg_img_save_path, cv2.cvtColor(pad_seg_img, cv2.COLOR_RGB2BGR))
            if i == 0:
                background_image_paths = seg_img_save_path
            else:
                partial_image_paths.append(seg_img_save_path)
        
        complete_image_path = os.path.join(scene_dir, 'canonical_view.png')
        gpt_fn = lambda seed: gpt4v_multi_images_predict(complete_image_path, partial_image_paths, model_name=model_name, seed=seed)
        parse_fn = parse_json_response
        result = gpt_wrapper(gpt_fn, parse_fn)
        
        complete_image_caption = {"complete object caption": result['complete object caption']}

        del result['complete object caption']
        
        if 'simulation type' in result:
            complete_image_caption['simulation type'] = result['simulation type']
            del result['simulation type']

        bg_idx = str(background_image_paths.split('/')[-1].split('.')[0])
        bg_object = f'partial object {bg_idx}'

        infos[bg_object] = {
            "caption": "background",
            "surface materials":
            {
                "material name": "background",
                "physical property": {
                    "Density": 0,
                    "Young's modulus": 0,
                    "Poisson's ratio": 0
                }
            },
            "interior materials":
            {
                "material name": "background",
                "physical property": {
                    "Density": 0,
                    "Young's modulus": 0,
                    "Poisson's ratio": 0
                }
            }
        }
        
        for info, img_path in zip(result.values(), partial_image_paths):
            idx = str(img_path.split('/')[-1].split('.')[0])
            foreground_object = f'partial object {idx}'
            infos[foreground_object] = info
    
        infos = {**complete_image_caption, **infos}

        print('info:' + '-' * 50)
        print(infos)
        print('info:' + '-' * 50)
        
        with open(os.path.join(scene_dir, 'info_new.json'), 'w') as f:
            json.dump(infos, f, indent=4)
                
    return infos
        
if __name__ == '__main__':

    args = get_args()
    
    openai.api_key = OPENAI_API_KEY
    BASE_SEED = args.seed

    start_time = time.time()
    predict_object_info_gpt4v_multi_images(args, args.scene_dir, model_name="gpt-4o")
    assign_physical_property(args)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    print("[INFO] Material identification completed.")