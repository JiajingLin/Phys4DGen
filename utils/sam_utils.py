import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import colorsys
import os
from PIL import Image
from utils.project_utils import load_image

# def get_seg_img(mask, image):
#     image = image.copy()



#     return seg_img

def get_seg_img(mask, image):
    try:
        image = image.copy()
        image[mask==0] = np.array([0, 0,  0], dtype=np.uint8) 
        
        x = np.where(mask==True)[1] 
        y = np.where(mask==True)[0] 
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        seg_img = image[y_min:y_max, x_min:x_max, ...] 
        # print("shape: ", seg_img.shape)
        return seg_img
    except Exception as e:
        # print("shape: ", (224, 224, 3))
        
        return np.zeros((224, 224, 3), dtype=np.uint8)

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def embed_clip_sam_tiles(args, image_path, mask_generator, clip_model, clip_preprocess):
    seg_images, seg_map = sam_encoder(args, image_path, mask_generator=mask_generator) 

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']: 
        tiles = seg_images[mode] 
        
        
        tiles = [clip_preprocess(Image.fromarray(tile)) for tile in tiles]
        
        tiles = torch.stack(tiles).to("cuda") 
        with torch.no_grad(): 
            clip_embed = clip_model.encode_image(tiles) 
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True) 
        
        
        clip_embeds[mode] = clip_embed.detach().cpu() 
    
    return clip_embeds, seg_map 

def sam_encoder(args, image_path, mask_generator):
    image = load_image(image_path)
    
    assert image.shape == (args.image_height, args.image_width, 3)
    
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image) 
    
    
    
    
    # if os.path.basename(image_path).startswith('canonical_view'):
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.8)
    
    
    image_name = os.path.basename(image_path)
    
    viz_image_default_dir = os.path.join(args.scene_dir, 'viz', image_name, 'default')
    viz_image_s_dir = os.path.join(args.scene_dir, 'viz', image_name, 's')
    viz_image_m_dir = os.path.join(args.scene_dir, 'viz', image_name, 'm')
    viz_image_l_dir = os.path.join(args.scene_dir, 'viz', image_name, 'l')
    
    os.makedirs(viz_image_default_dir, exist_ok=True)
    os.makedirs(viz_image_s_dir, exist_ok=True)
    os.makedirs(viz_image_m_dir, exist_ok=True)
    os.makedirs(viz_image_l_dir, exist_ok=True)
    
    show_anns(masks_default, alpha=1, image=image, output_filename=os.path.join(viz_image_default_dir, 'viz_default.png'))
    show_anns(masks_s, alpha=1, image=image, output_filename=os.path.join(viz_image_s_dir, 'viz_s.png'))
    show_anns(masks_m, alpha=1, image=image, output_filename=os.path.join(viz_image_m_dir, 'viz_m.png'))
    show_anns(masks_l, alpha=1, image=image, output_filename=os.path.join(viz_image_l_dir, 'viz_l.png'))
    
    def mask2segmap(masks, image, seg_img_save_dir):
        
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32) 
        # for i in range(len(sorted_masks)):
        #     mask = sorted_masks[i]   
        
        
        
        
        
        #     cv2.imwrite(os.path.join(seg_img_save_dir, f'{i}.png'), cv2.cvtColor(pad_seg_img, cv2.COLOR_RGB2BGR))
        
        
        
        
        for i in range(len(sorted_masks)):
            mask = sorted_masks[i]   
            seg_map[sorted_masks[i]['segmentation']] = i 
        
        for i in range(len(sorted_masks)):
            mask = seg_map == i
            seg_img = get_seg_img(mask, image) 
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224)) 
            seg_img_list.append(pad_seg_img) 
            
            cv2.imwrite(os.path.join(seg_img_save_dir, f'{i}.png'), cv2.cvtColor(pad_seg_img, cv2.COLOR_RGB2BGR))
              
            
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3 = b,224,224,3
        
        # seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2)).to('cuda') # b,3,224,224
        
        
        # seg_map: H,W
        # seg_imgs: b,3,224,224
        return seg_imgs, seg_map 

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image, viz_image_default_dir) 
    if len(masks_s) != 0: 
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image, viz_image_s_dir)
    if len(masks_m) != 0: 
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image, viz_image_m_dir)
    if len(masks_l) != 0: 
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image, viz_image_l_dir)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps 


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    
    masks_new = ()
    for masks_lvl in (args): 
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs) 
        masks_lvl = filter(keep_mask_nms, masks_lvl) 

        masks_new += (masks_lvl,)
    return masks_new 


def visualize_segmentation_map(segmentation_map, output_filename):
    unique_labels = np.unique(segmentation_map)
    n_labels = len(unique_labels)
    colors = generate_distinct_colors(n_labels)
    
    
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    
    height, width = segmentation_map.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for label, color in label_to_color.items():
        color_image[segmentation_map == label] = color
        
        
    if output_filename:
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        plt.figure(figsize=(10,10))
        plt.imshow(color_image)
        plt.axis('off')
        plt.tight_layout(pad=0) 
        
        
        legend_handles = [mpatches.Patch(color=np.array(color) / 255, label=str(label)) for label, color in label_to_color.items()]
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  
        colors.append([int(x * 255) for x in rgb])  
    return colors


def show_anns(anns, alpha=1, image=None, output_filename=None):
    if len(anns) == 0:
        return

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    # plt.show()

    
    colors = generate_distinct_colors(len(anns))

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.array(colors[i]) / 255, [alpha]])
        img[m] = color_mask
    ax.imshow(img)
    
    legend_handles = [mpatches.Patch(color=np.array(colors[i]) / 255, label=f'Class {i}') for i in range(len(colors))]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def show_index_map(index_map, index_num, save_path):
    index_num += 1  
    
    colors = generate_distinct_colors(index_num)  
    
    # colors[0] = [255, 255, 255]
    colors[1] = [255, 255, 255]
    
    color_image = torch.zeros((index_map.shape[0], index_map.shape[1], 3), dtype=torch.uint8, device='cuda')  
    indices = [i-1 for i in range(index_num)]
    for index in indices:  
        color_tensor = torch.tensor(colors[index+1], device='cuda', dtype=torch.uint8)  
        color_image[index_map == index] = color_tensor  
        
    color_img = Image.fromarray(color_image.cpu().numpy())  
    color_img.save(save_path)  
