import cv2
import numpy as np
from typing import List, Tuple
import colorsys

def get_color_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7
) -> np.ndarray:
    """Generate color palette for visualization"""
    hues = np.linspace(0, 1, int(n_colors))
    colors = [(0.0, 0.0, 0.0)]  # Start with black

    for h in hues[1:]:
        colors.append(colorsys.hls_to_rgb(h, lightness, saturation))

    return (255 * np.array(colors)).astype(np.uint8)

def colorize_mask(
    image: np.ndarray,
    index_mask: np.ndarray,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    max_idx = index_mask.max() 
    palette = get_color_palette(max_idx + 1) 

    color_mask = palette[index_mask.astype(int)] 

    # Blend image and mask
    blended = cv2.addWeighted(
        image.astype(np.uint8), alpha,
        color_mask.astype(np.uint8), 1 - alpha,
        0
    ) 

    return blended, color_mask