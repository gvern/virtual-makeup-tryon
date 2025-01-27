import cv2
import numpy as np

def overlay_segmentation(parsing_map, colors, alpha=0.5):
    """
    Overlay segmentation map on image for visualization.
    :param parsing_map: Segmentation map
    :param colors: List of colors for each class
    :param alpha: Transparency factor
    :return: Color image with segmentation overlay
    """
    color_seg = np.zeros((parsing_map.shape[0], parsing_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(colors):
        color_seg[parsing_map == label] = color
    return color_seg * alpha + color_seg * 0.5
