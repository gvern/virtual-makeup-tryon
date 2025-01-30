# utils/visualization.py

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def overlay_segmentation(image: np.ndarray, landmarks: List[Tuple[int, int]], makeup_types: List[str],
                         makeup_colors: Dict[str, Tuple[int, int, int]], thickness: int =2) -> np.ndarray:
    """
    Overlays facial segmentation outlines on the image based on landmarks and makeup types.

    :param image: Original image in BGR format.
    :param landmarks: List of facial landmarks as (x, y) tuples.
    :param makeup_types: List of makeup types to overlay.
    :param makeup_colors: Dictionary mapping makeup types to BGR color tuples.
    :param thickness: Thickness of the segmentation lines.
    :return: Image with segmentation overlays.
    """
    overlay_image = image.copy()

    for makeup_type in makeup_types:
        color = makeup_colors.get(makeup_type, (0, 255, 0))  # Default to green if not specified
        # Find the configuration for the makeup type
        config = next((mt for mt in MAKEUP_TYPES_CONFIG if mt.name == makeup_type), None)
        if not config:
            logging.warning(f"No configuration found for makeup type: {makeup_type}. Skipping overlay.")
            continue
        for region_name, landmark_pairs in config.facemesh_regions.items():
            # Extract unique landmark indices
            indices = list({idx for pair in landmark_pairs for idx in pair})
            region_landmarks = [landmarks[i] for i in indices]

            # Compute convex hull
            hull = cv2.convexHull(np.array(region_landmarks))
            cv2.polylines(overlay_image, [hull], isClosed=True, color=color, thickness=thickness)
            logging.debug(f"Overlayed {makeup_type} - {region_name} with color {color}")

    return overlay_image
