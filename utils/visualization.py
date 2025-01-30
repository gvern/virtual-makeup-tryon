# utils/visualization.py

import cv2
import numpy as np
from src.makeup_config import MAKEUP_TYPES_CONFIG
import logging

def overlay_segmentation(image, landmarks, makeup_types=['Lipstick'], outline_color=(0, 255, 0), thickness=2):
    """
    Overlay segmentation outlines based on facial landmarks for visualization.

    :param image: Original image in BGR format.
    :param landmarks: List of facial landmarks as (x, y) tuples.
    :param makeup_types: List of makeup types to visualize.
    :param outline_color: Tuple representing BGR color for the outlines.
    :param thickness: Thickness of the outline lines.
    :return: Image with segmentation outlines.
    """
    overlay = image.copy()
    
    # Define specific colors for each makeup type for better distinction (optional)
    makeup_colors_visual = {
        'Lipstick': (0, 0, 255),      # Red
        'Blush': (255, 0, 0),     # Blue
        'Eyebrow': (0, 255, 0),        # Green
        'Foundation': (128, 128, 128)  # Gray
    }
    
    for makeup_type in makeup_types:
        # Find the configuration for the makeup type
        config = next((mt for mt in MAKEUP_TYPES_CONFIG if mt.name == makeup_type), None)
        if not config:
            logging.warning(f"No configuration found for makeup type: {makeup_type}. Skipping.")
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        try:
            for region_name, landmark_pairs in config.facemesh_regions.items():
                # Extract unique landmark indices
                indices = list({idx for pair in landmark_pairs for idx in pair})
                region_landmarks = [landmarks[i] for i in indices]

                # Compute convex hull
                hull = cv2.convexHull(np.array(region_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug(f"{makeup_type} - {region_name} mask created.")

            # Clean the mask using morphological operations and Gaussian blur
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            logging.debug(f"{makeup_type} mask cleaned and blurred.")

            # Find contours from the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Determine the color for the outline
            color = makeup_colors_visual.get(makeup_type, outline_color)  # Use specific color if defined

            # Draw contours on the overlay image
            cv2.drawContours(overlay, contours, -1, color, thickness)
            logging.debug(f"Segmentation outline drawn for {makeup_type}.")

        except Exception as e:
            logging.error(f"Error overlaying {makeup_type}: {e}")
            continue  # Proceed with other makeup types

    return overlay.astype(np.uint8)
