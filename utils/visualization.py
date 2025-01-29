# utils/visualization.py

import cv2
import numpy as np
from assets.facemesh_landmarks import (
    FACEMESH_LIP_UPPER,
    FACEMESH_LIP_LOWER,
    FACEMESH_EYESHADOW_LEFT, 
    FACEMESH_EYESHADOW_RIGHT, 
    FACEMESH_LEFT_EYEBROW, 
    FACEMESH_RIGHT_EYEBROW, 
    FACEMESH_FACE
)
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
        'Eyeshadow': (255, 0, 0),     # Blue
        'Eyebrow': (0, 255, 0),         # Green
        'Foundation': (128, 128, 128) # Gray
    }
    
    for makeup_type in makeup_types:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        try:
            if makeup_type == 'Lipstick':
                #Lower lip
                indices = [idx for pair in FACEMESH_LIP_LOWER for idx in pair]
                lower_lip_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(lower_lip_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Lower lipstick mask created.")

                #Upper lip
                indices = [idx for pair in FACEMESH_LIP_LOWER for idx in pair]
                upper_lip_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(upper_lip_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Upper lipstick mask created.")
                
            elif makeup_type == 'Eyeshadow':
                # Left eye
                indices = [idx for pair in FACEMESH_EYESHADOW_LEFT for idx in pair]
                left_eye_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left eyeshadow mask created.")

                # Right eye
                indices = [idx for pair in FACEMESH_EYESHADOW_RIGHT for idx in pair]
                right_eye_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right eyeshadow mask created.")
            elif makeup_type == 'Eyebrow':
                # Left Eyebrow
                indices = [idx for pair in FACEMESH_LEFT_EYEBROW for idx in pair]
                left_cheek_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left Eyebrow mask created.")

                # Right Eyebrow
                indices = [idx for pair in FACEMESH_RIGHT_EYEBROW for idx in pair]
                right_cheek_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right Eyebrow mask created.")
            elif makeup_type == 'Foundation':
                # Entire face
                indices = [idx for pair in FACEMESH_FACE_OVAL for idx in pair]
                face_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(face_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Foundation mask created.")
            else:
                # Unknown makeup type
                logging.warning(f"Unknown makeup type: {makeup_type}. Skipping.")
                continue

            # Clean the mask using morphological operations and Gaussian blur
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            logging.debug(f"Mask for {makeup_type} cleaned and blurred.")

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
