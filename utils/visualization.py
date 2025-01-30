# utils/visualization.py

import cv2
import numpy as np
from assets.facemesh_landmarks import (
    FACEMESH_LIP_UPPER,
    FACEMESH_LIP_LOWER,
    FACEMESH_EYESHADOW_LEFT, 
    FACEMESH_EYESHADOW_RIGHT, 
    FACEMESH_EYES,  # Imported FACEMESH_EYES
    FACEMESH_LEFT_EYEBROW, 
    FACEMESH_RIGHT_EYEBROW, 
    FACEMESH_FACE
)
import logging

# Configure logging for this module
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during development
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_unique_indices(frozenset_pairs):
    """
    Extract unique landmark indices from a frozenset of tuples.

    :param frozenset_pairs: frozenset of (int, int) tuples
    :return: list of unique integers
    """
    unique_indices = set()
    for pair in frozenset_pairs:
        unique_indices.update(pair)
    logging.debug(f"Extracted unique indices: {unique_indices}")
    return list(unique_indices)

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
        'Lipstick': (0, 0, 255),       # Red
        'Eyeshadow': (255, 0, 0),      # Blue
        'Eyebrow': (0, 255, 0),        # Green
        'Foundation': (128, 128, 128)  # Gray
    }
    
    for makeup_type in makeup_types:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        try:
            if makeup_type == 'Lipstick':
                # Lower lip
                lower_lip_indices = get_unique_indices(FACEMESH_LIP_LOWER)
                lower_lip_landmarks = [landmarks[i] for i in lower_lip_indices]
                hull_lower = cv2.convexHull(np.array(lower_lip_landmarks))
                cv2.fillConvexPoly(mask, hull_lower, 255)
                logging.debug("Lower lipstick mask created.")

                # Upper lip
                upper_lip_indices = get_unique_indices(FACEMESH_LIP_UPPER)
                upper_lip_landmarks = [landmarks[i] for i in upper_lip_indices]
                hull_upper = cv2.convexHull(np.array(upper_lip_landmarks))
                cv2.fillConvexPoly(mask, hull_upper, 255)
                logging.debug("Upper lipstick mask created.")
            elif makeup_type == 'Eyeshadow':
                # Left eyeshadow
                left_eyeshadow_indices = get_unique_indices(FACEMESH_EYESHADOW_LEFT)
                left_eyeshadow_landmarks = [landmarks[i] for i in left_eyeshadow_indices]
                hull_left = cv2.convexHull(np.array(left_eyeshadow_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left eyeshadow mask created.")

                # Right eyeshadow
                right_eyeshadow_indices = get_unique_indices(FACEMESH_EYESHADOW_RIGHT)
                right_eyeshadow_landmarks = [landmarks[i] for i in right_eyeshadow_indices]
                hull_right = cv2.convexHull(np.array(right_eyeshadow_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right eyeshadow mask created.")

                # # Create eyes mask to exclude inner eyes
                # eyes_indices = get_unique_indices(FACEMESH_EYES)
                # eyes_landmarks = [landmarks[i] for i in eyes_indices]
                # hull_eyes = cv2.convexHull(np.array(eyes_landmarks))
                # eyes_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                # cv2.fillConvexPoly(eyes_mask, hull_eyes, 255)
                # logging.debug("Eyes mask created.")

                # # Subtract eyes mask from eyeshadow mask
                # mask = cv2.bitwise_and(mask, cv2.bitwise_not(eyes_mask))
                # logging.debug("Eyes mask subtracted from eyeshadow mask.")
            elif makeup_type == 'Eyebrow':
                # Left Eyebrow
                left_eyebrow_indices = get_unique_indices(FACEMESH_LEFT_EYEBROW)
                left_eyebrow_landmarks = [landmarks[i] for i in left_eyebrow_indices]
                hull_left = cv2.convexHull(np.array(left_eyebrow_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left Eyebrow mask created.")

                # Right Eyebrow
                right_eyebrow_indices = get_unique_indices(FACEMESH_RIGHT_EYEBROW)
                right_eyebrow_landmarks = [landmarks[i] for i in right_eyebrow_indices]
                hull_right = cv2.convexHull(np.array(right_eyebrow_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right Eyebrow mask created.")
            elif makeup_type == 'Foundation':
                # Entire face
                face_indices = get_unique_indices(FACEMESH_FACE)
                face_landmarks = [landmarks[i] for i in face_indices]
                hull_face = cv2.convexHull(np.array(face_landmarks))
                cv2.fillConvexPoly(mask, hull_face, 255)
                logging.debug("Foundation mask created.")
            else:
                # Unknown makeup type
                logging.warning(f"Unknown makeup type: {makeup_type}. Skipping.")
                continue

            # if makeup_type == 'Eyeshadow':
            #     # Clean the mask
            #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            #     mask = cv2.GaussianBlur(mask, (7, 7), 0)
            #     logging.debug(f"Mask for {makeup_type} cleaned and blurred.")

            #     # Find contours from the mask
            #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
            #     # Determine the color for the outline
            #     color = makeup_colors_visual.get(makeup_type, outline_color)  # Use specific color if defined

            #     # Draw contours on the overlay image
            #     cv2.drawContours(overlay, contours, -1, color, thickness)
            #     logging.debug(f"Segmentation outline drawn for {makeup_type}.")
            # else:
            #     # Handle other makeup types (Lipstick, Eyebrow, Foundation)
            #     # Similar contour drawing can be implemented here if needed
            #     # Example for Lipstick:
            #     if makeup_type == 'Lipstick':
            #         # Clean the mask
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
