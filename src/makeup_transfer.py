# src/makeup_transfer.py

import cv2
import numpy as np
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during development
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class MakeupTransfer:
    def __init__(self):
        logging.info("MakeupTransfer initialized.")
        self.makeup_colors = {}  # Dictionary to store colors per makeup type

    @staticmethod
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

    def convert_rgb_to_bgr(self, rgb_color):
        """
        Converts an RGB color tuple to BGR.

        :param rgb_color: Tuple of (R, G, B)
        :return: Tuple of (B, G, R)
        """
        if not isinstance(rgb_color, tuple) or len(rgb_color) != 3:
            logging.error("RGB color must be a tuple of 3 elements.")
            raise ValueError("RGB color must be a tuple of 3 elements.")
        bgr = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
        logging.debug(f"Converted RGB {rgb_color} to BGR {bgr}.")
        return bgr

    def extract_makeup_color(self, reference_image, landmarks, makeup_types=['Lipstick']):
        """
        Extract average color from the target regions in the reference image based on landmarks.

        :param reference_image: Original reference image in BGR
        :param landmarks: List of facial landmarks as (x, y) tuples
        :param makeup_types: List of makeup types to extract ('Lipstick', 'Eyeshadow', etc.)
        :return: Dictionary of makeup types to BGR color tuples
        """
        logging.info(f"Extracting makeup colors for types: {makeup_types}")
        makeup_colors = {}

        for makeup_type in makeup_types:
            mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)

            try:
                if makeup_type == 'Lipstick':
                    # Lower lip
                    lower_lip_indices = self.get_unique_indices(FACEMESH_LIP_LOWER)
                    lower_lip_landmarks = [landmarks[i] for i in lower_lip_indices]
                    hull_lower = cv2.convexHull(np.array(lower_lip_landmarks))
                    cv2.fillConvexPoly(mask, hull_lower, 255)
                    logging.debug("Lower lipstick mask created.")

                    # Upper lip
                    upper_lip_indices = self.get_unique_indices(FACEMESH_LIP_UPPER)
                    upper_lip_landmarks = [landmarks[i] for i in upper_lip_indices]
                    hull_upper = cv2.convexHull(np.array(upper_lip_landmarks))
                    cv2.fillConvexPoly(mask, hull_upper, 255)
                    logging.debug("Upper lipstick mask created.")
                elif makeup_type == 'Eyeshadow':
                    # Left eyeshadow
                    left_eyeshadow_indices = self.get_unique_indices(FACEMESH_EYESHADOW_LEFT)
                    left_eyeshadow_landmarks = [landmarks[i] for i in left_eyeshadow_indices]
                    hull_left = cv2.convexHull(np.array(left_eyeshadow_landmarks))
                    cv2.fillConvexPoly(mask, hull_left, 255)
                    logging.debug("Left eyeshadow mask created.")

                    # Right eyeshadow
                    right_eyeshadow_indices = self.get_unique_indices(FACEMESH_EYESHADOW_RIGHT)
                    right_eyeshadow_landmarks = [landmarks[i] for i in right_eyeshadow_indices]
                    hull_right = cv2.convexHull(np.array(right_eyeshadow_landmarks))
                    cv2.fillConvexPoly(mask, hull_right, 255)
                    logging.debug("Right eyeshadow mask created.")

                    # Create eyes mask to exclude inner eyes
                    eyes_indices = self.get_unique_indices(FACEMESH_EYES)
                    eyes_landmarks = [landmarks[i] for i in eyes_indices]
                    hull_eyes = cv2.convexHull(np.array(eyes_landmarks))
                    eyes_mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)
                    cv2.fillConvexPoly(eyes_mask, hull_eyes, 255)
                    logging.debug("Eyes mask created.")

                    # Subtract eyes mask from eyeshadow mask
                    mask = cv2.bitwise_and(mask, cv2.bitwise_not(eyes_mask))
                    logging.debug("Eyes mask subtracted from eyeshadow mask.")
                elif makeup_type == 'Eyebrow':
                    # Left Eyebrow
                    left_eyebrow_indices = self.get_unique_indices(FACEMESH_LEFT_EYEBROW)
                    left_eyebrow_landmarks = [landmarks[i] for i in left_eyebrow_indices]
                    hull_left = cv2.convexHull(np.array(left_eyebrow_landmarks))
                    cv2.fillConvexPoly(mask, hull_left, 255)
                    logging.debug("Left Eyebrow mask created.")

                    # Right Eyebrow
                    right_eyebrow_indices = self.get_unique_indices(FACEMESH_RIGHT_EYEBROW)
                    right_eyebrow_landmarks = [landmarks[i] for i in right_eyebrow_indices]
                    hull_right = cv2.convexHull(np.array(right_eyebrow_landmarks))
                    cv2.fillConvexPoly(mask, hull_right, 255)
                    logging.debug("Right Eyebrow mask created.")
                elif makeup_type == 'Foundation':
                    # Entire face
                    face_indices = self.get_unique_indices(FACEMESH_FACE)
                    face_landmarks = [landmarks[i] for i in face_indices]
                    hull_face = cv2.convexHull(np.array(face_landmarks))
                    cv2.fillConvexPoly(mask, hull_face, 255)
                    logging.debug("Foundation mask created.")
                else:
                    # Unknown makeup type
                    logging.warning(f"Unknown makeup type: {makeup_type}. Skipping.")
                    continue

                # Clean the mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                logging.debug(f"Mask for {makeup_type} cleaned and blurred.")

                # Compute the mean color within the mask
                mean_color = cv2.mean(reference_image, mask=mask)[:3]
                makeup_colors[makeup_type] = mean_color  # Store the extracted color
                logging.info(f"Extracted Makeup Color for {makeup_type} (BGR): {mean_color}")

            except Exception as e:
                logging.error(f"Error extracting {makeup_type} color: {e}")
                continue  # Proceed with other makeup types

        self.makeup_colors = makeup_colors  # Update the class attribute
        return makeup_colors

    def apply_makeup(self, target_image, landmarks, makeup_params):
        """
        Apply multiple makeup types to the target image based on landmarks and parameters.

        :param target_image: Original target image in BGR
        :param landmarks: List of facial landmarks as (x, y) tuples
        :param makeup_params: Dictionary with makeup types as keys and parameters as values
                              Each value should be a dictionary with 'color' (BGR tuple) and 'intensity' (float)
        :return: Image with applied makeup
        """
        logging.info(f"Applying makeup types: {list(makeup_params.keys())}")
        makeup_applied = target_image.copy()

        for makeup_type, params in makeup_params.items():
            color = params.get('color', (0, 0, 255))  # Default to red if not specified
            intensity = params.get('intensity', 0.6)

            mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

            try:
                if makeup_type == 'Lipstick':
                    # Lower lip
                    lower_lip_indices = self.get_unique_indices(FACEMESH_LIP_LOWER)
                    lower_lip_landmarks = [landmarks[i] for i in lower_lip_indices]
                    hull_lower = cv2.convexHull(np.array(lower_lip_landmarks))
                    cv2.fillConvexPoly(mask, hull_lower, 255)
                    logging.debug("Lower lipstick mask created.")

                    # Upper lip
                    upper_lip_indices = self.get_unique_indices(FACEMESH_LIP_UPPER)
                    upper_lip_landmarks = [landmarks[i] for i in upper_lip_indices]
                    hull_upper = cv2.convexHull(np.array(upper_lip_landmarks))
                    cv2.fillConvexPoly(mask, hull_upper, 255)
                    logging.debug("Upper lipstick mask created.")
                elif makeup_type == 'Eyeshadow':
                    # Left eyeshadow
                    left_eyeshadow_indices = self.get_unique_indices(FACEMESH_EYESHADOW_LEFT)
                    left_eyeshadow_landmarks = [landmarks[i] for i in left_eyeshadow_indices]
                    hull_left = cv2.convexHull(np.array(left_eyeshadow_landmarks))
                    cv2.fillConvexPoly(mask, hull_left, 255)
                    logging.debug("Left eyeshadow mask created.")

                    # Right eyeshadow
                    right_eyeshadow_indices = self.get_unique_indices(FACEMESH_EYESHADOW_RIGHT)
                    right_eyeshadow_landmarks = [landmarks[i] for i in right_eyeshadow_indices]
                    hull_right = cv2.convexHull(np.array(right_eyeshadow_landmarks))
                    cv2.fillConvexPoly(mask, hull_right, 255)
                    logging.debug("Right eyeshadow mask created.")

                    # # Create eyes mask to exclude inner eyes
                    # eyes_indices = self.get_unique_indices(FACEMESH_EYES)
                    # eyes_landmarks = [landmarks[i] for i in eyes_indices]
                    # hull_eyes = cv2.convexHull(np.array(eyes_landmarks))
                    # eyes_mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
                    # cv2.fillConvexPoly(eyes_mask, hull_eyes, 255)
                    # logging.debug("Eyes mask created.")

                    # # Subtract eyes mask from eyeshadow mask
                    # mask = cv2.bitwise_and(mask, cv2.bitwise_not(eyes_mask))
                    # logging.debug("Eyes mask subtracted from eyeshadow mask.")
                elif makeup_type == 'Eyebrow':
                    # Left Eyebrow
                    left_eyebrow_indices = self.get_unique_indices(FACEMESH_LEFT_EYEBROW)
                    left_eyebrow_landmarks = [landmarks[i] for i in left_eyebrow_indices]
                    hull_left = cv2.convexHull(np.array(left_eyebrow_landmarks))
                    cv2.fillConvexPoly(mask, hull_left, 255)
                    logging.debug("Left Eyebrow mask created.")

                    # Right Eyebrow
                    right_eyebrow_indices = self.get_unique_indices(FACEMESH_RIGHT_EYEBROW)
                    right_eyebrow_landmarks = [landmarks[i] for i in right_eyebrow_indices]
                    hull_right = cv2.convexHull(np.array(right_eyebrow_landmarks))
                    cv2.fillConvexPoly(mask, hull_right, 255)
                    logging.debug("Right Eyebrow mask created.")
                elif makeup_type == 'Foundation':
                    # Entire face
                    face_indices = self.get_unique_indices(FACEMESH_FACE)
                    face_landmarks = [landmarks[i] for i in face_indices]
                    hull_face = cv2.convexHull(np.array(face_landmarks))
                    cv2.fillConvexPoly(mask, hull_face, 255)
                    logging.debug("Foundation mask created.")
                else:
                    # Unknown makeup type
                    logging.warning(f"Unknown makeup type: {makeup_type}. Skipping.")
                    continue

                # Clean the mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                logging.debug(f"Mask for {makeup_type} cleaned and blurred.")

                if makeup_type == 'Eyeshadow':
                    # Compute mean color after excluding inner eyes
                    mean_color = cv2.mean(target_image, mask=mask)[:3]
                else:
                    # Compute mean color normally
                    mean_color = cv2.mean(target_image, mask=mask)[:3]

                # Create a color overlay for the makeup
                color_overlay = np.full(target_image.shape, color, dtype=np.uint8)
                color_overlay = cv2.GaussianBlur(color_overlay, (15, 15), 0)
                logging.debug(f"Color overlay for {makeup_type} created and blurred.")

                # Blend the color overlay with the target image
                blended = cv2.addWeighted(color_overlay, intensity, target_image, 1 - intensity, 0)
                logging.debug(f"Color overlay for {makeup_type} blended with target image.")

                # Create a boolean mask
                makeup_mask = mask.astype(bool)
                logging.debug(f"Makeup mask for {makeup_type} created with shape {makeup_mask.shape}.")

                # Apply the blended makeup to the target image
                makeup_applied[makeup_mask] = blended[makeup_mask]
                logging.debug(f"Makeup applied for {makeup_type}.")
            except Exception as e:
                logging.error(f"Error applying {makeup_type}: {e}")
                continue  # Proceed with other makeup types
        # Optional: Apply additional smoothing to the entire makeup-applied image
        makeup_applied = cv2.GaussianBlur(makeup_applied, (5, 5), 0)
        logging.debug("Applied additional Gaussian blur to the makeup-applied image.")

        return makeup_applied
