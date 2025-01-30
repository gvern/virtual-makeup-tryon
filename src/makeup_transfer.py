# src/makeup_transfer.py

import cv2
import numpy as np
import logging
from src.makeup_config import MAKEUP_TYPES_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MakeupTransfer:
    def __init__(self):
        logging.info("MakeupTransfer initialized.")
        self.makeup_colors = {}  # Dictionary to store colors per makeup type

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
        :param makeup_types: List of makeup types to extract.
        :return: Dictionary of makeup types to BGR color tuples
        """
        logging.info(f"Extracting makeup colors for types: {makeup_types}")
        makeup_colors = {}

        for makeup_type in makeup_types:
            # Find the configuration for the makeup type
            config = next((mt for mt in MAKEUP_TYPES_CONFIG if mt.name == makeup_type), None)
            if not config:
                logging.warning(f"No configuration found for makeup type: {makeup_type}. Skipping.")
                continue

            mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)

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
            # Find the configuration for the makeup type
            config = next((mt for mt in MAKEUP_TYPES_CONFIG if mt.name == makeup_type), None)
            if not config:
                logging.warning(f"No configuration found for makeup type: {makeup_type}. Skipping.")
                continue

            color = params.get('color', config.default_color)  # Use provided color or default
            intensity = params.get('intensity', config.default_intensity)  # Use provided intensity or default

            mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

            try:
                for region_name, landmark_pairs in config.facemesh_regions.items():
                    # Extract unique landmark indices
                    indices = list({idx for pair in landmark_pairs for idx in pair})
                    region_landmarks = [landmarks[i] for i in indices]

                    # Compute convex hull
                    hull = cv2.convexHull(np.array(region_landmarks))
                    cv2.fillConvexPoly(mask, hull, 255)
                    logging.debug(f"{makeup_type} - {region_name} mask created.")

                # Clean the mask
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                logging.debug(f"{makeup_type} mask cleaned and blurred.")

                # Create a color overlay for the makeup
                color_overlay = np.full(target_image.shape, color, dtype=np.uint8)
                color_overlay = cv2.GaussianBlur(color_overlay, (15, 15), 0)
                logging.debug(f"{makeup_type} color overlay created and blurred.")

                # Blend the color overlay with the target image
                blended = cv2.addWeighted(color_overlay, intensity, target_image, 1 - intensity, 0)
                logging.debug(f"{makeup_type} color overlay blended with target image.")

                # Create a boolean mask
                makeup_mask = mask.astype(bool)
                logging.debug(f"{makeup_type} makeup mask created with shape {makeup_mask.shape}.")

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
