# src/makeup_transfer.py

import cv2
import numpy as np
import logging
from assets.facemesh_landmarks import FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class MakeupTransfer:
    def __init__(self):
        logging.info("MakeupTransfer initialized.")
        self.makeup_color = None  # Initialize makeup_color to store the extracted color

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

    def extract_makeup_color(self, reference_image, landmarks, makeup_type='Lipstick'):
        """
        Extract average color from the target regions in the reference image based on landmarks.

        :param reference_image: Original reference image in BGR
        :param landmarks: List of facial landmarks as (x, y) tuples
        :param makeup_type: Type of makeup to extract ('Lipstick', 'Eyeshadow', etc.)
        :return: BGR color tuple
        """
        logging.info(f"Extracting makeup color for type: {makeup_type}")
        mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)

        try:
            if makeup_type == 'Lipstick':
                indices = [idx for pair in FACEMESH_LIPS for idx in pair]
                lip_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(lip_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Lipstick mask created.")
            elif makeup_type == 'Eyeshadow':
                # Left eye
                indices = [idx for pair in FACEMESH_LEFT_EYE for idx in pair]
                left_eye_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left eyeshadow mask created.")

                # Right eye
                indices = [idx for pair in FACEMESH_RIGHT_EYE for idx in pair]
                right_eye_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right eyeshadow mask created.")
            elif makeup_type == 'Blush':
                # Left blush
                indices = [idx for pair in FACEMESH_LEFT_EYEBROW for idx in pair]
                left_cheek_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left blush mask created.")

                # Right blush
                indices = [idx for pair in FACEMESH_RIGHT_EYEBROW for idx in pair]
                right_cheek_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right blush mask created.")
            elif makeup_type == 'Foundation':
                # Entire face
                indices = [idx for pair in FACEMESH_FACE_OVAL for idx in pair]
                face_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(face_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Foundation mask created.")
            else:
                # Default to entire face
                hull = cv2.convexHull(np.array(landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Default face mask created.")

            # Clean the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            logging.debug("Mask cleaned using morphological operations.")

            # Compute the mean color within the mask
            mean_color = cv2.mean(reference_image, mask=mask)[:3]
            self.makeup_color = mean_color  # Store the extracted color
            logging.info(f"Extracted Makeup Color (BGR): {mean_color}")
            return mean_color  # BGR
        except Exception as e:
            logging.error(f"Error extracting makeup color: {e}")
            raise e

    def apply_makeup(self, target_image, landmarks, makeup_type='Lipstick', makeup_color=None, alpha=0.6):
        """
        Apply the makeup color to the target regions based on landmarks.

        :param target_image: Original target image in BGR
        :param landmarks: List of facial landmarks
        :param makeup_type: Type of makeup to apply ('Lipstick', 'Eyeshadow', etc.)
        :param makeup_color: BGR color tuple. If None, uses the extracted color.
        :param alpha: Intensity of makeup (0.0 to 1.0)
        :return: Image with applied makeup
        """
        logging.info(f"Applying makeup: {makeup_type} with color {makeup_color} and alpha {alpha}")

        # Use the extracted color if makeup_color is not provided
        if makeup_color is None:
            if self.makeup_color is not None:
                makeup_color = self.makeup_color
                logging.debug("Using stored extracted makeup color.")
            else:
                logging.warning("Makeup color not provided and no color extracted. Using default color (0, 0, 255).")
                makeup_color = (0, 0, 255)  # Default to red if no color is extracted

        mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

        try:
            if makeup_type == 'Lipstick':
                indices = [idx for pair in FACEMESH_LIPS for idx in pair]
                lip_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(lip_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Lipstick mask created.")
            elif makeup_type == 'Eyeshadow':
                # Left eye
                indices = [idx for pair in FACEMESH_LEFT_EYE for idx in pair]
                left_eye_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left eyeshadow mask created.")

                # Right eye
                indices = [idx for pair in FACEMESH_RIGHT_EYE for idx in pair]
                right_eye_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_eye_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right eyeshadow mask created.")
            elif makeup_type == 'Blush':
                # Left blush
                indices = [idx for pair in FACEMESH_LEFT_EYEBROW for idx in pair]
                left_cheek_landmarks = [landmarks[i] for i in indices]
                hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_left, 255)
                logging.debug("Left blush mask created.")

                # Right blush
                indices = [idx for pair in FACEMESH_RIGHT_EYEBROW for idx in pair]
                right_cheek_landmarks = [landmarks[i] for i in indices]
                hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
                cv2.fillConvexPoly(mask, hull_right, 255)
                logging.debug("Right blush mask created.")
            elif makeup_type == 'Foundation':
                # Entire face
                indices = [idx for pair in FACEMESH_FACE_OVAL for idx in pair]
                face_landmarks = [landmarks[i] for i in indices]
                hull = cv2.convexHull(np.array(face_landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Foundation mask created.")
            else:
                # Default to entire face
                hull = cv2.convexHull(np.array(landmarks))
                cv2.fillConvexPoly(mask, hull, 255)
                logging.debug("Default face mask created.")

            # Clean the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            logging.debug("Mask cleaned and blurred.")

            # Create a color overlay
            color_overlay = np.full(target_image.shape, makeup_color, dtype=np.uint8)
            color_overlay = cv2.GaussianBlur(color_overlay, (15, 15), 0)
            logging.debug("Color overlay created and blurred.")

            # Blend the color overlay with the target image
            blended = cv2.addWeighted(color_overlay, alpha, target_image, 1 - alpha, 0)
            logging.debug("Color overlay blended with target image.")

            # Create a mask for the makeup region
            makeup_mask = mask.astype(bool)
            logging.debug(f"Makeup mask created with shape {makeup_mask.shape}.")

            # Apply the blended makeup to the target image
            makeup_applied = target_image.copy()
            makeup_applied[makeup_mask] = blended[makeup_mask]
            logging.debug("Makeup applied to the target image.")

            # Optional: Apply additional smoothing to the makeup region
            makeup_applied = cv2.GaussianBlur(makeup_applied, (5, 5), 0)
            logging.debug("Applied additional Gaussian blur to the makeup region.")

            return makeup_applied
        except Exception as e:
            logging.error(f"Error applying makeup: {e}")
            return target_image 