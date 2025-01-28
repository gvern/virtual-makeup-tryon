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
        pass

    def convert_rgb_to_bgr(self, rgb_color):
        """
        Converts an RGB color tuple to BGR.

        :param rgb_color: Tuple of (R, G, B)
        :return: Tuple of (B, G, R)
        """
        if not isinstance(rgb_color, tuple) or len(rgb_color) != 3:
            logging.error("RGB color must be a tuple of 3 elements.")
            raise ValueError("RGB color must be a tuple of 3 elements.")
        return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

    def extract_makeup_color(self, reference_image, landmarks, makeup_type='Lipstick'):
        """
        Extract average color from the target regions in the reference image based on landmarks.

        :param reference_image: Original reference image in BGR
        :param landmarks: List of facial landmarks
        :param makeup_type: Type of makeup to extract ('Lipstick', 'Eyeshadow', etc.)
        :return: BGR color tuple
        """
        mask = np.zeros(reference_image.shape[:2], dtype=np.uint8)

        if makeup_type == 'Lipstick':
            indices = [idx for pair in FACEMESH_LIPS for idx in pair]
            lip_landmarks = [landmarks[i] for i in indices]
            hull = cv2.convexHull(np.array(lip_landmarks))
            cv2.fillConvexPoly(mask, hull, 255)
        elif makeup_type == 'Eyeshadow':
            # Left eye
            indices = [idx for pair in FACEMESH_LEFT_EYE for idx in pair]
            left_eye_landmarks = [landmarks[i] for i in indices]
            hull_left = cv2.convexHull(np.array(left_eye_landmarks))
            cv2.fillConvexPoly(mask, hull_left, 255)
            # Right eye
            indices = [idx for pair in FACEMESH_RIGHT_EYE for idx in pair]
            right_eye_landmarks = [landmarks[i] for i in indices]
            hull_right = cv2.convexHull(np.array(right_eye_landmarks))
            cv2.fillConvexPoly(mask, hull_right, 255)
        elif makeup_type == 'Blush':
            # Left blush
            indices = [idx for pair in FACEMESH_LEFT_EYEBROW for idx in pair]
            left_cheek_landmarks = [landmarks[i] for i in indices]
            hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
            cv2.fillConvexPoly(mask, hull_left, 255)
            # Right blush
            indices = [idx for pair in FACEMESH_RIGHT_EYEBROW for idx in pair]
            right_cheek_landmarks = [landmarks[i] for i in indices]
            hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
            cv2.fillConvexPoly(mask, hull_right, 255)
        elif makeup_type == 'Foundation':
            # Entire face
            indices = [idx for pair in FACEMESH_FACE_OVAL for idx in pair]
            face_landmarks = [landmarks[i] for i in indices]
            hull = cv2.convexHull(np.array(face_landmarks))
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            # Default to entire face
            hull = cv2.convexHull(np.array(landmarks))
            cv2.fillConvexPoly(mask, hull, 255)

        # Clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Compute the mean color within the mask
        mean_color = cv2.mean(reference_image, mask=mask)[:3]
        logging.info(f"Extracted Makeup Color (BGR): {mean_color}")
        return mean_color  # BGR

    def apply_makeup(self, target_image, landmarks, makeup_type='Lipstick', makeup_color=(0, 0, 255), alpha=0.6):
        """
        Apply the makeup color to the target regions based on landmarks.

        :param target_image: Original target image in BGR
        :param landmarks: List of facial landmarks
        :param makeup_type: Type of makeup to apply ('Lipstick', 'Eyeshadow', etc.)
        :param makeup_color: BGR color tuple
        :param alpha: Intensity of makeup
        :return: Image with applied makeup
        """
        mask = np.zeros(target_image.shape[:2], dtype=np.uint8)

        if makeup_type == 'Lipstick':
            indices = [idx for pair in FACEMESH_LIPS for idx in pair]
            lip_landmarks = [landmarks[i] for i in indices]
            hull = cv2.convexHull(np.array(lip_landmarks))
            cv2.fillConvexPoly(mask, hull, 255)
        elif makeup_type == 'Eyeshadow':
            # Left eye
            indices = [idx for pair in FACEMESH_LEFT_EYE for idx in pair]
            left_eye_landmarks = [landmarks[i] for i in indices]
            hull_left = cv2.convexHull(np.array(left_eye_landmarks))
            cv2.fillConvexPoly(mask, hull_left, 255)
            # Right eye
            indices = [idx for pair in FACEMESH_RIGHT_EYE for idx in pair]
            right_eye_landmarks = [landmarks[i] for i in indices]
            hull_right = cv2.convexHull(np.array(right_eye_landmarks))
            cv2.fillConvexPoly(mask, hull_right, 255)
        elif makeup_type == 'Blush':
            # Left blush
            indices = [idx for pair in FACEMESH_LEFT_EYEBROW for idx in pair]
            left_cheek_landmarks = [landmarks[i] for i in indices]
            hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
            cv2.fillConvexPoly(mask, hull_left, 255)
            # Right blush
            indices = [idx for pair in FACEMESH_RIGHT_EYEBROW for idx in pair]
            right_cheek_landmarks = [landmarks[i] for i in indices]
            hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
            cv2.fillConvexPoly(mask, hull_right, 255)
        elif makeup_type == 'Foundation':
            # Entire face
            indices = [idx for pair in FACEMESH_FACE_OVAL for idx in pair]
            face_landmarks = [landmarks[i] for i in indices]
            hull = cv2.convexHull(np.array(face_landmarks))
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            # Default to entire face
            hull = cv2.convexHull(np.array(landmarks))
            cv2.fillConvexPoly(mask, hull, 255)

        # Clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Create makeup overlay
        makeup = np.zeros_like(target_image, dtype=np.uint8)
        makeup[:] = makeup_color
        makeup = cv2.GaussianBlur(makeup, (15, 15), 0)

        # Blend makeup with target image
        blended = cv2.addWeighted(makeup, alpha, target_image, 1 - alpha, 0)

        # Preserve skin texture by blending only on masked regions
        makeup_applied = np.where(mask[:, :, np.newaxis] == 255, blended, target_image)

        # Apply edge smoothing
        makeup_applied = cv2.GaussianBlur(makeup_applied, (5, 5), 0)

        return makeup_applied
