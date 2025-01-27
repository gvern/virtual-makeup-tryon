# utils/visualization.py

import cv2
import numpy as np

def overlay_segmentation(image, landmarks, makeup_type='Lipstick'):
    """
    Overlay segmentation based on facial landmarks for visualization.

    :param image: Original image in BGR format.
    :param landmarks: List of facial landmarks.
    :param makeup_type: Type of makeup to visualize.
    :return: Image with segmentation overlay.
    """
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if makeup_type == 'Lipstick':
        lip_indices = list(range(61, 80))  # Adjust based on actual landmark indices
        lip_landmarks = [landmarks[i] for i in lip_indices]
        hull = cv2.convexHull(np.array(lip_landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
    elif makeup_type == 'Eyeshadow':
        left_eye_indices = list(range(33, 133))  # Adjust indices as per MediaPipe FaceMesh
        left_eye_landmarks = [landmarks[i] for i in left_eye_indices]
        hull_left = cv2.convexHull(np.array(left_eye_landmarks))
        cv2.fillConvexPoly(mask, hull_left, 255)
        
        right_eye_indices = list(range(362, 263))  # Adjust indices as per MediaPipe FaceMesh
        right_eye_landmarks = [landmarks[i] for i in right_eye_indices]
        hull_right = cv2.convexHull(np.array(right_eye_landmarks))
        cv2.fillConvexPoly(mask, hull_right, 255)
    elif makeup_type == 'Blush':
        left_cheek_indices = list(range(234, 454))  # Adjust as needed
        right_cheek_indices = list(range(454, 674))  # Adjust as needed
        left_cheek_landmarks = [landmarks[i] for i in left_cheek_indices]
        right_cheek_landmarks = [landmarks[i] for i in right_cheek_indices]
        hull_left = cv2.convexHull(np.array(left_cheek_landmarks))
        hull_right = cv2.convexHull(np.array(right_cheek_landmarks))
        cv2.fillConvexPoly(mask, hull_left, 255)
        cv2.fillConvexPoly(mask, hull_right, 255)
    elif makeup_type == 'Foundation':
        hull = cv2.convexHull(np.array(landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        hull = cv2.convexHull(np.array(landmarks))
        cv2.fillConvexPoly(mask, hull, 255)
    
    # Create colored overlay
    colored_overlay = np.zeros_like(image, dtype=np.uint8)
    if makeup_type == 'Lipstick':
        colored_overlay[:] = (0, 0, 255)  # Red for lipstick
    elif makeup_type == 'Eyeshadow':
        colored_overlay[:] = (255, 0, 0)  # Blue for eyeshadow
    elif makeup_type == 'Blush':
        colored_overlay[:] = (0, 255, 0)  # Green for blush
    elif makeup_type == 'Foundation':
        colored_overlay[:] = (128, 128, 128)  # Gray for foundation
    else:
        colored_overlay[:] = (255, 255, 255)  # White by default
    
    # Apply the mask to the colored overlay
    overlay = cv2.addWeighted(image, 0.7, colored_overlay, 0.3, 0)
    overlay = np.where(mask[:, :, np.newaxis] == 255, overlay, image)
    
    return overlay.astype(np.uint8)
