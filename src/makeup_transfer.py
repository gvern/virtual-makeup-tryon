# src/makeup_transfer.py

import cv2
import numpy as np

class MakeupTransfer:
    def __init__(self):
        pass
    
    def extract_makeup_color(self, reference_image, parsing_map, target_regions=[11, 12]):
        """
        Extract average color from the target regions in the reference image.

        :param reference_image: Original reference image in BGR
        :param parsing_map: Parsed segmentation map
        :param target_regions: List of region labels for lips
        :return: BGR color tuple
        """
        mask = np.isin(parsing_map, target_regions).astype(np.uint8) * 255
        # Apply morphological operations to clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Compute the mean color within the mask
        mean_color = cv2.mean(reference_image, mask=mask)[:3]
        print(f"Extracted Makeup Color (BGR): {mean_color}")
        return mean_color  # BGR
    
    def apply_makeup(self, target_image, parsing_map, makeup_color, target_regions=[11, 12], alpha=0.6):
        """
        Apply the makeup color to the target regions.

        :param target_image: Original target image in BGR
        :param parsing_map: Parsed segmentation map
        :param makeup_color: BGR color tuple
        :param target_regions: List of region labels for lips
        :param alpha: Intensity of makeup
        :return: Image with applied makeup
        """
        # Create a mask for the target regions
        mask = np.isin(parsing_map, target_regions).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Create an overlay with the makeup color
        overlay = np.full(target_image.shape, makeup_color, dtype=np.uint8)
        
        # Apply Gaussian blur to the overlay to soften edges
        overlay = cv2.GaussianBlur(overlay, (7, 7), 0)
        
        # Blend the overlay with the target image using the mask
        blended = cv2.addWeighted(overlay, alpha, target_image, 1 - alpha, 0)
        
        # Apply the mask to restrict makeup to the target regions
        makeup_applied = np.where(mask[:, :, np.newaxis] == 255, blended, target_image)
        
        # Log the application
        print(f"Applying makeup with color {makeup_color} on regions {target_regions}")
        
        return makeup_applied
