import cv2
import numpy as np

class MakeupTransfer:
    def __init__(self):
        pass
    
    def extract_makeup_color(self, reference_image, parsing_map, target_region=12):
        """
        Extract average color from the target region in the reference image.

        :param reference_image: Original reference image in BGR
        :param parsing_map: Parsed segmentation map
        :param target_region: Region label for lips (update based on the new model)
        :return: BGR color tuple
        """
        mask = (parsing_map == target_region).astype(np.uint8) * 255
        # Optional: Apply morphological operations to clean the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Compute the mean color within the mask
        mean_color = cv2.mean(reference_image, mask=mask)[:3]
        return mean_color  # BGR
    
    def apply_makeup(self, target_image, parsing_map, makeup_color, target_region=12, alpha=0.5):
        """
        Apply the makeup color to the target region.

        :param target_image: Original target image in BGR
        :param parsing_map: Parsed segmentation map
        :param makeup_color: BGR color tuple
        :param target_region: Region label for lips (update based on the new model)
        :param alpha: Intensity of makeup
        :return: Image with applied makeup
        """
        overlay = target_image.copy()
        mask = (parsing_map == target_region).astype(np.uint8) * 255
        
        # Optional: Apply morphological operations to smooth the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # Create an overlay with the makeup color
        overlay[:] = makeup_color
        
        # Blend the overlay with the target image
        result = cv2.addWeighted(overlay, alpha, target_image, 1 - alpha, 0)
        
        # Apply the mask to restrict makeup to the target region
        target = cv2.bitwise_and(result, result, mask=mask)
        background = cv2.bitwise_and(target_image, target_image, mask=cv2.bitwise_not(mask))
        final = cv2.add(target, background)
        return final
