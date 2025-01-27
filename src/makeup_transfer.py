import cv2
import numpy as np

class MakeupTransfer:
    def __init__(self):
        pass
    
    def extract_makeup_color(self, reference_image, parsing_map, target_region=13):
        """
        Extract average color from the target region in the reference image.
        :param reference_image: Original reference image in BGR
        :param parsing_map: Parsed segmentation map
        :param target_region: Region label for lips (commonly 13 in BiSeNet)
        :return: BGR color tuple
        """
        mask = (parsing_map == target_region).astype(np.uint8) * 255
        masked = cv2.bitwise_and(reference_image, reference_image, mask=mask)
        # Compute the mean color
        mean_color = cv2.mean(reference_image, mask=mask)[:3]
        return mean_color  # BGR
    
    def apply_makeup(self, target_image, parsing_map, makeup_color, target_region=13, alpha=0.5):
        """
        Apply the makeup color to the target region.
        :param target_image: Original target image in BGR
        :param parsing_map: Parsed segmentation map
        :param makeup_color: BGR color tuple
        :param target_region: Region label for lips
        :param alpha: Intensity of makeup
        :return: Image with applied makeup
        """
        overlay = target_image.copy()
        mask = (parsing_map == target_region).astype(np.uint8) * 255
        overlay[:] = makeup_color
        # Blend the overlay with the target image
        result = cv2.addWeighted(overlay, alpha, target_image, 1 - alpha, 0)
        # Use mask to apply only to the target region
        target = cv2.bitwise_and(result, result, mask=mask)
        background = cv2.bitwise_and(target_image, target_image, mask=cv2.bitwise_not(mask))
        final = cv2.add(target, background)
        return final
