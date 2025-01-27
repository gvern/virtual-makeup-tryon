# src/face_parsing.py

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2
import numpy as np

class FaceParser:
    def __init__(self, model_name='jonathandinu/face-parsing', device='cpu'):
        self.device = torch.device(device)
        # Initialize the image processor and model from HuggingFace
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def parse(self, image, return_full_map=False):
        """
        Perform face parsing on the input image.

        :param image: BGR image (from OpenCV)
        :param return_full_map: If True, returns the colored segmentation map
        :return: Segmentation map as a NumPy array with shape (H, W), and optionally the colored map
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare the image for the model
        inputs = self.image_processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted segmentation logits
        logits = outputs.logits
        
        # Upsample logits to match the input image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode='bilinear',
            align_corners=False
        )
        
        # Get the segmentation map by taking the argmax
        segmentation = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        if return_full_map:
            # Define a color map for visualization (customize as needed)
            num_classes = segmentation.max() + 1
            colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
            color_seg = colors[segmentation]
            return segmentation, color_seg
        else:
            return segmentation  # Shape: (H, W)
