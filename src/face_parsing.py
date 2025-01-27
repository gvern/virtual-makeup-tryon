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
    
    def parse(self, image):
        """
        Perform face parsing on the input image.

        :param image: BGR image (from OpenCV)
        :return: Segmentation map as a NumPy array with shape (H, W)
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
        
        return segmentation  # Shape: (H, W)
