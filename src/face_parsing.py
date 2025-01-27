import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        # Define the BiSeNet architecture here or load from a repository
        # For simplicity, let's assume you have the model defined elsewhere
        # and you are loading the weights
        # Example:
        from model import BiSeNetModel  # Replace with actual import
        self.bisenet = BiSeNetModel(n_classes=n_classes)
    
    def forward(self, x):
        return self.bisenet(x)

class FaceParser:
    def __init__(self, model_path='models/bisenet.pth', device='cpu'):
        self.device = torch.device(device)
        self.n_classes = 19
        self.model = BiSeNet(self.n_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def parse(self, image):
        # image: BGR image (from OpenCV)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing  # Shape: (512, 512)
