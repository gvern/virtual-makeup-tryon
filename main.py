# src/main.py

import cv2
from src.face_detection import FaceDetector
from src.face_parsing import FaceParser
from src.makeup_transfer import MakeupTransfer

class MakeupTryOn:
    def __init__(self, device='cpu'):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_parser = FaceParser(device=device)
        self.makeup_transfer = MakeupTransfer()
        self.lipstick_color = None
        self.cap = None
        self.running = False

    def load_reference_image(self, reference_path):
        # Load reference image
        reference_image = cv2.imread(reference_path)
        if reference_image is None:
            raise ValueError(f"Unable to load image at {reference_path}")
        
        # Detect face in reference image
        faces = self.face_detector.detect_faces(reference_image)
        if not faces:
            raise ValueError("No face detected in the reference image.")
        
        # Parse face
        parsing_map_ref = self.face_parser.parse(reference_image)
        
        # Extract lipstick color (Assuming label 12 corresponds to lips)
        self.lipstick_color = self.makeup_transfer.extract_makeup_color(
            reference_image, 
            parsing_map_ref, 
            target_region=10 # Update if label differs
        )
        print(f"Extracted Lipstick Color (BGR): {self.lipstick_color}")

    def start_webcam(self, display_callback):
        if self.lipstick_color is None:
            raise ValueError("Reference image not loaded. Please load a reference image first.")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to access the webcam.")
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect face in the frame
            faces = self.face_detector.detect_faces(frame)
            if faces:
                parsing_map = self.face_parser.parse(frame)
                # Apply makeup
                frame = self.makeup_transfer.apply_makeup(
                    frame, 
                    parsing_map, 
                    makeup_color=self.lipstick_color, 
                    target_region=10,  # Update if label differs
                    alpha=0.6
                )
            
            # Convert the frame to RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Callback to update the GUI
            display_callback(rgb_frame)
        
        self.cap.release()

    def stop_webcam(self):
        self.running = False
