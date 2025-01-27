# src/main.py

import cv2
from src.face_detection import FaceDetector
from src.face_parsing import FaceParser
from src.makeup_transfer import MakeupTransfer

class MakeupTryOn:
    def __init__(self, device='cpu', frame_width=480, frame_height=360):
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_parser = FaceParser(device=device)
        self.makeup_transfer = MakeupTransfer()
        self.lipstick_color = None
        self.cap = None
        self.running = False
        self.frame_width = frame_width
        self.frame_height = frame_height

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
        parsing_map_ref, _ = self.face_parser.parse(reference_image, return_full_map=True)
        
        # Extract lipstick color (Target regions 11 and 12)
        self.lipstick_color = self.makeup_transfer.extract_makeup_color(
            reference_image, 
            parsing_map_ref, 
            target_regions=[11, 12]  # Updated to include both upper and lower lip
        )
        print(f"Extracted Lipstick Color (BGR): {self.lipstick_color}")

    def start_webcam(self, display_callback, visualize_segmentation=False, alpha=0.6):
        if self.lipstick_color is None:
            raise ValueError("Reference image not loaded. Please load a reference image first.")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Unable to access the webcam.")
        
        # Set frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect face in the frame
            faces = self.face_detector.detect_faces(frame)
            if faces:
                if visualize_segmentation:
                    parsing_map, color_seg = self.face_parser.parse(frame, return_full_map=True)
                    # Apply makeup to multiple regions
                    frame = self.makeup_transfer.apply_makeup(
                        frame, 
                        parsing_map, 
                        makeup_color=self.lipstick_color, 
                        target_regions=[11, 12],  # Updated to include both regions
                        alpha=alpha
                    )
                    # Overlay the segmentation map (optional)
                    frame = cv2.addWeighted(frame, 0.7, color_seg, 0.3, 0)
                    print("Makeup applied with segmentation visualization.")
                else:
                    parsing_map = self.face_parser.parse(frame)
                    # Apply makeup to multiple regions
                    frame = self.makeup_transfer.apply_makeup(
                        frame, 
                        parsing_map, 
                        makeup_color=self.lipstick_color, 
                        target_regions=[11, 12], 
                        alpha=alpha
                    )
                    print("Makeup applied.")
            else:
                print("No face detected. Skipping makeup application.")
            
            # Convert the frame to RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Callback to update the GUI
            display_callback(rgb_frame)
        
        self.cap.release()

    def stop_webcam(self):
        self.running = False
