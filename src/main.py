import cv2
import numpy as np
from face_detection import FaceDetector
from face_parsing import FaceParser
from makeup_transfer import MakeupTransfer
from utils.utils import load_image, save_image

def main():
    # Initialize components
    face_detector = FaceDetector()
    face_parser = FaceParser()
    makeup_transfer = MakeupTransfer()
    
    # Load reference image
    reference_path = 'assets/reference_images/reference1.jpg'  # Replace with your reference image
    reference_image = load_image(reference_path)
    
    # Detect face and parse
    faces = face_detector.detect_faces(reference_image)
    if not faces:
        print("No face detected in the reference image.")
        return
    parsing_map_ref = face_parser.parse(reference_image)
    
    # Extract makeup color from reference
    lipstick_color = makeup_transfer.extract_makeup_color(
        reference_image, 
        parsing_map_ref, 
        target_region=13  # Assuming 13 is the lip region
    )
    print(f"Extracted Lipstick Color (BGR): {lipstick_color}")
    
    # Start video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, replace with video file if needed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect face and landmarks
        faces = face_detector.detect_faces(frame)
        if faces:
            parsing_map = face_parser.parse(frame)
            # Apply makeup
            frame = makeup_transfer.apply_makeup(
                frame, 
                parsing_map, 
                makeup_color=lipstick_color, 
                target_region=13, 
                alpha=0.6
            )
        
        # Display the resulting frame
        cv2.imshow('Virtual Makeup Try-On', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
