import cv2
from src.face_detection import FaceDetector
from src.face_parsing import FaceParser
from src.makeup_transfer import MakeupTransfer
import threading
import queue
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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
        self.frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames

    def load_reference_image(self, reference_path):
        """
        Loads the reference image, detects the face, parses the facial regions,
        and extracts the average lipstick color.
        """
        logging.info(f"Loading reference image from: {reference_path}")
        # Load the image
        image = cv2.imread(reference_path)
        if image is None:
            logging.error("Failed to load the reference image. Please check the file path.")
            raise ValueError("Failed to load the reference image.")

        # Detect faces
        faces = self.face_detector.detect_faces(image)
        if not faces:
            logging.error("No faces detected in the reference image.")
            raise ValueError("No faces detected in the reference image.")

        # For simplicity, consider the first detected face
        face = faces[0]
        logging.info("Face detected in the reference image.")

        # Parse facial regions
        parsing_map = self.face_parser.parse(image)
        if parsing_map is None:
            logging.error("Failed to parse the facial regions.")
            raise ValueError("Failed to parse the facial regions.")

        logging.info("Facial regions parsed successfully.")

        # Extract the lip region (assuming label 12 corresponds to lips)
        lip_mask = (parsing_map == 12).astype(np.uint8)  # Modify label if different
        if np.sum(lip_mask) == 0:
            logging.error("No lip region found in the reference image.")
            raise ValueError("No lip region found in the reference image.")

        # Calculate the average color within the lip region
        lip_pixels = image[lip_mask == 1]
        average_color = np.mean(lip_pixels, axis=0)  # BGR format

        # Store the color as integers
        self.lipstick_color = tuple(map(int, average_color))
        logging.info(f"Lipstick color extracted: {self.lipstick_color} (B, G, R)")

    def start_webcam(self, display_callback, visualize_segmentation=False, alpha=0.6):
        if self.lipstick_color is None:
            logging.error("Reference image not loaded.")
            raise ValueError("Reference image not loaded. Please load a reference image first.")
        
        if self.running:
            logging.error("Webcam is already running.")
            return
        
        logging.info("Attempting to open webcam...")
        retries = 5
        for attempt in range(1, retries + 1):
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                logging.info(f"Webcam successfully opened on attempt {attempt}.")
                break
            else:
                logging.warning(f"Attempt {attempt} failed to open webcam. Retrying in 0.5 seconds...")
                time.sleep(0.5)
        else:
            logging.error("Unable to access the webcam after multiple attempts.")
            raise ValueError("Unable to access the webcam.")
        
        # Set frame dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.running = True
        logging.info("Webcam started.")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame from webcam.")
                    break
                else:
                    logging.debug("Frame read successfully.")
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                if faces:
                    face = faces[0]
                    if visualize_segmentation:
                        parsing_map, color_seg = self.face_parser.parse(frame, return_full_map=True)
                        frame = self.makeup_transfer.apply_makeup(
                            frame, 
                            parsing_map, 
                            makeup_color=self.lipstick_color, 
                            target_regions=[11, 12],  # Modify if different
                            alpha=alpha
                        )
                        frame = cv2.addWeighted(frame, 0.7, color_seg, 0.3, 0)
                        logging.info("Makeup applied with segmentation visualization.")
                    else:
                        parsing_map = self.face_parser.parse(frame)
                        frame = self.makeup_transfer.apply_makeup(
                            frame, 
                            parsing_map, 
                            makeup_color=self.lipstick_color, 
                            target_regions=[11, 12],  # Modify if different
                            alpha=alpha
                        )
                        logging.info("Makeup applied.")
                else:
                    logging.info("No face detected. Skipping makeup application.")
                    continue  # Skip to the next frame without applying makeup
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Enqueue frame
                if not self.frame_queue.full():
                    self.frame_queue.put(rgb_frame)
                else:
                    logging.warning("Frame queue is full. Discarding frame.")
                
                # Sleep briefly to reduce CPU usage
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"An error occurred in the webcam thread: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logging.info("Webcam resource released.")
            self.running = False
            logging.info("Webcam stopped.")

    def stop_webcam(self):
        if not self.running:
            logging.warning("Webcam is not running.")
            return
        self.running = False
        logging.info("Stopping webcam...")
