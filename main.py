# main.py

import cv2
from src.face_detection import FaceDetector
from src.makeup_transfer import MakeupTransfer
from utils.visualization import overlay_segmentation
import threading
import queue
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MakeupTryOn:
    def __init__(self, frame_width=640, frame_height=480):
        # Initialize components
        self.face_detector = FaceDetector()
        self.makeup_transfer = MakeupTransfer()
        self.makeup_colors = {}  # Dictionary to store colors per makeup type
        self.cap = None
        self.running = False
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_queue = queue.Queue(maxsize=10)  # Queue to hold frames

    def convert_rgb_to_bgr(self, rgb_color):
        """
        Converts an RGB color tuple to BGR.

        :param rgb_color: Tuple of (R, G, B)
        :return: Tuple of (B, G, R)
        """
        if not isinstance(rgb_color, tuple) or len(rgb_color) != 3:
            logging.error("RGB color must be a tuple of 3 elements.")
            raise ValueError("RGB color must be a tuple of 3 elements.")
        return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

    def load_reference_image(self, reference_path, makeup_types=['Lipstick']):
        """
        Loads the reference image, detects the face, parses the facial regions,
        and extracts the average makeup color(s) based on the selected makeup types.

        :param reference_path: Path to the reference image.
        :param makeup_types: List of makeup types to extract.
        """
        logging.info(f"Loading reference image from: {reference_path}")
        # Load the image
        image = cv2.imread(reference_path)
        if image is None:
            logging.error("Failed to load the reference image. Please check the file path.")
            raise ValueError("Failed to load the reference image.")

        # Detect faces and landmarks
        faces_landmarks = self.face_detector.detect_faces(image)
        if not faces_landmarks:
            logging.error("No faces detected in the reference image.")
            raise ValueError("No faces detected in the reference image.")

        # For simplicity, consider the first detected face
        landmarks = faces_landmarks[0]
        logging.info("Face detected in the reference image.")

        # Extract makeup colors based on the selected makeup styles
        makeup_colors = self.makeup_transfer.extract_makeup_color(image, landmarks, makeup_types=makeup_types)
        self.makeup_colors.update(makeup_colors)
        logging.info(f"Makeup colors extracted: {self.makeup_colors}")

    def start_webcam(self, display_callback, visualize_segmentation=False, makeup_params={}):
        """
        Starts the webcam and applies makeup in real-time based on the provided parameters.

        :param display_callback: Function to call with the processed frame for display.
        :param visualize_segmentation: Boolean indicating whether to visualize segmentation.
        :param makeup_params: Dictionary with makeup types as keys and their parameters as values.
                               Each value should be a dictionary with 'color' (BGR tuple) and 'intensity' (float)
        """
        if not self.makeup_colors:
            logging.error("Makeup colors not loaded.")
            raise ValueError("Makeup colors not loaded. Please load a reference image first.")

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
        
        # Set frame dimensions (optional, can be removed or adjusted)
        if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width):
            logging.warning(f"Failed to set frame width to {self.frame_width}")
        else:
            logging.debug(f"Frame width set to {self.frame_width}")
        
        if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height):
            logging.warning(f"Failed to set frame height to {self.frame_height}")
        else:
            logging.debug(f"Frame height set to {self.frame_height}")
        
        self.running = True
        logging.info("Webcam started.")
        
        try:
            while self.running:
                logging.debug("Attempting to read frame from webcam.")
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame from webcam.")
                    break
                else:
                    logging.debug("Frame read successfully.")
                
                # Detect faces and landmarks
                faces_landmarks = self.face_detector.detect_faces(frame)
                if faces_landmarks:
                    for landmarks in faces_landmarks:
                        # Apply makeup based on the provided makeup parameters
                        frame = self.makeup_transfer.apply_makeup(
                            frame, 
                            landmarks, 
                            makeup_params=makeup_params
                        )
                        logging.info("Makeup applied.")

                        if visualize_segmentation:
                            # Overlay segmentation masks for each makeup type
                            frame = overlay_segmentation(
                                frame, 
                                landmarks, 
                                makeup_types=list(makeup_params.keys())
                            )
                            logging.debug("Segmentation overlay applied.")
                else:
                    logging.info("No face detected. Skipping makeup application.")
                    continue  # Skip makeup application

                # Convert to RGB for Tkinter compatibility
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
