# src/main.py

from src.face_detection import FaceDetector
from src.makeup_transfer import MakeupTransfer
import threading
import cv2
import logging
import queue

class MakeupTryOn:
    def __init__(self):
        """
        Initializes the MakeupTryOn class with necessary components.
        """
        self.face_detector = FaceDetector()
        self.makeup_transfer = MakeupTransfer()
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)  # Prevent unlimited growth
        self.thread = None
        self.makeup_params = {}
        self.makeup_params_lock = threading.Lock()
        self.reference_image = None
        self.landmarks = []
    
    def set_makeup_params(self, makeup_params: dict):
        """
        Sets the makeup parameters to be applied.
        
        :param makeup_params: Dictionary with makeup types as keys and parameters as values.
        """
        with self.makeup_params_lock:
            self.makeup_params = makeup_params
            logging.debug(f"Makeup parameters set to: {makeup_params}")
    
    def load_reference_image(self, reference_path: str, makeup_types: list):
        """
        Loads the reference image and detects facial landmarks.
        
        :param reference_path: Path to the reference image file.
        :param makeup_types: List of makeup types to extract from the reference image.
        """
        logging.info(f"Loading reference image from {reference_path}")
        self.reference_image = cv2.imread(reference_path)
        if self.reference_image is None:
            logging.error(f"Failed to load reference image from {reference_path}")
            raise ValueError(f"Failed to load reference image from {reference_path}")
        self.landmarks = self.face_detector.detect_faces(self.reference_image)
        if not self.landmarks:
            logging.error("No faces detected in the reference image.")
            raise ValueError("No faces detected in the reference image.")
        logging.info("Reference image loaded and faces detected.")
        
        # Extract makeup colors based on the reference image
        makeup_colors = self.makeup_transfer.extract_makeup_color(
            self.reference_image,
            self.landmarks[0],  # Assuming single face
            makeup_types=makeup_types
        )
        # Update makeup parameters with extracted colors
        with self.makeup_params_lock:
            for makeup_type, color in makeup_colors.items():
                if makeup_type in self.makeup_params:
                    self.makeup_params[makeup_type]['color'] = color
                    logging.debug(f"Updated color for {makeup_type}: {color}")
    
    def start_webcam_thread(self):
        """
        Starts the webcam thread for real-time makeup application.
        """
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._webcam_loop, daemon=True)
            self.thread.start()
            logging.info("Webcam thread started.")
    
    def stop_webcam(self):
        """
        Stops the webcam thread and releases resources.
        """
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            logging.info("Webcam thread stopped.")
    
    def _webcam_loop(self):
        """
        The main loop for webcam feed processing.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Unable to access the webcam.")
            self.running = False
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from webcam.")
                continue
            # Detect faces and landmarks
            faces_landmarks = self.face_detector.detect_faces(frame)
            if faces_landmarks:
                # For simplicity, use the first detected face
                with self.makeup_params_lock:
                    makeup_params = self.makeup_params.copy()
                # Apply makeup
                frame = self.makeup_transfer.apply_makeup(frame, faces_landmarks[0], makeup_params)
            # Put the processed frame into the queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        cap.release()
        logging.info("Webcam resource released.")
