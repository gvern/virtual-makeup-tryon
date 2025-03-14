# src/face_detection.py

import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_faces(self, image):
        """
        Detects faces and returns a list of facial landmarks.

        :param image: BGR image from OpenCV
        :return: List of landmarks for each detected face
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        faces_landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    ih, iw, _ = image.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks.append((x, y))
                faces_landmarks.append(landmarks)
        return faces_landmarks
