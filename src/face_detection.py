import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=0, 
            min_detection_confidence=detection_confidence
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def detect_faces(self, image):
        results = self.face_detection.process(image)
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = [
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                ]
                faces.append(bbox)
        return faces

    def get_landmarks(self, image):
        results = self.face_mesh.process(image)
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = []
                for id, lm_pt in enumerate(face_landmarks.landmark):
                    ih, iw, _ = image.shape
                    x, y = int(lm_pt.x * iw), int(lm_pt.y * ih)
                    lm.append((x, y))
                landmarks.append(lm)
        return landmarks
