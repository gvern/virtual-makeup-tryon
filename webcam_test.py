import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)  # Try 0, 1, 2 if multiple webcams are present
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Webcam opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('Webcam Test - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
