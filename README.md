# Real-Time Virtual Makeup Try-On

A proof of concept for a real-time virtual makeup try-on application using BiSeNet for face parsing. This application allows users to input a reference image and see themselves with the transferred makeup in real time via their webcam.

## Features

- **Face Detection & Landmark Tracking:** Utilizes MediaPipe for efficient and real-time face detection.
- **Face Parsing:** Employs BiSeNet for semantic segmentation of facial regions.
- **Makeup Transfer:** Applies lipstick color from a reference image to the user's face in real time.
- **Real-Time Performance:** Processes webcam feed and applies makeup augmentation live.

## Repository Structure


virtual-makeup-tryon/ ├── README.md ├── requirements.txt ├── models/ │ └── bisenet.pth ├── assets/ │ └── reference_images/ ├── src/ │ ├── face_detection.py │ ├── face_parsing.py │ ├── makeup_transfer.py │ └── main.py ├── utils/ │ ├── utils.py │ └── visualization.py └── scripts/ └── download_models.sh

bash
Copier

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/virtual-makeup-tryon.git
cd virtual-makeup-tryon
2. Create and Activate a Virtual Environment
bash
Copier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copier
pip install -r requirements.txt
4. Download Pre-trained Models
bash
Copier
chmod +x scripts/download_models.sh
./scripts/download_models.sh
5. Add Reference Images
Place your reference images in the assets/reference_images/ directory. Ensure the images have clear and prominent makeup for better results.

6. Run the Application
bash
Copier
python src/main.py
A window titled "Virtual Makeup Try-On" will open, displaying your webcam feed with the applied makeup from the reference image. Press 'q' to exit.

Usage
Choose a Reference Image: Select a reference image with the desired makeup style and place it in the assets/reference_images/ directory.
Run the Application: Execute main.py to start the live makeup transfer.
Interact in Real-Time: Move your face naturally in front of the webcam to see the makeup applied dynamically.
Dependencies
Python 3.6+
OpenCV
PyTorch
MediaPipe
NumPy
Pillow
dlib
Future Enhancements
Multiple Makeup Regions: Extend to eyeshadow, blush, foundation, etc.
Attribute Controls: Allow users to adjust makeup intensity, glossiness, and color.
Mobile Deployment: Optimize and deploy the application on Android and iOS devices.
Advanced Rendering: Incorporate differentiable rendering for better lighting and geometry adaptation.
License
MIT License

Acknowledgments
BiSeNet for Face Parsing
MediaPipe for Face Detection and Landmark Tracking
markdown
Copier

---

## **7. Additional Tips**

- **Model Accuracy:** Ensure that the BiSeNet model is accurately parsing the facial regions. You may need to fine-tune the model or use higher-resolution inputs for better segmentation.
  
- **Makeup Realism:** To achieve more realistic makeup effects, consider blending techniques that preserve the skin texture and account for lighting variations. Techniques like alpha blending with edge smoothing can help.

- **Performance Monitoring:** Monitor the frame rate and optimize bottlenecks. Profiling tools like **cProfile** can help identify slow parts of the code.

- **Error Handling:** Implement robust error handling for cases where no face is detected or parsing fails.

- **Extensibility:** Structure your code to easily add new makeup types or switch between different makeup styles.

---

## **8. References**

- **BiSeNet for Real-Time Semantic Segmentation:** [GitHub Repository](https://github.com/ZhaoJ9014/face-parsing.PyTorch)
- **MediaPipe Face Detection and Face Mesh:** [Google MediaPipe](https://google.github.io/mediapipe/)
- **OpenCV Documentation:** [OpenCV](https://opencv.org/)
- **PyTorch Documentation:** [PyTorch](https://pytorch.org/docs/stable/index.html)

---

By following the above guide, you should be able to create a functional proof of concept for a real-time virtual makeup try-on application. This PoC focuses on lipstick application, but the framework can be extended to include other makeup types and more sophisticated style transfers as outlined in your initial project vision.

Feel free to reach out if you encounter any issues or need further assistance!