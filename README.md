# Real-Time Virtual Makeup Try-On

A proof of concept for a real-time virtual makeup try-on application. This application allows users to upload a reference image and see themselves with the transferred makeup in real-time via their webcam.

## Features

- **Face Detection & Landmark Tracking:** Utilizes MediaPipe for efficient and real-time face detection.
- **Face Parsing:** Employs BiSeNet for semantic segmentation of facial regions.
- **Makeup Transfer:** Applies lipstick color from a reference image to the user's face in real-time.
- **Custom Makeup Color Selection:** Allows users to pick and apply custom makeup colors.
- **Snapshot Capture:** Enables users to capture snapshots of the makeup application.
- **Visual Segmentation (Optional):** Provides an option to visualize the segmentation overlay on the webcam feed.
- **Real-Time Performance:** Processes webcam feed and applies makeup augmentation live.

## Repository Structure

gvern-virtual-makeup-tryon/ ├── README.md ├── interface.py ├── main.py ├── requirements.txt ├── webcam_test.py ├── assets/ │ └── reference_images/ ├── src/ │ ├── init.py │ ├── face_detection.py │ ├── face_parsing.py │ └── makeup_transfer.py └── utils/ ├── init.py ├── utils.py └── visualization.py


- **interface.py:** The main GUI application that users interact with.
- **main.py:** Contains the `MakeupTryOn` class responsible for loading images, processing webcam feed, and applying makeup.
- **requirements.txt:** Lists all the Python dependencies required for the project.
- **webcam_test.py:** A simple script to test webcam functionality.
- **assets/reference_images/:** Directory to store reference images with desired makeup styles.
- **src/:** Contains modules for face detection, face parsing, and makeup transfer.
- **utils/:** Utility scripts for image handling and visualization.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gvern-virtual-makeup-tryon.git
cd gvern-virtual-makeup-tryon
```
### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download Pre-trained Models
Ensure that you have the necessary pre-trained models for face detection and parsing. If not included, you may need to download them manually or modify the scripts to fetch them automatically.

### 5. Add Reference Images
Place your reference images in the assets/reference_images/ directory. Ensure the images have clear and prominent makeup for better results.

### 6. Run the Application
```bash
python interface.py
```
A window titled "Real-Time Virtual Makeup Try-On" will open, displaying your webcam feed with the applied makeup from the reference image.

## Usage
1- Upload a Reference Image:

- Click on the "Upload Reference Image" button.
- Select a valid image (sample_makeup.jpeg) with a clear view of the lips.

2- Customize Makeup (Optional):

- Use the "Pick Makeup Color" button to choose a custom lipstick color.
- Adjust the "Makeup Intensity" slider to control the makeup's opacity.

3- Start Makeup Application:

- Click on the "Start Makeup" button to begin the real-time makeup try-on.
- The webcam feed will display with the applied makeup from the reference image.

4- Visualize Segmentation (Optional):

- Check the "Visualize Segmentation" option to see the segmentation overlay on the webcam feed.

5- Capture Snapshot:

- Click on the "Capture Snapshot" button to save an image of the current makeup application.

6- Stop Makeup Application:

- Click on the "Stop Makeup" button to end the makeup try-on session and release webcam resources.

## Dependencies
- Python 3.6+
- OpenCV
- PyTorch
- MediaPipe
- NumPy
- Pillow
- dlib
- Transformers
- HuggingFace Hub

## Future Enhancements
- Multiple Makeup Regions: Extend to eyeshadow, blush, foundation, etc.
- Attribute Controls: Allow users to adjust makeup intensity, glossiness, and color.
- Mobile Deployment: Optimize and deploy the application on Android and iOS devices.
- Advanced Rendering: Incorporate differentiable rendering for better lighting and geometry adaptation.
- Improved Face Detection: Enhance face detection accuracy under various lighting and angles.
- User Interface Enhancements: Improve the GUI for better user experience and accessibility.

## License
MIT License

## Acknowledgments
BiSeNet for Face Parsing: GitHub Repository
MediaPipe for Face Detection and Landmark Tracking: Google MediaPipe
OpenCV Documentation: OpenCV
PyTorch Documentation: PyTorch
Transformers by HuggingFace: Transformers