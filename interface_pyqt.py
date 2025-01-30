# interface_pyqt.py

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QFileDialog,
                             QColorDialog, QSlider, QCheckBox, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from main import MakeupTryOn
import json
import logging
from src.makeup_config import load_makeup_configs
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class WebcamThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, makeup_tryon: MakeupTryOn):
        super().__init__()
        self.makeup_tryon = makeup_tryon

    def run(self):
        self.makeup_tryon.start_webcam_thread()
        while self.makeup_tryon.running:
            if not self.makeup_tryon.frame_queue.empty():
                frame = self.makeup_tryon.frame_queue.get()
                self.frame_ready.emit(frame)
            self.msleep(10)

class MakeupTryOnApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Virtual Makeup Try-On")
        self.makeup_tryon = MakeupTryOn()
        self.makeup_configs = load_makeup_configs()
        self.selected_makeup = {}
        for config in self.makeup_configs:
            self.selected_makeup[config.name] = {
                'enabled': True,
                'color': config.default_color,
                'intensity': config.default_intensity
            }
        self.init_ui()
        self.webcam_thread = WebcamThread(self.makeup_tryon)
        self.webcam_thread.frame_ready.connect(self.update_image)
    
    def init_ui(self):
        layout = QVBoxLayout()

        # Upload Reference Image Button
        upload_btn = QPushButton("Upload Reference Image")
        upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(upload_btn)

        # Makeup Selection Group
        makeup_group = QGroupBox("Select Makeup Types")
        makeup_layout = QVBoxLayout()
        for config in self.makeup_configs:
            makeup_type_layout = QHBoxLayout()

            chk = QCheckBox(config.name)
            chk.setChecked(True)
            makeup_type_layout.addWidget(chk)

            color_btn = QPushButton("Color")
            color_btn.clicked.connect(lambda checked, c=config.name: self.pick_color(c))
            makeup_type_layout.addWidget(color_btn)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(config.default_intensity * 100))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            slider.valueChanged.connect(lambda value, c=config.name: self.set_intensity(c, value))
            makeup_type_layout.addWidget(QLabel("Intensity"))
            makeup_type_layout.addWidget(slider)

            makeup_layout.addLayout(makeup_type_layout)

        makeup_group.setLayout(makeup_layout)
        layout.addWidget(makeup_group)

        # Start/Stop Makeup Button
        self.start_btn = QPushButton("Start Makeup")
        self.start_btn.clicked.connect(self.toggle_makeup)
        layout.addWidget(self.start_btn)

        # Capture Snapshot Button
        snapshot_btn = QPushButton("Capture Snapshot")
        snapshot_btn.clicked.connect(self.capture_snapshot)
        layout.addWidget(snapshot_btn)

        # Save/Load Parameters Buttons
        param_layout = QHBoxLayout()

        save_btn = QPushButton("Save Parameters")
        save_btn.clicked.connect(self.save_parameters)
        param_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Parameters")
        load_btn.clicked.connect(self.load_parameters)
        param_layout.addWidget(load_btn)

        layout.addLayout(param_layout)

        # Video Display Label
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        self.setLayout(layout)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Image Files (*.jpg *.jpeg *.png)", options=options
        )
        if file_path:
            logging.debug(f"Selected file path: {file_path}")
            try:
                if not self.is_valid_image(file_path):
                    logging.debug("Image validation failed.")
                    raise ValueError("Invalid image file.")
                logging.debug("Image validation passed.")
                self.makeup_tryon.load_reference_image(
                    file_path, 
                    makeup_types=[mt.name for mt in self.makeup_configs]
                )
                # Update makeup colors
                makeup_types_enabled = [mt.name for mt in self.makeup_configs if self.selected_makeup[mt.name]['enabled']]
                makeup_colors = self.makeup_tryon.makeup_transfer.extract_makeup_color(
                    self.makeup_tryon.reference_image,
                    self.makeup_tryon.landmarks[0],
                    makeup_types=makeup_types_enabled
                )
                for makeup_type, color in makeup_colors.items():
                    if makeup_type in self.selected_makeup:
                        self.selected_makeup[makeup_type]['color'] = color
                        logging.info(f"Updated color for {makeup_type}: {color}")
                QMessageBox.information(self, "Image Loaded", "Reference image loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading reference image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load reference image: {e}")

    def is_valid_image(self, path: str) -> bool:
        """
        Validates if the provided path points to a valid image.

        :param path: Path to the image file.
        :return: True if valid, False otherwise.
        """
        try:
            # Attempt to read with OpenCV
            img = cv2.imread(path)
            if img is not None:
                return True
            
            # Fallback to Pillow
            from PIL import Image
            with Image.open(path) as pil_img:
                pil_img.verify()  # Verify that it's an image
            return True
        except Exception as e:
            logging.error(f"Pillow failed to verify image: {e}")
            return False

    def pick_color(self, makeup_type: str):
        color = QColorDialog.getColor()
        if color.isValid():
            # Convert QColor to BGR tuple
            bgr_color = (color.blue(), color.green(), color.red())
            self.selected_makeup[makeup_type]['color'] = bgr_color
            logging.info(f"Selected color for {makeup_type}: {bgr_color}")

    def set_intensity(self, makeup_type: str, value: int):
        intensity = value / 100.0
        self.selected_makeup[makeup_type]['intensity'] = intensity
        logging.info(f"Set intensity for {makeup_type}: {intensity}")

    def toggle_makeup(self):
        if not self.makeup_tryon.running:
            # Gather makeup parameters
            makeup_params = {}
            for makeup_type, params in self.selected_makeup.items():
                if params['enabled']:
                    makeup_params[makeup_type] = {
                        'color': params['color'],
                        'intensity': params['intensity']
                    }
            if not makeup_params:
                QMessageBox.warning(self, "No Makeup Selected", "Please select at least one makeup type.")
                return
            self.makeup_tryon.set_makeup_params(makeup_params)
            self.webcam_thread.start()
            self.start_btn.setText("Stop Makeup")
        else:
            self.makeup_tryon.stop_webcam()
            self.webcam_thread.quit()
            self.webcam_thread.wait()
            self.start_btn.setText("Start Makeup")

    def capture_snapshot(self):
        if self.makeup_tryon.frame_queue.empty():
            QMessageBox.warning(self, "No Frame", "No frame available to capture.")
            return
        frame = self.makeup_tryon.frame_queue.get()
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)", options=options
        )
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Snapshot Saved", f"Snapshot saved at {save_path}")

    def save_parameters(self):
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "",
            "JSON Files (*.json)", options=options
        )
        if save_path:
            params = {}
            for makeup_type, param in self.selected_makeup.items():
                params[makeup_type] = {
                    'enabled': param['enabled'],
                    'color': param['color'],
                    'intensity': param['intensity']
                }
            try:
                with open(save_path, 'w') as f:
                    json.dump(params, f, indent=4)
                QMessageBox.information(self, "Parameters Saved", f"Makeup parameters saved at {save_path}")
            except Exception as e:
                logging.error(f"Error saving parameters: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save parameters: {e}")

    def load_parameters(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "",
            "JSON Files (*.json)", options=options
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                for makeup_type, param in params.items():
                    if makeup_type in self.selected_makeup:
                        self.selected_makeup[makeup_type]['enabled'] = param.get('enabled', True)
                        self.selected_makeup[makeup_type]['color'] = tuple(param.get('color', self.selected_makeup[makeup_type]['color']))
                        self.selected_makeup[makeup_type]['intensity'] = param.get('intensity', 0.5)
                        logging.info(f"Loaded parameters for {makeup_type}")
                QMessageBox.information(self, "Parameters Loaded", f"Makeup parameters loaded from {file_path}")
            except Exception as e:
                logging.error(f"Error loading parameters: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")

    def update_image(self, frame: np.ndarray):
        """
        Updates the video display with the latest frame.

        :param frame: Latest frame from the webcam in BGR format.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MakeupTryOnApp()
    window.show()
    sys.exit(app.exec_())
