# interface.py

import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from src.main import MakeupTryOn
import json
import os
import logging
from src.makeup_config import load_makeup_configs

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MakeupTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Virtual Makeup Try-On")
        self.makeup_tryon = MakeupTryOn()

        # Load makeup configurations
        self.makeup_configs = load_makeup_configs()

        # Initialize variables
        self.selected_makeup = {}
        for config in self.makeup_configs:
            self.selected_makeup[config.name] = {
                'enabled': tk.BooleanVar(value=True),
                'color': config.default_color,
                'intensity': tk.DoubleVar(value=config.default_intensity)
            }

        self.setup_ui()

    def setup_ui(self):
        # Upload reference image button
        upload_btn = tk.Button(self.root, text="Upload Reference Image", command=self.upload_image)
        upload_btn.pack(pady=10)

        # Makeup selection frame
        makeup_frame = tk.LabelFrame(self.root, text="Select Makeup Types")
        makeup_frame.pack(padx=10, pady=10, fill="both", expand=True)

        for config in self.makeup_configs:
            frame = tk.Frame(makeup_frame)
            frame.pack(fill="x", padx=5, pady=2)

            # Makeup type checkbox
            chk = tk.Checkbutton(frame, text=config.name, variable=self.selected_makeup[config.name]['enabled'])
            chk.pack(side="left")

            # Color picker button
            color_btn = tk.Button(frame, text="Color", command=lambda c=config.name: self.pick_color(c))
            color_btn.pack(side="left", padx=5)

            # Intensity slider
            slider = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05,
                              orient="horizontal", label="Intensity",
                              variable=self.selected_makeup[config.name]['intensity'])
            slider.pack(side="left", padx=5)

        # Start/Stop button
        self.start_btn = tk.Button(self.root, text="Start Makeup", command=self.toggle_makeup)
        self.start_btn.pack(pady=10)

        # Snapshot button
        snapshot_btn = tk.Button(self.root, text="Capture Snapshot", command=self.capture_snapshot)
        snapshot_btn.pack(pady=5)

        # Save/Load parameters
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=5)

        save_btn = tk.Button(param_frame, text="Save Parameters", command=self.save_parameters)
        save_btn.pack(side="left", padx=5)

        load_btn = tk.Button(param_frame, text="Load Parameters", command=self.load_parameters)
        load_btn.pack(side="left", padx=5)

        # Video display area (placeholder)
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.makeup_tryon.load_reference_image(file_path, makeup_types=[mt.name for mt in self.makeup_configs])
                # Update makeup colors based on the reference image
                makeup_types_enabled = [mt.name for mt in self.makeup_configs if self.selected_makeup[mt.name]['enabled'].get()]
                makeup_colors = self.makeup_tryon.makeup_transfer.extract_makeup_color(
                    self.makeup_tryon.reference_image,
                    self.makeup_tryon.landmarks[0],
                    makeup_types=makeup_types_enabled
                )
                # Update selected_makeup colors
                for makeup_type, color in makeup_colors.items():
                    if makeup_type in self.selected_makeup:
                        self.selected_makeup[makeup_type]['color'] = color
                        logging.info(f"Updated color for {makeup_type}: {color}")
                messagebox.showinfo("Image Loaded", "Reference image loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading reference image: {e}")
                messagebox.showerror("Error", f"Failed to load reference image: {e}")

    def pick_color(self, makeup_type):
        color_code = colorchooser.askcolor(title=f"Choose color for {makeup_type}")
        if color_code and color_code[0]:
            # Convert RGB to BGR
            bgr_color = (int(color_code[0][2]), int(color_code[0][1]), int(color_code[0][0]))
            self.selected_makeup[makeup_type]['color'] = bgr_color
            logging.info(f"Selected color for {makeup_type}: {bgr_color}")

    def toggle_makeup(self):
        if not self.makeup_tryon.running:
            # Gather makeup parameters
            makeup_params = {}
            for makeup_type, params in self.selected_makeup.items():
                if params['enabled'].get():
                    makeup_params[makeup_type] = {
                        'color': params['color'],
                        'intensity': params['intensity'].get()
                    }
            if not makeup_params:
                messagebox.showwarning("No Makeup Selected", "Please select at least one makeup type.")
                return
            self.makeup_tryon.set_makeup_params(makeup_params)
            self.makeup_tryon.start_webcam()
            self.start_btn.config(text="Stop Makeup")
        else:
            self.makeup_tryon.stop_webcam()
            self.start_btn.config(text="Start Makeup")

    def capture_snapshot(self):
        if self.makeup_tryon.frame_queue.empty():
            messagebox.showwarning("No Frame", "No frame available to capture.")
            return
        frame = self.makeup_tryon.frame_queue.get()
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg *.jpeg")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Snapshot Saved", f"Snapshot saved at {save_path}")

    def save_parameters(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON Files", "*.json")])
        if save_path:
            params = {}
            for makeup_type, param in self.selected_makeup.items():
                params[makeup_type] = {
                    'enabled': param['enabled'].get(),
                    'color': param['color'],
                    'intensity': param['intensity'].get()
                }
            try:
                with open(save_path, 'w') as f:
                    json.dump(params, f, indent=4)
                messagebox.showinfo("Parameters Saved", f"Makeup parameters saved at {save_path}")
            except Exception as e:
                logging.error(f"Error saving parameters: {e}")
                messagebox.showerror("Error", f"Failed to save parameters: {e}")

    def load_parameters(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                for makeup_type, param in params.items():
                    if makeup_type in self.selected_makeup:
                        self.selected_makeup[makeup_type]['enabled'].set(param.get('enabled', True))
                        self.selected_makeup[makeup_type]['color'] = tuple(param.get('color', self.selected_makeup[makeup_type]['color']))
                        self.selected_makeup[makeup_type]['intensity'].set(param.get('intensity', 0.5))
                        logging.info(f"Loaded parameters for {makeup_type}")
                messagebox.showinfo("Parameters Loaded", f"Makeup parameters loaded from {file_path}")
            except Exception as e:
                logging.error(f"Error loading parameters: {e}")
                messagebox.showerror("Error", f"Failed to load parameters: {e}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = MakeupTryOnApp(root)
    app.run()
