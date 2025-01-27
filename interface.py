# interface.py

import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox
from PIL import Image, ImageTk
import threading
import cv2
from main import MakeupTryOn
import numpy as np
import torch
import queue
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MakeupApp:
    def __init__(self, root):
        logging.info("Initializing MakeupApp GUI...")
        self.root = root
        self.root.title("Real-Time Virtual Makeup Try-On")
        self.root.geometry("1200x800")  # Increased window size for better layout

        # Initialize MakeupTryOn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")
        self.makeup_tryon = MakeupTryOn(device=device, frame_width=640, frame_height=480)

        # Configure grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        # Reference Image Frame
        self.ref_frame = tk.LabelFrame(root, text="Reference Image", padx=10, pady=10)
        self.ref_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.ref_image_label = tk.Label(self.ref_frame)
        self.ref_image_label.pack()

        self.upload_button = tk.Button(self.ref_frame, text="Upload Reference Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Lipstick Color Display
        self.color_frame = tk.LabelFrame(root, text="Extracted Makeup Color", padx=10, pady=10)
        self.color_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.color_display = tk.Canvas(self.color_frame, width=100, height=50)
        self.color_display.pack(pady=5)

        # Makeup Style Selection
        self.style_frame = tk.LabelFrame(root, text="Makeup Style", padx=10, pady=10)
        self.style_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.makeup_style_var = tk.StringVar(value='Lipstick')
        self.makeup_styles = ['Lipstick', 'Eyeshadow', 'Blush', 'Foundation']
        self.style_menu = tk.OptionMenu(self.style_frame, self.makeup_style_var, *self.makeup_styles)
        self.style_menu.pack(pady=5)

        # Webcam Feed Frame
        self.webcam_frame = tk.LabelFrame(root, text="Webcam Feed", padx=10, pady=10)
        self.webcam_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.webcam_label = tk.Label(self.webcam_frame)
        self.webcam_label.pack()

        # Controls Frame inside Webcam Feed
        self.controls_frame = tk.Frame(self.webcam_frame)
        self.controls_frame.pack(pady=10)

        self.start_button = tk.Button(self.controls_frame, text="Start Makeup", command=self.start_makeup)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(self.controls_frame, text="Stop Makeup", command=self.stop_makeup, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Option to visualize segmentation
        self.visualize_var = tk.BooleanVar()
        self.visualize_check = tk.Checkbutton(
            self.controls_frame, 
            text="Visualize Segmentation", 
            variable=self.visualize_var
        )
        self.visualize_check.grid(row=1, column=0, columnspan=2, pady=5)

        # Makeup Intensity Slider
        self.alpha_label = tk.Label(self.controls_frame, text="Makeup Intensity:")
        self.alpha_label.grid(row=2, column=0, pady=5, sticky="e")
        self.alpha_slider = tk.Scale(
            self.controls_frame, 
            from_=0.0, 
            to=1.0, 
            resolution=0.05, 
            orient=tk.HORIZONTAL
        )
        self.alpha_slider.set(0.6)  # Default value
        self.alpha_slider.grid(row=2, column=1, pady=5, sticky="w")

        # Color Picker Button
        self.color_picker_button = tk.Button(
            self.controls_frame, 
            text="Pick Makeup Color", 
            command=self.pick_makeup_color
        )
        self.color_picker_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Capture Snapshot Button
        self.snapshot_button = tk.Button(
            self.controls_frame, 
            text="Capture Snapshot", 
            command=self.capture_snapshot
        )
        self.snapshot_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.thread = None  # Track the thread instance
        self.running = False
        self.current_frame = None  # To store the latest frame
        self.update_delay = 30  # milliseconds

        # Start the periodic GUI update
        self.root.after(self.update_delay, self.process_queue)

    def upload_image(self):
        logging.info("Upload Image button clicked.")
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                logging.info(f"Loading reference image from: {file_path}")
                # Load and process the reference image
                makeup_style = self.makeup_style_var.get()
                self.makeup_tryon.load_reference_image(file_path, makeup_type=makeup_style)

                # Display the reference image
                img = Image.open(file_path)
                img = img.resize((250, 250), Image.LANCZOS)
                self.ref_photo = ImageTk.PhotoImage(img)
                self.ref_image_label.configure(image=self.ref_photo)

                # Display the makeup color
                if self.makeup_tryon.makeup_color is None:
                    logging.error("Makeup color extraction failed.")
                    raise ValueError("Makeup color extraction failed.")

                b, g, r = self.makeup_tryon.makeup_color
                color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                self.color_display.delete("all")  # Clear previous color
                self.color_display.create_rectangle(10, 10, 90, 40, fill=color_hex, outline=color_hex)
                logging.info(f"Makeup Color Displayed: {color_hex}")

                messagebox.showinfo("Success", "Reference image loaded successfully!")
                logging.info("Reference image loaded and displayed.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                logging.error(f"Error loading reference image: {e}")

    def start_makeup(self):
        logging.info("Start Makeup button clicked.")
        if self.makeup_tryon.makeup_color is None:
            messagebox.showwarning("Warning", "Please upload a reference image first.")
            logging.warning("Makeup try-on not started: Reference image not loaded.")
            return

        if self.makeup_tryon.running:
            messagebox.showwarning("Warning", "Makeup application is already running.")
            logging.warning("Makeup try-on is already running.")
            return

        # Get the current alpha value from the slider
        alpha = self.alpha_slider.get()
        makeup_style = self.makeup_style_var.get()

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True

        # Clear the frame queue before starting
        with self.makeup_tryon.frame_queue.mutex:
            self.makeup_tryon.frame_queue.queue.clear()
            logging.info("Frame queue cleared.")

        # Start the webcam in a separate thread
        self.thread = threading.Thread(
            target=self.makeup_tryon.start_webcam, 
            args=(self.update_webcam_feed, self.visualize_var.get(), makeup_style, alpha),  # positional args
            daemon=True  # Daemon thread to ensure it exits with the main program
        )
        self.thread.start()
        logging.info("Webcam feed thread started.")

    def stop_makeup(self):
        logging.info("Stop Makeup button clicked.")
        if not self.makeup_tryon.running:
            messagebox.showwarning("Warning", "Makeup application is not running.")
            logging.warning("Makeup try-on is not running.")
            return

        self.makeup_tryon.stop_webcam()
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        logging.info("Makeup application stopped.")

        # Join the thread to ensure it has fully terminated
        if self.thread and self.thread.is_alive():
            self.thread.join()
            logging.info("Webcam thread joined successfully.")

    def update_webcam_feed(self, frame):
        """
        This method is called by the webcam thread to enqueue frames for the main thread to process.
        """
        try:
            self.makeup_tryon.frame_queue.put_nowait(frame)
        except queue.Full:
            # If the queue is full, discard the frame to maintain performance
            logging.debug("Frame queue is full. Discarding frame.")
            pass

    def process_queue(self):
        """
        Periodically called to process frames from the queue and update the GUI.
        """
        try:
            while not self.makeup_tryon.frame_queue.empty():
                frame = self.makeup_tryon.frame_queue.get_nowait()
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the Label
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)

                # Store the current frame for snapshot
                self.current_frame = frame.copy()
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error processing frame from queue: {e}")

        # Schedule the next queue check
        self.root.after(self.update_delay, self.process_queue)

    def pick_makeup_color(self):
        color_code = colorchooser.askcolor(title="Choose Makeup Color")
        if color_code and color_code[0]:
            r, g, b = color_code[0]
            # Use the convert_rgb_to_bgr function from MakeupTryOn
            try:
                self.makeup_tryon.makeup_color = self.makeup_tryon.convert_rgb_to_bgr((r, g, b))
                color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                self.color_display.delete("all")  # Clear previous color
                self.color_display.create_rectangle(10, 10, 90, 40, fill=color_hex, outline=color_hex)
                logging.info(f"Custom Makeup Color Selected: {color_hex}")
            except ValueError as ve:
                messagebox.showerror("Error", str(ve))
                logging.error(f"Error converting color: {ve}")

    def capture_snapshot(self):
        if self.current_frame is not None:
            # Generate a unique filename
            timestamp = int(time.time())
            filename = f'snapshot_{timestamp}.png'
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Snapshot Captured", f"Snapshot saved as {filename}")
            logging.info(f"Snapshot saved as {filename}")
        else:
            messagebox.showwarning("No Frame", "No frame available to capture.")
            logging.warning("No frame available to capture.")

    def on_closing(self):
        logging.info("Closing application...")
        if self.makeup_tryon.running:
            self.makeup_tryon.stop_webcam()
            if self.thread and self.thread.is_alive():
                self.thread.join()
                logging.info("Webcam thread joined during application close.")
        self.root.destroy()
        logging.info("Application closed.")

def main():
    logging.info("Launching GUI...")
    root = tk.Tk()
    app = MakeupApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    logging.info("GUI main loop ended.")

if __name__ == "__main__":
    main()
