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
        self.root.geometry("1400x900")  # Increased window size for better layout

        # Initialize MakeupTryOn
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # logging.info(f"Using device: {device}")
        self.makeup_tryon = MakeupTryOn( frame_width=640, frame_height=480)

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

        # Makeup Color Display Frame
        self.color_frame = tk.LabelFrame(root, text="Extracted Makeup Colors", padx=10, pady=10)
        self.color_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Initialize color display canvases for each makeup type
        self.makeup_types = ['Lipstick', 'Eyeshadow', 'Eyebrow', 'Foundation']
        self.color_canvases = {}
        for i, makeup_type in enumerate(self.makeup_types):
            frame = tk.Frame(self.color_frame)
            frame.pack(pady=5, anchor='w')

            label = tk.Label(frame, text=f"{makeup_type} Color:")
            label.pack(side=tk.LEFT)

            canvas = tk.Canvas(frame, width=50, height=25, bg='white', highlightthickness=1, highlightbackground="black")
            canvas.pack(side=tk.LEFT, padx=5)
            self.color_canvases[makeup_type] = canvas

        # Makeup Style Selection Frame
        self.style_frame = tk.LabelFrame(root, text="Makeup Style Selection", padx=10, pady=10)
        self.style_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Checkboxes for each makeup type
        self.selected_makeups = {}
        for i, makeup_type in enumerate(self.makeup_types):
            var = tk.BooleanVar()
            chk = tk.Checkbutton(
                self.style_frame, 
                text=makeup_type, 
                variable=var, 
                command=self.update_makeup_controls
            )
            chk.grid(row=i, column=0, sticky='w', pady=2)
            self.selected_makeups[makeup_type] = var

            # Color picker button
            btn = tk.Button(
                self.style_frame, 
                text=f"Pick {makeup_type} Color", 
                command=lambda mt=makeup_type: self.pick_makeup_color(mt),
                state=tk.DISABLED
            )
            btn.grid(row=i, column=1, padx=5, pady=2)

            # Intensity slider
            slider = tk.Scale(
                self.style_frame, 
                from_=0.0, 
                to=1.0, 
                resolution=0.05, 
                orient=tk.HORIZONTAL,
                label=f"{makeup_type} Intensity",
                state=tk.DISABLED
            )
            slider.set(0.6)  # Default value
            slider.grid(row=i, column=2, padx=5, pady=2)

            # Store references to buttons and sliders
            self.selected_makeups[makeup_type+'_btn'] = btn
            self.selected_makeups[makeup_type+'_slider'] = slider

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



        # Snapshot and Save Makeup Parameters
        self.snapshot_button = tk.Button(
            self.controls_frame, 
            text="Capture Snapshot", 
            command=self.capture_snapshot
        )
        self.snapshot_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.save_params_button = tk.Button(
            self.controls_frame, 
            text="Save Makeup Parameters", 
            command=self.save_makeup_parameters
        )
        self.save_params_button.grid(row=3, column=0, columnspan=2, pady=5)

        # Load Makeup Parameters
        self.load_params_button = tk.Button(
            self.controls_frame, 
            text="Load Makeup Parameters", 
            command=self.load_makeup_parameters
        )
        self.load_params_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.thread = None  # Track the thread instance
        self.running = False
        self.current_frame = None  # To store the latest frame
        self.update_delay = 30  # milliseconds

        # Start the periodic GUI update
        self.root.after(self.update_delay, self.process_queue)

    def update_makeup_controls(self):
        """
        Enable or disable makeup controls based on selection.
        """
        for makeup_type in self.makeup_types:
            selected = self.selected_makeups[makeup_type].get()
            btn = self.selected_makeups[makeup_type+'_btn']
            slider = self.selected_makeups[makeup_type+'_slider']
            if selected:
                btn.config(state=tk.NORMAL)
                slider.config(state=tk.NORMAL)
            else:
                btn.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)

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
                selected_makeups = [mt for mt, var in self.selected_makeups.items() if mt in self.makeup_types and self.selected_makeups[mt].get()]
                if not selected_makeups:
                    messagebox.showwarning("No Makeup Selected", "Please select at least one makeup type.")
                    logging.warning("No makeup type selected for extraction.")
                    return

                # Load reference image
                self.makeup_tryon.load_reference_image(file_path, makeup_types=selected_makeups)

                # Display the reference image
                img = Image.open(file_path)
                img = img.resize((250, 250), Image.LANCZOS)
                self.ref_photo = ImageTk.PhotoImage(img)
                self.ref_image_label.configure(image=self.ref_photo)

                # Display the makeup colors
                for makeup_type in self.makeup_types:
                    color = self.makeup_tryon.makeup_colors.get(makeup_type, (255, 255, 255))  # Default to white if not set
                    b, g, r = color
                    color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                    canvas = self.color_canvases[makeup_type]
                    canvas.delete("all")  # Clear previous color
                    canvas.create_rectangle(0, 0, 50, 25, fill=color_hex, outline=color_hex)
                    logging.info(f"Makeup Color Displayed for {makeup_type}: {color_hex}")

                messagebox.showinfo("Success", "Reference image loaded successfully!")
                logging.info("Reference image loaded and displayed.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                logging.error(f"Error loading reference image: {e}")

    def start_makeup(self):
        logging.info("Start Makeup button clicked.")
        if not self.makeup_tryon.makeup_colors:
            messagebox.showwarning("Warning", "Please upload a reference image first.")
            logging.warning("Makeup try-on not started: Reference image not loaded.")
            return

        if self.makeup_tryon.running:
            messagebox.showwarning("Warning", "Makeup application is already running.")
            logging.warning("Makeup try-on is already running.")
            return

        # Get the intensity values from sliders
        makeup_params = {}
        for makeup_type in self.makeup_types:
            if self.selected_makeups[makeup_type].get():
                intensity = self.selected_makeups[makeup_type+'_slider'].get()
                makeup_params[makeup_type] = {
                    'intensity': intensity,
                    'color': self.makeup_tryon.makeup_colors.get(makeup_type, (0, 0, 255))
                }

        if not makeup_params:
            messagebox.showwarning("No Makeup Selected", "Please select at least one makeup type.")
            logging.warning("No makeup type selected for application.")
            return


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
            args=(self.update_webcam_feed, self.visualize_var.get(), makeup_params),  # positional args
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

    def pick_makeup_color(self, makeup_type):
        color_code = colorchooser.askcolor(title=f"Choose {makeup_type} Color")
        if color_code and color_code[0]:
            r, g, b = color_code[0]
            # Use the convert_rgb_to_bgr function from MakeupTryOn
            try:
                bgr_color = self.makeup_tryon.convert_rgb_to_bgr((r, g, b))
                # Update the stored makeup color
                self.makeup_tryon.makeup_colors[makeup_type] = bgr_color
                # Update the color display
                color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                canvas = self.color_canvases[makeup_type]
                canvas.delete("all")  # Clear previous color
                canvas.create_rectangle(0, 0, 50, 25, fill=color_hex, outline=color_hex)
                logging.info(f"Custom Makeup Color Selected for {makeup_type}: {color_hex}")
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

    def save_makeup_parameters(self):
        """
        Saves the current makeup parameters to a file.
        """
        import json
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Makeup Parameters"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.makeup_tryon.makeup_params, f)
                messagebox.showinfo("Success", f"Makeup parameters saved to {file_path}")
                logging.info(f"Makeup parameters saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters: {e}")
                logging.error(f"Failed to save makeup parameters: {e}")

    def load_makeup_parameters(self):
        """
        Loads makeup parameters from a file.
        """
        import json
        file_path = filedialog.askopenfilename(
            title="Load Makeup Parameters",
            filetypes=[("JSON Files", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                # Apply the loaded parameters
                for makeup_type, attributes in params.items():
                    if makeup_type in self.makeup_types:
                        self.selected_makeups[makeup_type].set(True)
                        self.update_makeup_controls()
                        # Set color
                        b, g, r = attributes.get('color', (255, 255, 255))
                        color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                        canvas = self.color_canvases[makeup_type]
                        canvas.delete("all")
                        canvas.create_rectangle(0, 0, 50, 25, fill=color_hex, outline=color_hex)
                        # Set intensity
                        intensity = attributes.get('intensity', 0.6)
                        slider = self.selected_makeups[makeup_type+'_slider']
                        slider.set(intensity)
                        # Update the stored color
                        self.makeup_tryon.makeup_colors[makeup_type] = (b, g, r)
                messagebox.showinfo("Success", f"Makeup parameters loaded from {file_path}")
                logging.info(f"Makeup parameters loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {e}")
                logging.error(f"Failed to load makeup parameters: {e}")

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
