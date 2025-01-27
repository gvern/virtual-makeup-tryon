# src/interface.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
from main import MakeupTryOn
import numpy as np
import torch

class MakeupApp:
    def __init__(self, root):
        print("Initializing MakeupApp GUI...")
        self.root = root
        self.root.title("Real-Time Virtual Makeup Try-On")
        self.root.geometry("1000x700")  # Increased height for additional elements
        
        # Initialize MakeupTryOn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.makeup_tryon = MakeupTryOn(device=device, frame_width=640, frame_height=480)
        
        # Reference Image Frame
        self.ref_frame = tk.LabelFrame(root, text="Reference Image", padx=10, pady=10)
        self.ref_frame.pack(side=tk.TOP, padx=10, pady=10)
        
        self.ref_image_label = tk.Label(self.ref_frame)
        self.ref_image_label.pack()
        
        self.upload_button = tk.Button(self.ref_frame, text="Upload Reference Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        # Lipstick Color Display
        self.color_frame = tk.LabelFrame(root, text="Extracted Lipstick Color", padx=10, pady=10)
        self.color_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.color_display = tk.Canvas(self.color_frame, width=100, height=50)
        self.color_display.pack(pady=5)
        
        # Webcam Feed Frame
        self.webcam_frame = tk.LabelFrame(root, text="Webcam Feed", padx=10, pady=10)
        self.webcam_frame.pack(side=tk.TOP, padx=10, pady=10)
        
        self.webcam_label = tk.Label(self.webcam_frame)
        self.webcam_label.pack()
        
        self.start_button = tk.Button(self.webcam_frame, text="Start Makeup", command=self.start_makeup)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.webcam_frame, text="Stop Makeup", command=self.stop_makeup, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        # Option to visualize segmentation
        self.visualize_var = tk.BooleanVar()
        self.visualize_check = tk.Checkbutton(
            self.webcam_frame, 
            text="Visualize Segmentation", 
            variable=self.visualize_var
        )
        self.visualize_check.pack(pady=5)
        
        self.thread = None
        self.running = False

    def upload_image(self):
        print("Upload Image button clicked.")
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                print(f"Loading reference image from: {file_path}")
                # Load and process the reference image
                self.makeup_tryon.load_reference_image(file_path)
                
                # Display the reference image
                img = Image.open(file_path)
                img = img.resize((250, 250), Image.LANCZOS)  # Updated here
                self.ref_photo = ImageTk.PhotoImage(img)
                self.ref_image_label.configure(image=self.ref_photo)
                
                # Display the lipstick color
                b, g, r = self.makeup_tryon.lipstick_color
                color_hex = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                self.color_display.create_rectangle(10, 10, 90, 40, fill=color_hex, outline=color_hex)
                print(f"Lipstick Color Displayed: {color_hex}")
                
                messagebox.showinfo("Success", "Reference image loaded successfully!")
                print("Reference image loaded and displayed.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                print(f"Error loading reference image: {e}")

    def start_makeup(self):
        print("Start Makeup button clicked.")
        if self.makeup_tryon.lipstick_color is None:
            messagebox.showwarning("Warning", "Please upload a reference image first.")
            print("Makeup try-on not started: Reference image not loaded.")
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.thread = threading.Thread(
            target=self.makeup_tryon.start_webcam, 
            args=(self.update_webcam_feed, self.visualize_var.get())
        )
        self.thread.start()
        print("Webcam feed started.")

    def stop_makeup(self):
        print("Stop Makeup button clicked.")
        self.makeup_tryon.stop_webcam()
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        print("Webcam feed stopped.")

    def update_webcam_feed(self, frame):
        if not self.running:
            print("Not running. Skipping frame update.")
            return
        
        # Convert the frame to ImageTk format
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the Label
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)
        # Optional: To prevent excessive print statements, you can comment out the following line
        # print("Webcam frame updated.")

    def on_closing(self):
        print("Closing application...")
        if self.running:
            self.stop_makeup()
        self.root.destroy()
        print("Application closed.")

def main():
    print("Launching GUI...")
    root = tk.Tk()
    app = MakeupApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    print("GUI main loop ended.")

if __name__ == "__main__":
    main()
