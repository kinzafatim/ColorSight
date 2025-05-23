import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

TRANSFORMS = {
    "Normal": np.eye(3),
    "Protanomaly ": [[0.817, 0.183, 0], [0.333, 0.667, 0], [0, 0.125, 0.875]], # Mild red-weak color blindness
    "Deuteranomaly": [[0.8, 0.2, 0], [0.258, 0.742, 0], [0, 0.142, 0.858]], # Mild green-weak color blindness 
    "Tritanomaly": [[0.967, 0.033, 0], [0, 0.733, 0.267], [0, 0.183, 0.817]], # Mild blue-weak color blindness
    "Protanopia": [[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]], # Severe red-blind color blindness
    "Deuteranopia": [[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0.3, 0.7]], # Severe green-blind color blindness
    "Tritanopia": [[0.95, 0.05, 0], [0, 0.433, 0.567], [0, 0.475, 0.525]], # Severe blue-Yellow-blind color blindness
    "Monochrome": "grayscale" # Complete color blindness
}

def apply_transform(image, matrix):
    if isinstance(matrix, str) and matrix == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.merge([gray, gray, gray])
    flat = image.reshape((-1, 3))
    transformed = np.clip(flat @ np.array(matrix).T, 0, 255)
    return transformed.reshape(image.shape).astype(np.uint8)

class ColorBlindSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Vision Simulator")
        self.root.geometry("1200x800")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=6)
        self.style.configure("Title.TLabel", font=('Helvetica', 14, 'bold'), background="#f0f0f0")
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create widgets
        self.create_controls()
        self.create_image_panels()
        self.create_status_bar()
        
        # Initialize variables
        self.img = None
        self.img_path = None
        self.preview_size = (400, 400)
        self.tk_images = []

    def create_controls(self):
        # control panel
        control_frame = ttk.Frame(self.main_frame, padding=(10, 5))
        control_frame.pack(fill=tk.X)
        
        # Load image button
        self.btn_load = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # Checkboxes for color vision types
        self.lbl_filters = ttk.Label(control_frame, text="Color Vision Types:", style="Title.TLabel")
        self.lbl_filters.pack(side=tk.LEFT, padx=15)
        
        self.checkbox_vars = {}
        checkbox_frame = ttk.Frame(control_frame)
        checkbox_frame.pack(side=tk.LEFT, padx=5)
        
        for name in TRANSFORMS:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(checkbox_frame, text=name, variable=var)
            chk.pack(anchor='w', pady=1)
            self.checkbox_vars[name] = var


        # Action buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=20)
        
        self.btn_simulate = ttk.Button(btn_frame, text="Apply Simulation", command=self.simulate)
        self.btn_simulate.pack(pady=2)
        
        self.btn_clear = ttk.Button(btn_frame, text="Clear Results", command=self.clear_results)
        self.btn_clear.pack(pady=2)

    def create_image_panels(self):
        #Create image display panels
        img_frame = ttk.Frame(self.main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image panel
        self.original_frame = ttk.LabelFrame(img_frame, text="Original Image", padding=10)
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_original = ttk.Label(self.original_frame)
        self.lbl_original.pack(fill=tk.BOTH, expand=True)
        
        # Simulated images panel
        self.simulated_frame = ttk.LabelFrame(img_frame, text="Simulated Results", padding=10)
        self.simulated_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas and scrollbar for simulated images
        self.canvas = tk.Canvas(self.simulated_frame, bg="white")
        self.scrollbar = ttk.Scrollbar(self.simulated_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_status_bar(self):
        #Create status bar at bottom
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("Ready")

    def load_image(self): #Load and display selected image
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            try:
                self.img = cv2.imread(path)
                self.img_path = path
                self.show_original_image()
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def show_original_image(self):
        #Display original image in left panel
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail(self.preview_size, Image.LANCZOS)
        self.tk_original = ImageTk.PhotoImage(pil_img)
        self.lbl_original.configure(image=self.tk_original)

    def simulate(self):
        #Apply selected simulations and display results
        if self.img is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        selected_names = [name for name, var in self.checkbox_vars.items() if var.get()]
        if not selected_names:
            messagebox.showwarning("No Selection", "Please select at least one simulation type.")
            return

        # Clear previous results
        self.clear_results()

        # Process selected simulations
        for name in selected_names:
            matrix = TRANSFORMS[name]

            # Apply color transformation
            transformed = apply_transform(self.img.copy(), matrix)

            # Convert and resize for display
            img_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((300, 300), Image.LANCZOS)

            # Create display panel for each result
            frame = ttk.Frame(self.scrollable_frame, padding=5)
            frame.pack(fill=tk.X, pady=5)

            # Display image
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl_img = ttk.Label(frame, image=tk_img)
            lbl_img.image = tk_img  # Keep reference
            lbl_img.pack(side=tk.LEFT)

            # Display simulation name
            lbl_text = ttk.Label(frame, text=name, font=('Helvetica', 11, 'bold'))
            lbl_text.pack(side=tk.LEFT, padx=10)

            self.tk_images.append(tk_img)  # Maintain references

        self.status_var.set(f"Applied {len(selected_names)} simulations")


    def clear_results(self): #Clear results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.tk_images = []
        self.status_var.set("Cleared results")


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorBlindSimulatorApp(root)
    root.mainloop()