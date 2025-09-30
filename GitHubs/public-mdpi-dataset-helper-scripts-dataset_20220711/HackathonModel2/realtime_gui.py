"""
Real-time RF Detection GUI
=========================
Live monitoring interface for RF frame detection with manual labeling capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
import os
import json
from typing import List, Tuple, Optional
from realtime_yolo_detector import RealtimeYOLODetector, Detection
from realtime_spectrogram_generator import RealtimeSpectrogramGenerator

class RealtimeRFDetectionGUI:
    """
    GUI for real-time RF detection monitoring and manual labeling.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-time RF Frame Detection System")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.detector = None
        self.generator = None
        self.running = False
        
        # Queues for communication
        self.spectrogram_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)
        
        # Current data
        self.current_image = None
        self.current_detections = []
        self.current_spectrogram_path = None
        
        # Manual labeling
        self.manual_labels = []
        self.labeling_mode = False
        self.current_label = "WLAN"  # Default label
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'wlan_count': 0,
            'bluetooth_count': 0,
            'collision_count': 0,
            'manual_labels': 0
        }
        
        self.setup_gui()
        self.setup_threads()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Image display area
        self.setup_image_display(main_frame)
        
        # Detection panel
        self.setup_detection_panel(main_frame)
        
        # Statistics panel
        self.setup_statistics_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="YOLO Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_path_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_path_var, width=47, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=(0, 5))
        ttk.Button(control_frame, text="Refresh", command=self.refresh_models).grid(row=0, column=2)
        ttk.Button(control_frame, text="Browse", command=self.browse_model).grid(row=0, column=3)
        
        # Initialize model list
        self.refresh_models()
        
        # Packet source
        ttk.Label(control_frame, text="Packet Source:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.packet_source_var = tk.StringVar(value="../spectrogram_training_data_20220711/")
        packet_entry = ttk.Entry(control_frame, textvariable=self.packet_source_var, width=50)
        packet_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="Browse", command=self.browse_packet_source).grid(row=1, column=2, pady=(5, 0))
        
        # Packet type selection
        ttk.Label(control_frame, text="Packet Type:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.packet_type_var = tk.StringVar(value="single")
        packet_type_combo = ttk.Combobox(control_frame, textvariable=self.packet_type_var, 
                                       values=["single", "merged", "both"], state="readonly", width=15)
        packet_type_combo.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Batch generation controls
        batch_frame = ttk.Frame(control_frame)
        batch_frame.grid(row=3, column=0, columnspan=4, pady=(10, 0), sticky=tk.W)
        
        ttk.Label(batch_frame, text="Batch Generate:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(batch_frame, text="Count:").pack(side=tk.LEFT, padx=(0, 2))
        self.batch_count_var = tk.StringVar(value="10")
        ttk.Entry(batch_frame, textvariable=self.batch_count_var, width=5).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(batch_frame, text="Packets:").pack(side=tk.LEFT, padx=(0, 2))
        self.batch_packets_var = tk.StringVar(value="5")
        ttk.Entry(batch_frame, textvariable=self.batch_packets_var, width=5).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(batch_frame, text="Generate Batch", command=self.generate_batch).pack(side=tk.LEFT, padx=(5, 0))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=4, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT)
    
    def setup_image_display(self, parent):
        """Setup the image display area."""
        image_frame = ttk.LabelFrame(parent, text="Spectrogram Display", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Image canvas
        self.image_canvas = tk.Canvas(image_frame, bg='black', width=800, height=400)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for manual labeling
        self.image_canvas.bind("<Button-1>", self.on_canvas_click)
        self.image_canvas.bind("<Motion>", self.on_canvas_motion)
        
        # Image info
        self.image_info_var = tk.StringVar(value="No image loaded")
        ttk.Label(image_frame, textvariable=self.image_info_var).pack(pady=(5, 0))
    
    def setup_detection_panel(self, parent):
        """Setup the detection results panel."""
        detection_frame = ttk.LabelFrame(parent, text="Detection Results", padding=10)
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Detection list
        self.detection_listbox = tk.Listbox(detection_frame, height=6)
        self.detection_listbox.pack(fill=tk.X, pady=(0, 10))
        
        # Manual labeling controls
        label_frame = ttk.Frame(detection_frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text="Manual Label:").pack(side=tk.LEFT, padx=(0, 5))
        self.label_var = tk.StringVar(value="WLAN")
        label_combo = ttk.Combobox(label_frame, textvariable=self.label_var, 
                                  values=["WLAN", "bluetooth", "collision"], state="readonly")
        label_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(label_frame, text="Add Manual Label", command=self.add_manual_label).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(label_frame, text="Clear Labels", command=self.clear_manual_labels).pack(side=tk.LEFT)
    
    def setup_statistics_panel(self, parent):
        """Setup the statistics panel."""
        stats_frame = ttk.LabelFrame(parent, text="Detection Statistics", padding=10)
        stats_frame.pack(fill=tk.X)
        
        # Statistics display
        self.stats_text = tk.Text(stats_frame, height=4, width=80)
        self.stats_text.pack(fill=tk.X)
        
        # Update statistics
        self.update_statistics()
    
    def setup_threads(self):
        """Setup background threads."""
        self.detection_thread = None
        self.gui_update_thread = None
    
    def refresh_models(self):
        """Refresh the list of available models."""
        models = []
        
        # Look for models in common locations
        search_paths = [
            "yolo_training_improved/",
            "yolo_training/",
            "checkpoints/",
            "models/",
            "weights/"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.pt'):
                            full_path = os.path.join(root, file)
                            models.append(full_path)
        
        # Remove duplicates and sort
        models = sorted(list(set(models)))
        
        # Update combo box
        self.model_combo['values'] = models
        if models:
            self.model_combo.set(models[0])  # Select first model by default
    
    def browse_model(self):
        """Browse for YOLO model file."""
        filename = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def generate_batch(self):
        """Generate a batch of spectrograms for testing."""
        try:
            packet_source = self.packet_source_var.get()
            packet_type = self.packet_type_var.get()
            batch_count = int(self.batch_count_var.get())
            batch_packets = int(self.batch_packets_var.get())
            
            if not os.path.exists(packet_source):
                messagebox.showerror("Error", f"Packet source directory not found: {packet_source}")
                return
            
            # Create generator with selected packet type
            generator = RealtimeSpectrogramGenerator(packet_type=packet_type)
            
            # Generate batch
            output_dir = "batch_spectrograms"
            results = generator.generate_multiple_spectrograms(
                base_dir=packet_source,
                output_dir=output_dir,
                num_spectrograms=batch_count,
                packets_per_spectrogram=batch_packets
            )
            
            if results:
                messagebox.showinfo("Success", f"Generated {len(results)} spectrograms in {output_dir}/")
                print(f"Batch generation completed: {len(results)} spectrograms saved to {output_dir}/")
            else:
                messagebox.showwarning("Warning", "No spectrograms were generated")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for batch count and packets")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate batch: {e}")
    
    def browse_packet_source(self):
        """Browse for packet source directory."""
        directory = filedialog.askdirectory(title="Select Packet Source Directory")
        if directory:
            self.packet_source_var.set(directory)
    
    def start_detection(self):
        """Start the detection system."""
        model_path = self.model_path_var.get()
        packet_source = self.packet_source_var.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        if not os.path.exists(packet_source):
            messagebox.showerror("Error", f"Packet source directory not found: {packet_source}")
            return
        
        try:
            # Initialize components
            self.detector = RealtimeYOLODetector(model_path)
            packet_type = self.packet_type_var.get()
            self.generator = RealtimeSpectrogramGenerator(packet_type=packet_type)
            
            # Start detection thread
            self.running = True
            self.detection_thread = threading.Thread(
                target=self.detection_loop,
                daemon=True
            )
            self.detection_thread.start()
            
            # Start GUI update thread
            self.gui_update_thread = threading.Thread(
                target=self.gui_update_loop,
                daemon=True
            )
            self.gui_update_thread.start()
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            print("Detection system started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")
    
    def stop_detection(self):
        """Stop the detection system."""
        self.running = False
        
        if self.detector:
            self.detector.stop_realtime_detection()
        if self.generator:
            self.generator.stop_realtime_processing()
        
        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        print("Detection system stopped")
    
    def detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                # Generate spectrogram
                if self.generator and os.path.exists(self.packet_source_var.get()):
                    # Get random packet files
                    packet_dir = self.packet_source_var.get()
                    packet_files = []
                    for root, dirs, files in os.walk(packet_dir):
                        for file in files:
                            if file.endswith('.packet'):
                                packet_files.append(os.path.join(root, file))
                    
                    if packet_files:
                        # Select random subset
                        n_packets = min(5, len(packet_files))
                        selected_files = np.random.choice(packet_files, size=n_packets, replace=False)
                        
                        # Generate spectrogram
                        image, spectrogram_path = self.generator.generate_from_packets(selected_files)
                        
                        if image is not None:
                            # Run detection
                            result_image, detections, output_path = self.detector.process_spectrogram(image)
                            
                            # Update statistics
                            self.update_detection_stats(detections)
                            
                            # Store current data
                            self.current_image = result_image
                            self.current_detections = detections
                            self.current_spectrogram_path = spectrogram_path
                
                time.sleep(2.0)  # Process every 2 seconds
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(1.0)
    
    def gui_update_loop(self):
        """GUI update loop."""
        while self.running:
            try:
                if self.current_image is not None:
                    # Update image display
                    self.update_image_display()
                    
                    # Update detection list
                    self.update_detection_list()
                
                time.sleep(0.5)  # Update GUI every 0.5 seconds
                
            except Exception as e:
                print(f"Error in GUI update loop: {e}")
                time.sleep(1.0)
    
    def update_image_display(self):
        """Update the image display."""
        if self.current_image is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling
                img_height, img_width = image_rgb.shape[:2]
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y)
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize image
                image_resized = cv2.resize(image_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                image_pil = Image.fromarray(image_resized)
                self.photo = ImageTk.PhotoImage(image_pil)
                
                # Update canvas
                self.image_canvas.delete("all")
                self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                             image=self.photo, anchor=tk.CENTER)
                
                # Update info
                info_text = f"Image: {self.current_spectrogram_path} | Detections: {len(self.current_detections)}"
                self.image_info_var.set(info_text)
    
    def update_detection_list(self):
        """Update the detection list display."""
        self.detection_listbox.delete(0, tk.END)
        
        for i, detection in enumerate(self.current_detections):
            text = f"{detection.class_name}: {detection.confidence:.3f} at ({detection.x1:.0f}, {detection.y1:.0f})"
            self.detection_listbox.insert(tk.END, text)
        
        # Add manual labels
        for i, label in enumerate(self.manual_labels):
            text = f"MANUAL: {label['class']} at ({label['x']:.0f}, {label['y']:.0f})"
            self.detection_listbox.insert(tk.END, text)
    
    def update_detection_stats(self, detections):
        """Update detection statistics."""
        self.detection_stats['total_detections'] += len(detections)
        
        for detection in detections:
            if detection.class_name == 'WLAN':
                self.detection_stats['wlan_count'] += 1
            elif detection.class_name == 'bluetooth':
                self.detection_stats['bluetooth_count'] += 1
            elif detection.class_name == 'collision':
                self.detection_stats['collision_count'] += 1
    
    def update_statistics(self):
        """Update the statistics display."""
        stats_text = f"""Total Detections: {self.detection_stats['total_detections']}
WLAN: {self.detection_stats['wlan_count']} | Bluetooth: {self.detection_stats['bluetooth_count']} | Collision: {self.detection_stats['collision_count']}
Manual Labels: {self.detection_stats['manual_labels']}"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        
        # Schedule next update
        self.root.after(1000, self.update_statistics)
    
    def on_canvas_click(self, event):
        """Handle canvas click for manual labeling."""
        if self.labeling_mode and self.current_image is not None:
            # Convert canvas coordinates to image coordinates
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.current_image.shape[:2]
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y)
                
                # Convert to image coordinates
                img_x = (event.x - canvas_width//2) / scale + img_width//2
                img_y = (event.y - canvas_height//2) / scale + img_height//2
                
                # Add manual label
                self.add_manual_label_at_position(img_x, img_y)
    
    def on_canvas_motion(self, event):
        """Handle canvas mouse motion."""
        pass  # Could add hover effects here
    
    def add_manual_label(self):
        """Add manual label at current position."""
        self.labeling_mode = True
        print("Click on the image to add a manual label")
    
    def add_manual_label_at_position(self, x, y):
        """Add manual label at specific position."""
        label = {
            'class': self.label_var.get(),
            'x': x,
            'y': y,
            'timestamp': time.time()
        }
        self.manual_labels.append(label)
        self.detection_stats['manual_labels'] += 1
        
        # Redraw image with manual labels
        self.draw_manual_labels()
    
    def draw_manual_labels(self):
        """Draw manual labels on the current image."""
        if self.current_image is not None:
            image_with_labels = self.current_image.copy()
            
            for label in self.manual_labels:
                x, y = int(label['x']), int(label['y'])
                class_name = label['class']
                
                # Draw circle for manual label
                color = (0, 255, 255) if class_name == 'WLAN' else (255, 0, 255) if class_name == 'bluetooth' else (255, 255, 0)
                cv2.circle(image_with_labels, (x, y), 10, color, 2)
                cv2.putText(image_with_labels, f"M:{class_name}", (x+15, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Update current image
            self.current_image = image_with_labels
    
    def clear_manual_labels(self):
        """Clear all manual labels."""
        self.manual_labels.clear()
        self.detection_stats['manual_labels'] = 0
    
    def load_image(self):
        """Load an image for testing."""
        filename = filedialog.askopenfilename(
            title="Load Spectrogram Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if filename:
            self.current_image = cv2.imread(filename)
            if self.detector:
                result_image, detections, output_path = self.detector.process_spectrogram(self.current_image)
                self.current_detections = detections
                self.current_image = result_image
            print(f"Loaded image: {filename}")
    
    def save_results(self):
        """Save detection results."""
        if self.current_image is not None:
            filename = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                cv2.imwrite(filename, self.current_image)
                
                # Save detection data
                results_data = {
                    'detections': [
                        {
                            'class': det.class_name,
                            'confidence': det.confidence,
                            'bbox': det.bbox
                        } for det in self.current_detections
                    ],
                    'manual_labels': self.manual_labels,
                    'statistics': self.detection_stats
                }
                
                json_filename = filename.replace('.png', '_data.json')
                with open(json_filename, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
                print(f"Results saved to: {filename}")
                print(f"Detection data saved to: {json_filename}")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    app = RealtimeRFDetectionGUI()
    app.run()
