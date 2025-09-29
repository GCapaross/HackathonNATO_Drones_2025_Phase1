#!/usr/bin/env python3
"""
Live Interactive RF Detection Pipeline
=====================================
Continuous spectrogram generation with real-time detection and human review.

Features:
- Generate spectrograms one by one
- Model analyzes each spectrogram automatically
- Human review and correction interface
- Continuous pipeline with manual oversight
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import json
import threading
import time
import queue
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_spectrogram_generator import RealtimeSpectrogramGenerator
from realtime_yolo_detector import RealtimeYOLODetector

class LiveInteractivePipeline:
    """
    Live interactive pipeline for continuous RF detection and human review.
    """
    
    def __init__(self, model_path: str, packet_source: str, output_dir: str = "live_output"):
        self.model_path = model_path
        self.packet_source = packet_source
        self.output_dir = output_dir
        self.current_image = None
        self.current_detections = []
        self.manual_boxes = []
        self.current_box = None
        self.drawing = False
        
        # Live processing state
        self.is_generating = False
        self.generation_interval = 3.0  # seconds
        self.packets_per_spectrogram = 5
        
        # Initialize components
        self.spectrogram_generator = RealtimeSpectrogramGenerator(
            fs=42e6,
            section_length=0.1,
            noise_sigma=0.1,
            packet_type="single"
        )
        self.yolo_detector = RealtimeYOLODetector(model_path)
        
        # Class names and colors
        self.class_names = ['WLAN', 'collision', 'bluetooth']
        self.class_colors = ['red', 'blue', 'green']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup GUI
        self.setup_gui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_gui(self):
        """Setup the live interactive GUI."""
        self.root = tk.Tk()
        self.root.title("Live Interactive RF Detection Pipeline")
        self.root.geometry("1400x900")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Live Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E, pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value=self.model_path)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_path_var, width=50, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load", command=self.load_selected_model).pack(side=tk.LEFT, padx=5)
        
        # Source info
        ttk.Label(control_frame, text=f"Source: {self.packet_source}").grid(row=1, column=0, sticky=tk.W)
        
        # Live processing controls
        live_frame = ttk.Frame(control_frame)
        live_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(live_frame, text="Generate Next Spectrogram", 
                  command=self.generate_next_spectrogram).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(live_frame, text="Run Detection", 
                  command=self.run_detection).pack(side=tk.LEFT, padx=5)
        
        # Processing settings
        settings_frame = ttk.LabelFrame(control_frame, text="Processing Settings", padding=5)
        settings_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(settings_frame, text="Packets per Spectrogram:").pack(side=tk.LEFT)
        self.packets_var = tk.StringVar(value="5")
        packets_entry = ttk.Entry(settings_frame, textvariable=self.packets_var, width=10)
        packets_entry.pack(side=tk.LEFT, padx=5)
        
        # Review controls
        review_frame = ttk.Frame(control_frame)
        review_frame.grid(row=4, column=0, columnspan=3, pady=5)
        
        ttk.Button(review_frame, text="✓ Approve All", 
                  command=self.approve_all, style="Success.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(review_frame, text="✗ Reject All", 
                  command=self.reject_all, style="Danger.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(review_frame, text="Save & Next", 
                  command=self.save_and_next).pack(side=tk.LEFT, padx=5)
        
        # Manual labeling controls
        manual_frame = ttk.LabelFrame(control_frame, text="Manual Labeling", padding=5)
        manual_frame.grid(row=5, column=0, columnspan=3, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(manual_frame, text="Class:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value="WLAN")
        class_combo = ttk.Combobox(manual_frame, textvariable=self.class_var, 
                                  values=self.class_names, state="readonly", width=10)
        class_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(manual_frame, text="Clear Manual", 
                  command=self.clear_manual_boxes).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Start live processing to begin")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, pady=5)
        
        # Image display
        self.setup_image_display(main_frame)
        
        # Detection list
        self.setup_detection_list(main_frame)
        
        # Configure styles
        self.setup_styles()
        
        # Initialize model list
        self.refresh_models()
        
    def setup_image_display(self, parent):
        """Setup the image display area."""
        image_frame = ttk.LabelFrame(parent, text="Live Spectrogram", padding=5)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse events for manual labeling
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
    def setup_detection_list(self, parent):
        """Setup the detection list panel."""
        detection_frame = ttk.LabelFrame(parent, text="Detections", padding=5)
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Treeview for detections
        columns = ('Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2', 'Source')
        self.detection_tree = ttk.Treeview(detection_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(detection_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        self.detection_tree.configure(yscrollcommand=scrollbar.set)
        
        self.detection_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Detection actions
        action_frame = ttk.Frame(detection_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Delete Selected", 
                  command=self.delete_selected_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Edit Selected", 
                  command=self.edit_selected_detection).pack(side=tk.LEFT, padx=5)
        
    def setup_styles(self):
        """Setup custom button styles."""
        style = ttk.Style()
        style.configure("Success.TButton", foreground="green")
        style.configure("Danger.TButton", foreground="red")
        
    def refresh_models(self):
        """Refresh the list of available models."""
        models = []
        
        # Look for .pt files in common directories
        search_dirs = [
            ".",
            "yolo_training_improved",
            "../YOLO_detection/checkpoints",
            "../OurCNN/checkpoints"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.pt'):
                            model_path = os.path.join(root, file)
                            models.append(model_path)
        
        # Remove duplicates and sort
        models = sorted(list(set(models)))
        
        # Update combo box
        self.model_combo['values'] = models
        if models and self.model_path in models:
            self.model_combo.set(self.model_path)
        elif models:
            self.model_combo.set(models[0])
    
    def browse_model(self):
        """Browse for YOLO model file."""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            self.model_path_var.set(file_path)
            self.load_selected_model()
    
    def load_selected_model(self):
        """Load the selected model."""
        selected_model = self.model_path_var.get()
        
        if not selected_model:
            messagebox.showwarning("Warning", "No model selected")
            return
            
        if not os.path.exists(selected_model):
            messagebox.showerror("Error", f"Model file not found: {selected_model}")
            return
            
        try:
            # Update model path
            self.model_path = selected_model
            
            # Reload YOLO detector with new model
            self.yolo_detector = RealtimeYOLODetector(selected_model, confidence_threshold=0.3)
            
            # Clear current detections
            self.current_detections = []
            self.display_image()
            self.update_detection_list()
            
            self.status_var.set(f"Model loaded: {os.path.basename(selected_model)}")
            messagebox.showinfo("Success", f"Model loaded successfully: {os.path.basename(selected_model)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def generate_next_spectrogram(self):
        """Generate the next spectrogram manually."""
        try:
            packets = int(self.packets_var.get())
            
            # Generate spectrogram
            results = self.spectrogram_generator.generate_multiple_spectrograms(
                base_dir=self.packet_source,
                output_dir=self.output_dir,
                num_spectrograms=1,
                packets_per_spectrogram=packets
            )
            
            if results:
                image, file_path = results[0]
                if image is not None:
                    # Load image fresh from file (like interactive_pipeline.py)
                    self.current_image = cv2.imread(file_path)
                    self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                    self.current_image_path = file_path
                    self.current_detections = []
                    self.manual_boxes = []
                    
                    # Update display
                    self.display_image()
                    self.update_detection_list()
                    self.status_var.set(f"Generated: {os.path.basename(file_path)} - Ready for detection")
                    
                    print(f"Generated spectrogram: {os.path.basename(file_path)}")
                    
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
            self.status_var.set(f"Error generating spectrogram: {e}")
    
    def run_detection(self):
        """Run YOLO detection on current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return
            
        try:
            self.status_var.set("Running detection...")
            self.root.update()
            
            # Run detection using YOLO model directly (like interactive_pipeline.py)
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            
            # Run inference
            results = model(self.current_image, conf=0.3)
            
            # Extract detections
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    detections.append([x1, y1, x2, y2, conf, class_id])
            
            self.current_detections = detections
            
            # Update display
            self.display_image()
            self.update_detection_list()
            self.status_var.set(f"Found {len(detections)} detections")
            
        except Exception as e:
            self.status_var.set(f"Detection error: {str(e)}")
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            print(f"Full error: {e}")  # Debug print
    
    def display_image(self):
        """Display the current image with detections."""
        if self.current_image is None:
            return
            
        self.ax.clear()
        
        # Display RGB image directly (no colormap needed)
        self.ax.imshow(self.current_image)
            
        self.ax.set_title("RF Spectrogram - Click and drag to add manual labels")
        
        # Draw model detections
        for i, detection in enumerate(self.current_detections):
            x1, y1, x2, y2, conf, class_id = detection
            color = self.class_colors[class_id]
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
            self.ax.add_patch(rect)
            
            # Add label
            self.ax.text(x1, y1-5, f"{self.class_names[class_id]} {conf:.2f}", 
                        color=color, fontsize=8, weight='bold')
        
        # Draw manual boxes (outline only, like YOLO labels)
        for box in self.manual_boxes:
            x1, y1, x2, y2, class_id = box
            color = self.class_colors[class_id]
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
            self.ax.add_patch(rect)
            
            # Add label
            self.ax.text(x1, y1-5, f"{self.class_names[class_id]} (Manual)", 
                        color=color, fontsize=8, weight='bold')
        
        # Set proper axis limits
        if self.current_image is not None:
            self.ax.set_xlim(0, self.current_image.shape[1])
            self.ax.set_ylim(self.current_image.shape[0], 0)
        
        self.ax.set_aspect('equal')
        self.canvas.draw()
        
    def update_detection_list(self):
        """Update the detection list."""
        # Clear existing items
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
            
        # Add model detections
        for i, detection in enumerate(self.current_detections):
            x1, y1, x2, y2, conf, class_id = detection
            self.detection_tree.insert('', 'end', values=(
                self.class_names[class_id], f"{conf:.3f}", 
                f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}", "Model"
            ))
            
        # Add manual detections
        for box in self.manual_boxes:
            x1, y1, x2, y2, class_id = box
            self.detection_tree.insert('', 'end', values=(
                self.class_names[class_id], "1.000", 
                f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}", "Manual"
            ))
    
    def on_mouse_press(self, event):
        """Handle mouse press for manual labeling."""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            self.drawing = True
            self.current_box = [event.xdata, event.ydata, event.xdata, event.ydata]
            
    def on_mouse_move(self, event):
        """Handle mouse move for manual labeling."""
        if not self.drawing or self.current_box is None or event.inaxes != self.ax:
            return
            
        self.current_box[2] = event.xdata
        self.current_box[3] = event.ydata
        
        # Redraw with current box
        self.display_image()
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.7)
            self.ax.add_patch(rect)
            self.canvas.draw()
            
    def on_mouse_release(self, event):
        """Handle mouse release for manual labeling."""
        if not self.drawing or self.current_box is None:
            return
            
        if event.inaxes == self.ax:
            # Finalize the box
            x1, y1, x2, y2 = self.current_box
            if abs(x2-x1) > 10 and abs(y2-y1) > 10:  # Minimum size
                class_id = self.class_names.index(self.class_var.get())
                self.manual_boxes.append([x1, y1, x2, y2, class_id])
                self.update_detection_list()
                
        self.drawing = False
        self.current_box = None
        self.display_image()
        
    def approve_all(self):
        """Approve all model detections."""
        self.status_var.set("All model detections approved")
        
    def reject_all(self):
        """Reject all model detections."""
        self.current_detections = []
        self.display_image()
        self.update_detection_list()
        self.status_var.set("All model detections rejected")
        
    def delete_selected_detection(self):
        """Delete selected detection from list."""
        selection = self.detection_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "No detection selected")
            return
            
        item = selection[0]
        values = self.detection_tree.item(item, 'values')
        source = values[6]
        
        if source == "Model":
            # Remove from model detections
            idx = self.detection_tree.index(item)
            if idx < len(self.current_detections):
                del self.current_detections[idx]
                self.status_var.set("Model detection removed")
        else:
            # Remove from manual boxes
            idx = self.detection_tree.index(item) - len(self.current_detections)
            if 0 <= idx < len(self.manual_boxes):
                del self.manual_boxes[idx]
                self.status_var.set("Manual detection removed")
                
        self.display_image()
        self.update_detection_list()
        
    def edit_selected_detection(self):
        """Edit selected detection."""
        selection = self.detection_tree.selection()
        if not selection:
            return
        # TODO: Implement edit functionality
        messagebox.showinfo("Info", "Edit functionality coming soon!")
        
    def clear_manual_boxes(self):
        """Clear all manual boxes."""
        self.manual_boxes = []
        self.display_image()
        self.update_detection_list()
        
    def save_and_next(self):
        """Save current results and prepare for next image."""
        if self.current_image is None:
            return
            
        # Convert numpy arrays to Python lists for JSON serialization
        model_detections_serializable = []
        for detection in self.current_detections:
            x1, y1, x2, y2, conf, class_id = detection
            model_detections_serializable.append([
                float(x1), float(y1), float(x2), float(y2), 
                float(conf), int(class_id)
            ])
            
        manual_detections_serializable = []
        for box in self.manual_boxes:
            x1, y1, x2, y2, class_id = box
            manual_detections_serializable.append([
                float(x1), float(y1), float(x2), float(y2), int(class_id)
            ])
        
        # Save detection results
        results = {
            'image_path': self.current_image_path,
            'model_detections': model_detections_serializable,
            'manual_detections': manual_detections_serializable,
            'timestamp': str(Path(self.current_image_path).stat().st_mtime)
        }
        
        # Save to JSON file
        output_file = os.path.join(self.output_dir, 
                                  f"results_{os.path.basename(self.current_image_path)}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save labeled image
        self.save_labeled_image_silent()
        
        # Save YOLO labels
        self.save_yolo_labels()
        
        # Clear current state
        self.current_detections = []
        self.manual_boxes = []
        self.current_image = None
        
        self.status_var.set("Results saved. Ready for next spectrogram.")
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Ready for next spectrogram", 
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=16)
        self.canvas.draw()
        
        # Clear detection list
        for item in self.detection_tree.get_children():
            self.detection_tree.delete(item)
    
    def save_labeled_image_silent(self):
        """Save the labeled image silently."""
        if self.current_image is None:
            return
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(self.current_image)
            ax.set_title("RF Spectrogram with Detections", fontsize=14)
            
            # Draw model detections
            for detection in self.current_detections:
                x1, y1, x2, y2, conf, class_id = detection
                color = self.class_colors[class_id]
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                
                ax.text(x1, y1-5, f"{self.class_names[class_id]} {conf:.2f}", 
                        color=color, fontsize=10, weight='bold')
            
            # Draw manual boxes
            for box in self.manual_boxes:
                x1, y1, x2, y2, class_id = box
                color = self.class_colors[class_id]
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                ax.add_patch(rect)
                
                ax.text(x1, y1-5, f"{self.class_names[class_id]} (Manual)", 
                        color=color, fontsize=10, weight='bold')
            
            # Set axis properties
            ax.set_xlim(0, self.current_image.shape[1])
            ax.set_ylim(self.current_image.shape[0], 0)
            ax.set_aspect('equal')
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_labeled.png")
            
            # Save the image
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Error saving labeled image silently: {e}")
    
    def save_yolo_labels(self):
        """Save detections in YOLO format (.txt file)."""
        if self.current_image is None:
            return
            
        try:
            # Get image dimensions
            img_height, img_width = self.current_image.shape[:2]
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            yolo_file = os.path.join(self.output_dir, f"{base_name}.txt")
            
            with open(yolo_file, 'w') as f:
                # Write model detections
                for detection in self.current_detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    center_x = (x1 + x2) / 2.0 / img_width
                    center_y = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write line: class_id center_x center_y width height
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # Write manual detections
                for box in self.manual_boxes:
                    x1, y1, x2, y2, class_id = box
                    
                    # Convert to YOLO format
                    center_x = (x1 + x2) / 2.0 / img_width
                    center_y = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write line: class_id center_x center_y width height
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"YOLO labels saved: {yolo_file}")
            
        except Exception as e:
            print(f"Error saving YOLO labels: {e}")
    
    def on_closing(self):
        """Handle window close event."""
        # Stop any running processes
        self.is_generating = False
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live Interactive RF Detection Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--packets", type=str, 
                       default="../spectrogram_training_data_20220711/",
                       help="Path to packet source directory")
    parser.add_argument("--output", type=str, default="live_output", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create and run GUI
    app = LiveInteractivePipeline(args.model, args.packets, args.output)
    app.run()

if __name__ == "__main__":
    main()
