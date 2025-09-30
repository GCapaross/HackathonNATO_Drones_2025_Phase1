"""
Real-time YOLO RF Frame Detector
================================
Detects RF frames in real-time spectrograms using trained YOLO model.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from typing import List, Tuple, Dict, Optional
import threading
import queue
from dataclasses import dataclass

@dataclass
class Detection:
    """Represents a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_center, y_center, width, height
    x1: float
    y1: float
    x2: float
    y2: float

class RealtimeYOLODetector:
    """
    Real-time YOLO detector for RF frame detection in spectrograms.
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.5,
                 class_names: List[str] = None):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            class_names: List of class names
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Default class names (matching training data)
        self.class_names = class_names or ['WLAN', 'collision', 'bluetooth']
        self.class_colors = {
            0: (255, 0, 0),      # WLAN - Red
            1: (0, 255, 0),      # Collision - Green  
            2: (0, 0, 255)       # Bluetooth - Blue
        }
        
        # Load model
        self.model = None
        self.load_model()
        
        # For real-time processing
        self.detection_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect_in_image(self, image: np.ndarray) -> List[Detection]:
        """
        Detect RF frames in a single spectrogram image.
        
        Args:
            image: Input spectrogram image (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, 
                               iou=self.nms_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box
                        
                        # Convert to center format
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                            confidence=float(conf),
                            bbox=(x_center, y_center, width, height),
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2)
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes on the image.
        
        Args:
            image: Input image
            detections: List of detections to draw
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            # Get color for this class
            color = self.class_colors.get(detection.class_id, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (int(detection.x1), int(detection.y1)),
                         (int(detection.x2), int(detection.y2)),
                         color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(result_image,
                         (int(detection.x1), int(detection.y1) - label_size[1] - 10),
                         (int(detection.x1) + label_size[0], int(detection.y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(result_image, label,
                       (int(detection.x1), int(detection.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image
    
    def process_spectrogram(self, image: np.ndarray, 
                           save_result: bool = True,
                           output_dir: str = "detection_results") -> Tuple[np.ndarray, List[Detection], str]:
        """
        Process a single spectrogram and return results.
        
        Args:
            image: Input spectrogram image
            save_result: Whether to save the result image
            output_dir: Directory to save results
            
        Returns:
            Tuple of (result_image, detections, output_path)
        """
        # Run detection
        detections = self.detect_in_image(image)
        
        # Draw detections
        result_image = self.draw_detections(image, detections)
        
        output_path = None
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detection_{timestamp}.png")
            cv2.imwrite(output_path, result_image)
        
        return result_image, detections, output_path
    
    def start_realtime_detection(self, spectrogram_queue: queue.Queue,
                               output_dir: str = "detection_results"):
        """
        Start real-time detection processing.
        
        Args:
            spectrogram_queue: Queue of spectrograms to process
            output_dir: Directory to save detection results
        """
        self.running = True
        os.makedirs(output_dir, exist_ok=True)
        
        print("Starting real-time YOLO detection...")
        
        while self.running:
            try:
                # Get next spectrogram from queue
                image, spectrogram_path = spectrogram_queue.get(timeout=1.0)
                
                if image is not None:
                    # Process the spectrogram
                    result_image, detections, output_path = self.process_spectrogram(
                        image, save_result=True, output_dir=output_dir
                    )
                    
                    # Print detection results
                    if detections:
                        print(f"Detected {len(detections)} RF frames:")
                        for det in detections:
                            print(f"  - {det.class_name}: {det.confidence:.3f}")
                    else:
                        print("No RF frames detected")
                    
                    # Add to detection queue for GUI
                    if not self.detection_queue.full():
                        self.detection_queue.put((result_image, detections, output_path))
                    else:
                        print("Warning: Detection queue full, dropping result")
                
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("Stopping real-time detection...")
                self.running = False
            except Exception as e:
                print(f"Error in real-time detection: {e}")
                time.sleep(0.1)
    
    def stop_realtime_detection(self):
        """Stop real-time detection."""
        self.running = False
    
    def get_next_detection(self) -> Tuple[np.ndarray, List[Detection], str]:
        """
        Get the next detection result from the queue.
        
        Returns:
            Tuple of (result_image, detections, output_path) or (None, [], None) if queue empty
        """
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None, [], None

# Example usage and testing
if __name__ == "__main__":
    # Test the detector
    model_path = "yolo_training_improved/rf_spectrogram_detection_improved2/weights/best.pt"
    
    if os.path.exists(model_path):
        detector = RealtimeYOLODetector(
            model_path=model_path,
            confidence_threshold=0.3
        )
        
        # Test with a sample image
        test_image_path = "test_spectrogram.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            result_image, detections, output_path = detector.process_spectrogram(image)
            
            print(f"Processed image: {test_image_path}")
            print(f"Detections: {len(detections)}")
            for det in detections:
                print(f"  - {det.class_name}: {det.confidence:.3f}")
            
            if output_path:
                print(f"Result saved to: {output_path}")
        else:
            print(f"Test image {test_image_path} not found")
    else:
        print(f"Model {model_path} not found")
        print("Please train a model first or update the model path")
