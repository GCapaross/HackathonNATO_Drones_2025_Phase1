"""
YOLO Model for RF Signal Detection
Simplified YOLO implementation for spectrogram object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class SimpleYOLO(nn.Module):
    """
    Simplified YOLO model for RF signal detection
    """
    
    def __init__(self, num_classes=4, img_size=640):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Backbone - simplified CNN
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, (5 + num_classes) * 3),  # 3 anchors, 5 + num_classes per anchor
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Detection outputs
        """
        # Backbone
        features = self.backbone(x)
        
        # Detection head
        output = self.detection_head(features)
        
        # Reshape to (batch_size, 3, 5 + num_classes)
        batch_size = x.size(0)
        output = output.view(batch_size, 3, 5 + self.num_classes)
        
        return output
    
    def predict(self, x, conf_threshold=0.5, nms_threshold=0.4):
        """
        Make predictions with confidence and NMS filtering
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Extract predictions
            batch_size = outputs.size(0)
            predictions = []
            
            for b in range(batch_size):
                batch_outputs = outputs[b]  # Shape: (3, 5 + num_classes)
                
                # Apply confidence threshold
                conf_scores = torch.sigmoid(batch_outputs[:, 4])  # Confidence scores
                conf_mask = conf_scores > conf_threshold
                
                if conf_mask.sum() > 0:
                    filtered_outputs = batch_outputs[conf_mask]
                    filtered_conf = conf_scores[conf_mask]
                    
                    # Get class predictions
                    class_scores = torch.softmax(filtered_outputs[:, 5:], dim=1)
                    class_conf, class_pred = torch.max(class_scores, dim=1)
                    
                    # Final confidence
                    final_conf = filtered_conf * class_conf
                    
                    # Extract bounding boxes (normalized coordinates)
                    boxes = filtered_outputs[:, :4]  # x_center, y_center, width, height
                    
                    # Apply NMS
                    keep_indices = self._apply_nms(boxes, final_conf, nms_threshold)
                    
                    if len(keep_indices) > 0:
                        final_boxes = boxes[keep_indices]
                        final_conf = final_conf[keep_indices]
                        final_classes = class_pred[keep_indices]
                        
                        predictions.append({
                            'boxes': final_boxes,
                            'confidences': final_conf,
                            'classes': final_classes
                        })
                    else:
                        predictions.append({
                            'boxes': torch.empty(0, 4),
                            'confidences': torch.empty(0),
                            'classes': torch.empty(0, dtype=torch.long)
                        })
                else:
                    predictions.append({
                        'boxes': torch.empty(0, 4),
                        'confidences': torch.empty(0),
                        'classes': torch.empty(0, dtype=torch.long)
                    })
            
            return predictions
    
    def _apply_nms(self, boxes, scores, nms_threshold):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Convert to corner format for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        _, indices = torch.sort(scores, descending=True)
        
        keep = []
        while len(indices) > 0:
            # Pick the box with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            others = indices[1:]
            
            # Calculate intersection
            xx1 = torch.max(x1[current], x1[others])
            yy1 = torch.max(y1[current], y1[others])
            xx2 = torch.min(x2[current], x2[others])
            yy2 = torch.min(y2[current], y2[others])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # Calculate union
            union = areas[current] + areas[others] - intersection
            
            # Calculate IoU
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            indices = others[iou <= nms_threshold]
        
        return torch.tensor(keep, dtype=torch.long)


def create_yolo_model(num_classes=4, img_size=640, model_type='simple'):
    """
    Create YOLO model for RF signal detection
    
    Args:
        num_classes: Number of classes (Background, WLAN, Bluetooth, BLE)
        img_size: Input image size
        model_type: Type of model ('simple' or 'ultralytics')
    
    Returns:
        YOLO model instance
    """
    if model_type == 'simple':
        model = SimpleYOLO(num_classes=num_classes, img_size=img_size)
    elif model_type == 'ultralytics':
        # Use pre-trained YOLOv8
        model = YOLO('yolov8n.pt')  # nano version for faster training
        # Modify for our classes
        model.model[-1].nc = num_classes
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("=== YOLO Model Test ===")
    
    # Test simple model
    model = create_yolo_model(num_classes=4, model_type='simple')
    print(f"Simple YOLO parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
    
    print("✓ Model test completed")