"""
Fix Custom YOLO Implementation
This fixes our custom YOLO by implementing a proper loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProperYOLOLoss(nn.Module):
    """
    Proper YOLO loss function that actually teaches object detection
    """
    
    def __init__(self, num_classes=3, lambda_coord=5.0, lambda_noobj=0.5):
        super(ProperYOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets):
        """
        Compute proper YOLO loss
        predictions: (B, 3, 5 + num_classes) - model outputs
        targets: list of (N, 5) tensors - ground truth boxes
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Initialize loss components
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        
        for b in range(batch_size):
            # Get predictions for this batch
            pred = predictions[b]  # Shape: (3, 5 + num_classes)
            
            # Get targets for this batch
            if b < len(targets) and len(targets[b]) > 0:
                gt_boxes = targets[b]  # Shape: (N, 5)
                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.to(device)
                else:
                    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            else:
                gt_boxes = torch.empty(0, 5, device=device)
            
            # For each anchor
            for anchor_idx in range(3):
                anchor_pred = pred[anchor_idx]  # Shape: (5 + num_classes)
                
                # Extract components
                pred_x = torch.sigmoid(anchor_pred[0])
                pred_y = torch.sigmoid(anchor_pred[1])
                pred_w = anchor_pred[2]
                pred_h = anchor_pred[3]
                pred_conf = torch.sigmoid(anchor_pred[4])
                pred_classes = anchor_pred[5:]
                
                if len(gt_boxes) > 0:
                    # Find best matching ground truth box
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        gt_x, gt_y, gt_w, gt_h, gt_class = gt_box
                        
                        # Calculate IoU
                        iou = self.calculate_iou(pred_x, pred_y, pred_w, pred_h,
                                               gt_x, gt_y, gt_w, gt_h)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # If IoU > 0.5, this anchor is responsible for the object
                    if best_iou > 0.5:
                        gt_box = gt_boxes[best_gt_idx]
                        gt_x, gt_y, gt_w, gt_h, gt_class = gt_box
                        
                        # Coordinate loss (only for responsible anchors)
                        coord_loss += F.mse_loss(pred_x, gt_x)
                        coord_loss += F.mse_loss(pred_y, gt_y)
                        coord_loss += F.mse_loss(pred_w, gt_w)
                        coord_loss += F.mse_loss(pred_h, gt_h)
                        
                        # Confidence loss (should be 1 for responsible anchors)
                        conf_loss += F.binary_cross_entropy(pred_conf, torch.tensor(1.0, device=device))
                        
                        # Classification loss
                        gt_class_one_hot = F.one_hot(gt_class.long(), num_classes=self.num_classes).float()
                        class_loss += F.cross_entropy(pred_classes.unsqueeze(0), gt_class.long().unsqueeze(0))
                    else:
                        # No object - confidence should be 0
                        conf_loss += F.binary_cross_entropy(pred_conf, torch.tensor(0.0, device=device))
                else:
                    # No ground truth boxes - confidence should be 0
                    conf_loss += F.binary_cross_entropy(pred_conf, torch.tensor(0.0, device=device))
        
        # Combine losses with proper weights
        total_loss = (
            self.lambda_coord * coord_loss +
            1.0 * conf_loss +
            1.0 * class_loss
        )
        
        return total_loss / batch_size
    
    def calculate_iou(self, pred_x, pred_y, pred_w, pred_h, gt_x, gt_y, gt_w, gt_h):
        """Calculate Intersection over Union"""
        # Convert to corner format
        pred_x1 = pred_x - pred_w / 2
        pred_y1 = pred_y - pred_h / 2
        pred_x2 = pred_x + pred_w / 2
        pred_y2 = pred_y + pred_h / 2
        
        gt_x1 = gt_x - gt_w / 2
        gt_y1 = gt_y - gt_h / 2
        gt_x2 = gt_x + gt_w / 2
        gt_y2 = gt_y + gt_h / 2
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        return iou

def fix_training_script():
    """Update the training script to use proper YOLO loss"""
    print("=== Fixing Training Script ===")
    
    # Read current training script
    with open('train_yolo.py', 'r') as f:
        content = f.read()
    
    # Replace the loss function
    old_loss = "self.criterion = YOLOLoss(num_classes=3)"
    new_loss = "self.criterion = ProperYOLOLoss(num_classes=3)"
    
    if old_loss in content:
        content = content.replace(old_loss, new_loss)
        print("Updated loss function in training script")
    else:
        print("Loss function not found in training script")
    
    # Add import for ProperYOLOLoss
    if "from fix_custom_yolo import ProperYOLOLoss" not in content:
        content = content.replace("from yolo_model import create_yolo_model, count_parameters", 
                                "from yolo_model import create_yolo_model, count_parameters\nfrom fix_custom_yolo import ProperYOLOLoss")
        print("Added import for ProperYOLOLoss")
    
    # Write updated script
    with open('train_yolo_fixed.py', 'w') as f:
        f.write(content)
    
    print("Created train_yolo_fixed.py with proper loss function")

def main():
    """Main function to fix custom YOLO"""
    print("=== Fixing Custom YOLO Implementation ===")
    
    # Create the proper loss function
    print("1. Created ProperYOLOLoss class")
    
    # Fix the training script
    fix_training_script()
    
    print("\n=== Next Steps ===")
    print("1. Run: python3 train_yolo_fixed.py")
    print("2. This will use the proper YOLO loss function")
    print("3. The model should actually learn to detect objects")
    print("4. Monitor training to ensure loss decreases properly")

if __name__ == "__main__":
    main()
