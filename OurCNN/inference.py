"""
Real-time inference script for trained CNN model
Load a trained model and classify new spectrogram images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import numpy as np

# Import our modules
from cnn_model import create_model

class SpectrogramClassifier:
    def __init__(self, model_path, class_names=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default class names
        if class_names is None:
            self.class_names = ['Background', 'WLAN', 'BT_classic', 'BLE']
        else:
            self.class_names = class_names
        
        # Load model
        self.model = create_model(num_classes=len(self.class_names))[0]
        
        # Load checkpoint (handle both model-only and full checkpoint formats)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Model-only format
            self.model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")

    def predict(self, image_path):
        """Predict class for a single image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, 1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

    def predict_batch(self, image_paths):
        """Predict classes for multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                pred_class, confidence, probs = self.predict(image_path)
                results.append({
                    'image_path': image_path,
                    'predicted_class': self.class_names[pred_class],
                    'class_id': pred_class,
                    'confidence': confidence,
                    'all_probabilities': dict(zip(self.class_names, probs))
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

    def classify_and_display(self, image_path):
        """Classify image and display results nicely"""
        print(f"\n=== Classifying: {os.path.basename(image_path)} ===")
        
        pred_class, confidence, probs = self.predict(image_path)
        
        print(f"Predicted Class: {self.class_names[pred_class]} (ID: {pred_class})")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        
        print("\nAll Class Probabilities:")
        for class_name, prob in zip(self.class_names, probs):
            bar_length = int(prob * 50)  # Scale to 50 chars
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_name:<15}: {prob:.4f} |{bar}| {prob*100:.1f}%")
        
        return pred_class, confidence, probs

def main():
    parser = argparse.ArgumentParser(description='Classify spectrogram images with trained CNN')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to spectrogram image to classify')
    parser.add_argument('--batch', type=str, nargs='+',
                        help='Multiple image paths for batch classification')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Make sure you've trained a model first with: python train_cnn.py")
        return
    
    # Create classifier
    classifier = SpectrogramClassifier(args.model_path)
    
    if args.batch:
        # Batch classification
        print("=== Batch Classification ===")
        results = classifier.predict_batch(args.batch)
        
        print(f"\nProcessed {len(results)} images:")
        for result in results:
            if 'error' in result:
                print(f"  {result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"  {result['image_path']}: {result['predicted_class']} "
                      f"(confidence: {result['confidence']:.3f})")
    
    else:
        # Single image classification
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found!")
            return
        
        classifier.classify_and_display(args.image)

if __name__ == "__main__":
    main()
