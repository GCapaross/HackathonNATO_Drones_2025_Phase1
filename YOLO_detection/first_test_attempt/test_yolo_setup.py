"""
Test script to verify YOLO setup
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_data_loader import create_yolo_data_loaders, analyze_dataset
from yolo_model import create_yolo_model, count_parameters

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    try:
        data_dir = "../spectrogram_training_data_20220711"
        
        # Analyze dataset
        stats = analyze_dataset(data_dir)
        print(f"✓ Dataset analysis completed")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Images with labels: {stats['images_with_labels']}")
        print(f"  Total labels: {stats['total_labels']}")
        
        # Create data loaders
        train_loader, val_loader, class_names = create_yolo_data_loaders(
            data_dir, batch_size=4, test_size=0.2, img_size=640
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  Classes: {class_names}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    try:
        # Create model
        model = create_yolo_model(num_classes=4, img_size=640)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {count_parameters(model):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 640, 640)
        print(f"  Input shape: {x.shape}")
        
        with torch.no_grad():
            outputs = model(x)
        
        print(f"  Number of output scales: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Scale {i} output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")
    try:
        # Create model
        model = create_yolo_model(num_classes=4, img_size=640)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Test training step
        x = torch.randn(2, 3, 640, 640)
        
        optimizer.zero_grad()
        outputs = model(x)
        
        # Create dummy target with correct shape
        # Our model outputs shape: (B, 3, 5 + num_classes) = (2, 3, 9)
        target = torch.zeros_like(outputs)
        
        # Simplified loss computation
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training setup successful")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training setup error: {e}")
        return False

def test_device():
    """Test device availability"""
    print("\nTesting device...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device: {device}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        
        return True
    except Exception as e:
        print(f"✗ Device test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== YOLO Setup Test ===")
    
    tests = [
        ("Imports", test_imports),
        ("Device", test_device),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Training Setup", test_training_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! YOLO setup is ready.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
