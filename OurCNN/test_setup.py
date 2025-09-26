"""
Test script to verify CNN setup
Tests data loading, model creation, and basic functionality
"""

import torch
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ Torchvision import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✓ PIL {PIL.__version__}")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from data_loader import analyze_dataset, create_data_loaders
        
        # Test dataset analysis
        data_dir = "/home/gabriel/Desktop/HackathonNATO_Drones_2025"
        total_images, labeled_images = analyze_dataset(data_dir)
        
        if total_images > 0:
            print(f"✓ Found {total_images} images, {labeled_images} with labels")
        else:
            print("✗ No images found in dataset")
            return False
        
        # Test data loader creation (small batch for testing)
        train_loader, val_loader, class_names = create_data_loaders(
            data_dir, batch_size=4, test_size=0.2, task='classification'
        )
        
        print(f"✓ Created data loaders: {len(train_loader)} train, {len(val_loader)} val batches")
        print(f"✓ Classes: {class_names}")
        
        # Test loading a batch
        for images, labels in train_loader:
            print(f"✓ Batch shape: {images.shape}, labels: {labels.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        from cnn_model import create_model, count_parameters
        
        # Test custom model only
        print(f"\nCustom SpectrogramCNN Model:")
        model, model_name = create_model(num_classes=4)
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"✓ {model_name}: {num_params:,} parameters")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        with torch.no_grad():
            output = model(x)
        
        print(f"✓ Forward pass: {x.shape} -> {output.shape}")
        
        # Test optimizer
        from cnn_model import get_optimizer, get_scheduler
        optimizer = get_optimizer(model, optimizer_type='adam', lr=0.001)
        scheduler = get_scheduler(optimizer, scheduler_type='step')
        print(f"✓ Optimizer: {type(optimizer).__name__}")
        print(f"✓ Scheduler: {type(scheduler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_device():
    """Test CUDA availability"""
    print("\nTesting device availability...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    return True

def test_training_setup():
    """Test training setup"""
    print("\nTesting training setup...")
    
    try:
        from train_cnn import CNNTrainer
        from data_loader import create_data_loaders
        from cnn_model import create_model
        
        # Create small test setup
        data_dir = "/home/gabriel/Desktop/HackathonNATO_Drones_2025"
        train_loader, val_loader, class_names = create_data_loaders(
            data_dir, batch_size=4, test_size=0.2, task='classification'
        )
        
        model, model_name = create_model(num_classes=len(class_names))
        
        # Test trainer creation
        trainer = CNNTrainer(model, train_loader, val_loader, class_names, device='cpu')
        print(f"✓ Trainer created successfully")
        
        # Test one training step
        trainer.model.train()
        for images, labels in train_loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = trainer.model(images)
            loss = trainer.criterion(outputs, labels)
            print(f"✓ Training step successful, loss: {loss.item():.4f}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("NATO Hackathon Custom CNN Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Device Check", test_device),
        ("Training Setup", test_training_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Run: python train_cnn.py --epochs 10")
        print("2. Check training progress and results")
        print("3. Monitor accuracy and loss curves")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Make sure all packages are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()