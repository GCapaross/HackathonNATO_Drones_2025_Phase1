"""
CNN Model for Spectrogram Classification
NATO Hackathon - Drone Detection and Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np

class SpectrogramCNN(nn.Module):
    """
    Custom CNN for spectrogram classification
    Optimized for RF signal spectrograms (1024x192 -> 224x224)
    """
    
    def __init__(self, num_classes=4, input_channels=3):
        super(SpectrogramCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual-like blocks
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Removed ResNet and EfficientNet models - focusing on custom CNN only

def create_model(num_classes=4):
    """
    Create custom CNN model for spectrogram classification
    
    Args:
        num_classes: Number of output classes (default: 4)
    
    Returns:
        model, model_name
    """
    
    # Using only custom CNN architecture
    model = SpectrogramCNN(num_classes=num_classes)
    model_name = 'SpectrogramCNN'
    print("Using Custom SpectrogramCNN - optimized for RF spectrograms")
    
    return model, model_name

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer(model, optimizer_type='adam', lr=0.001, weight_decay=1e-4):
    """
    Create optimizer for model
    
    Args:
        model: PyTorch model
        optimizer_type: 'adam' or 'sgd'
        lr: Learning rate
        weight_decay: Weight decay for regularization
    
    Returns:
        optimizer
    """
    
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def get_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'step' or 'cosine'
        **kwargs: Additional scheduler parameters
    
    Returns:
        scheduler
    """
    
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

if __name__ == "__main__":
    # Test model creation
    print("Testing Custom CNN model creation...")
    
    # Test custom model
    print("\nCustom SpectrogramCNN Model:")
    model, model_name = create_model(num_classes=4)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"  Model: {model_name}")
    print(f"  Parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    with torch.no_grad():
        output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test optimizer
    optimizer = get_optimizer(model, optimizer_type='adam', lr=0.001)
    scheduler = get_scheduler(optimizer, scheduler_type='step')
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Scheduler: {type(scheduler).__name__}")
    
    print("\nCustom CNN model test completed!")
