# NATO Hackathon CNN Project

## Overview
This project implements a CNN-based approach for drone detection and classification using RF spectrograms. The system can classify different types of RF signals (WLAN, Bluetooth, BLE) and detect their presence in spectrogram images.

## Project Structure
```
OurCNN/
├── data_loader.py          # Dataset loading and preprocessing
├── cnn_model.py           # CNN model architectures
├── train_cnn.py           # Training script
├── test_setup.py          # Setup verification
├── requirements.txt       # Python dependencies
├── workflow.txt          # Complete workflow explanation
├── Notes.txt             # Strategic notes
├── Notes2.txt            # Dataset analysis
└── README_CNN.md         # This file
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv nato_cnn_env # or just venv venv 
source nato_cnn_env/bin/activate  # Linux
# or nato_cnn_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Setup
```bash
# Verify everything is working
python test_setup.py
```

### 3. Train Model
```bash
# Train custom CNN model
python train_cnn.py --epochs 50 --batch_size 32
```

## CNN Architecture - Detailed Explanation

### Problem Understanding
Our CNN solves a **multi-class classification problem** for drone detection:
1. **Binary Detection**: "Is there a drone present?" (Yes/No)
2. **Multi-class Classification**: "What type of signal is this?" (Background, WLAN, Bluetooth, BLE)
3. **Signal Analysis**: Analyze RF spectrograms to identify drone communication patterns

### Custom SpectrogramCNN Architecture

#### **Why This Architecture?**
- **Spectrograms are 2D images** showing frequency (y-axis) vs time (x-axis)
- **RF signals have specific patterns** that CNNs can learn to recognize
- **Custom design** optimized for spectrogram characteristics (1024x192 → 224x224)

#### **Layer-by-Layer Breakdown:**

```
Input: 224x224x3 (RGB spectrogram image)
    ↓
Conv1: 7x7 kernel, stride=2, 32 filters
    ↓ (Why 7x7? Large receptive field to capture signal patterns)
BatchNorm1 + ReLU
    ↓ (Why ReLU? Prevents vanishing gradients, faster training)
MaxPool2d: 3x3 kernel, stride=2
    ↓ (Why MaxPool? Reduces spatial dimensions, keeps important features)
Conv2: 3x3 kernel, 64 filters
    ↓ (3x3 is standard for feature extraction)
BatchNorm2 + ReLU
    ↓
Conv3: 3x3 kernel, stride=2, 128 filters
    ↓ (Increasing filters to capture complex patterns)
BatchNorm3 + ReLU
    ↓
Conv4: 3x3 kernel, stride=2, 256 filters
    ↓ (Final feature extraction layer)
BatchNorm4 + ReLU
    ↓
Global Average Pooling
    ↓ (Why GAP? Reduces overfitting, better than FC layers)
Dropout(0.5) + FC(256→128) + ReLU
    ↓ (Why Dropout? Prevents overfitting)
Dropout(0.3) + FC(128→4)
    ↓
Output: 4 classes (Background, WLAN, Bluetooth, BLE)
```

#### **Key Design Choices Explained:**

**1. ReLU Activation Function:**
- **What it does**: ReLU(x) = max(0, x) - sets negative values to 0
- **Why ReLU**: 
  - Prevents vanishing gradient problem
  - Faster computation than sigmoid/tanh
  - Sparse activation (many neurons output 0)
  - Works well with spectrograms (positive values)

**2. MaxPool2d:**
- **What it does**: Takes maximum value in each 3x3 region
- **Why MaxPool**:
  - Reduces spatial dimensions (224→112→56→28→14)
  - Keeps most important features
  - Reduces computational load
  - Translation invariance (signal position doesn't matter)

**3. Batch Normalization:**
- **What it does**: Normalizes inputs to each layer
- **Why BatchNorm**:
  - Faster training convergence
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Acts as regularization

**4. Global Average Pooling:**
- **What it does**: Averages all spatial locations
- **Why GAP**:
  - Reduces overfitting (fewer parameters)
  - Better than fully connected layers
  - More robust to spatial translations
  - Standard in modern CNNs

### Why Custom CNN for Spectrograms?

#### **Optimized for RF Signals:**
- **Designed specifically** for spectrogram characteristics
- **7x7 first convolution** captures large signal patterns
- **Progressive feature extraction** from simple to complex
- **Global average pooling** reduces overfitting
- **Efficient architecture** for real-time processing

### Optimizers Explained

#### **Adam Optimizer:**
- **What it does**: Adaptive learning rate for each parameter
- **Why Adam**:
  - Combines momentum + adaptive learning rates
  - Works well with sparse gradients
  - Good default choice
  - Less sensitive to hyperparameters

#### **SGD Optimizer:**
- **What it does**: Stochastic Gradient Descent with momentum
- **Why SGD**:
  - Simpler, more interpretable
  - Can achieve better final performance
  - Requires more tuning
  - Good for fine-tuning

### Learning Rate Schedulers

#### **StepLR Scheduler:**
- **What it does**: Reduces learning rate by factor every N epochs
- **Why StepLR**:
  - Simple and effective
  - Helps convergence in later epochs
  - Prevents oscillation around minimum
  - Standard practice

#### **CosineAnnealingLR Scheduler:**
- **What it does**: Learning rate follows cosine curve
- **Why CosineAnnealing**:
  - Smoother learning rate decay
  - Better for fine-tuning
  - Can escape local minima
  - More sophisticated approach

### Multi-class Classification Strategy

#### **Problem Decomposition:**
1. **Binary Detection**: "Signal present?" (Background vs Signal)
2. **Signal Classification**: "What type?" (WLAN vs Bluetooth vs BLE)
3. **Confidence Scoring**: How certain is the model?

#### **Loss Function: Cross-Entropy**
- **What it does**: Measures difference between predicted and true probabilities
- **Why Cross-Entropy**:
  - Perfect for multi-class problems
  - Penalizes confident wrong predictions
  - Works well with softmax output
  - Standard choice for classification

#### **Output Interpretation:**
```
Model Output: [0.1, 0.7, 0.15, 0.05]  # Probabilities for 4 classes
Prediction: Class 1 (WLAN) with 70% confidence
```

### Why This Architecture Works for Drone Detection

#### **Spectrogram Characteristics:**
- **Time-frequency representation** of RF signals
- **Drone signals** have specific patterns (frequency hopping, modulation)
- **Different protocols** (WLAN, Bluetooth, BLE) have distinct signatures
- **CNN excels** at learning these spatial patterns

#### **Real-world Application:**
1. **Raw RF signal** → **Spectrogram generation** → **CNN classification**
2. **Model outputs** → **Post-processing** → **Drone detection decision**
3. **Confidence thresholds** → **False positive reduction**

### Custom CNN Advantages

#### **Why This Architecture Works:**
- **Optimized for spectrograms** (not general images)
- **Efficient training** with fewer parameters
- **Fast inference** for real-time applications
- **Easy to understand** and modify
- **Good baseline** performance for drone detection

## Dataset
- **Source**: spectrogram_training_data_20220711/results/
- **Images**: 40,000+ spectrogram images (1024x192 pixels)
- **Labels**: YOLO format annotations
- **Classes**: 4 classes (Background, WLAN, Bluetooth, BLE)
- **Split**: 80% training, 20% validation

## Training Process

### 1. Data Loading
- Load spectrogram images and YOLO labels
- Resize images to 224x224 for CNN input
- Apply data augmentation (rotation, noise, etc.)
- Create train/validation splits

### 2. Model Training
- **Loss**: Cross-entropy loss for classification
- **Optimizer**: Adam optimizer with learning rate scheduling
- **Metrics**: Accuracy, precision, recall, F1-score
- **Validation**: Monitor validation accuracy to prevent overfitting

### 3. Model Evaluation
- Classification report with per-class metrics
- Confusion matrix visualization
- Training history plots
- Model checkpointing

## Usage Examples

### Basic Training
```python
from data_loader import create_data_loaders
from cnn_model import create_model
from train_cnn import CNNTrainer

# Create data loaders
train_loader, val_loader, class_names = create_data_loaders(
    data_dir="/path/to/dataset", 
    batch_size=32, 
    task='classification'
)

# Create model
model, model_name = create_model(model_type='custom', num_classes=4)

# Train model
trainer = CNNTrainer(model, train_loader, val_loader, class_names)
best_acc, best_epoch = trainer.train(num_epochs=50)
```

### Model Inference
```python
import torch
from cnn_model import create_model

# Load trained model
model, _ = create_model(model_type='custom', num_classes=4)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    output = model(spectrogram_image)
    predicted_class = torch.argmax(output, dim=1)
```

## Results and Metrics

### Training Metrics
- **Accuracy**: Model performance on validation set
- **Loss**: Training and validation loss curves
- **Confusion Matrix**: Per-class performance analysis
- **Classification Report**: Precision, recall, F1-score

### Model Performance
- **Custom CNN**: Baseline model for spectrogram classification
- **ResNet**: Transfer learning approach with pre-trained weights
- **EfficientNet**: Advanced architecture for better performance

## Next Steps

### 1. Model Optimization
- Experiment with different architectures
- Hyperparameter tuning
- Data augmentation strategies
- Ensemble methods

### 2. Real-time Implementation
- Signal-to-spectrogram conversion
- Model optimization for inference
- Integration with YOLO for object detection
- Real-time processing pipeline

### 3. Deployment
- Model quantization for edge devices
- API development for real-time classification
- Integration with drone detection systems
- Performance monitoring and logging

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size
2. **Data loading errors**: Check dataset paths
3. **Model convergence**: Adjust learning rate
4. **Overfitting**: Increase regularization

### Performance Tips
1. Use GPU for training (CUDA)
2. Increase batch size if memory allows
3. Use data augmentation for better generalization
4. Monitor validation metrics during training

## Contributing
This project is part of the NATO Hackathon for AI-based drone detection. Contributions and improvements are welcome!

## License
NATO Hackathon Project - Educational and Research Use
