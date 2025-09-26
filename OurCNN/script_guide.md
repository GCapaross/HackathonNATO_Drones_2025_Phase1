# CNN Scripts Guide - When to Use What

## Overview
This guide explains all the Python scripts in the OurCNN folder and when to use each one.

---

## **1. Setup & Environment**

### `test_setup.py`
**Purpose**: Verify your environment works before training
**When to use**: Before any training, when you get errors, or when setting up on a new machine

```bash
python test_setup.py
```

**What it does:**
- Tests if all packages are installed correctly
- Verifies data can be loaded without errors
- Checks if model can be created and run forward pass
- Confirms GPU/CPU setup works

**Output**: Pass/Fail for each component

---

## **2. Training Scripts**

### `train_cnn.py` (Standard Training)
**Purpose**: Train CNN with single train/validation split
**When to use**: Quick training, initial experiments, when you want fast results

```bash
python train_cnn.py --epochs 50 --batch_size 32
```

**What it does:**
- Splits data: 80% training, 20% validation
- Trains model for specified epochs
- Shows real-time training and validation accuracy
- Saves best model as `best_model.pth`
- Generates training plots

**Best for**: Quick training, initial experiments

### `train_cnn_kfold.py` (Robust Training)
**Purpose**: Train CNN with k-fold cross-validation (more robust)
**When to use**: Final model training, when you need reliable performance estimates

```bash
python train_cnn_kfold.py --num_folds 5 --epochs 50
```

**What it does:**
- Splits data into 5 folds (5 different train/validation splits)
- Trains 5 models, one for each fold
- Shows mean ± std accuracy across folds
- Saves best model as `best_model_kfold.pth`
- Generates comprehensive k-fold plots

**Best for**: Final model training, research papers, production models

---

## **3. Testing & Evaluation Scripts**

### `test_model.py` (Comprehensive Testing)
**Purpose**: Detailed evaluation of trained model
**When to use**: After training, to understand model performance

```bash
# Test on validation set
python test_model.py --model_path best_model.pth

# Test random samples
python test_model.py --model_path best_model.pth --random_samples 20

# Test single image
python test_model.py --model_path best_model.pth --test_single path/to/image.png
```

**What it does:**
- Tests model on validation set
- Generates confusion matrix plot
- Shows per-class accuracy plot
- Detailed classification report (precision, recall, F1-score)
- Tests random samples for quick verification
- Analyzes prediction patterns and confidence

**Best for**: Understanding model performance, debugging, research analysis

### `inference.py` (Real-world Usage)
**Purpose**: Use trained model for actual predictions
**When to use**: In production, for real-time classification, testing new data

```bash
# Classify single image
python inference.py --model_path best_model.pth --image path/to/spectrogram.png

# Classify multiple images
python inference.py --model_path best_model.pth --batch image1.png image2.png image3.png
```

**What it does:**
- Loads trained model
- Classifies single or multiple images
- Shows class probabilities with visual bars
- Fast, optimized for real-time use
- Perfect for deployment

**Best for**: Real-world applications, live drone detection, batch processing

---

## **4. Core Modules (Not run directly)**

### `data_loader.py`
**Purpose**: Handles loading spectrogram images and labels
**Contains**: `SpectrogramDataset` class, data loading functions
**Used by**: All training and testing scripts

### `cnn_model.py`
**Purpose**: Defines the CNN architecture
**Contains**: `SpectrogramCNN` class, model creation functions
**Used by**: All training and testing scripts

---

## **Complete Workflow Examples**

### **Quick Development Workflow**
```bash
# 1. Test setup
python test_setup.py

# 2. Quick training
python train_cnn.py --epochs 30

# 3. Test the model
python test_model.py --model_path best_model.pth

# 4. Use for inference
python inference.py --model_path best_model.pth --image test_spectrogram.png
```

### **Production-Ready Workflow**
```bash
# 1. Test setup
python test_setup.py

# 2. Robust training with k-fold
python train_cnn_kfold.py --num_folds 5 --epochs 50

# 3. Comprehensive evaluation
python test_model.py --model_path best_model_kfold.pth

# 4. Deploy for real-time use
python inference.py --model_path best_model_kfold.pth --image live_spectrogram.png
```

### **Research/Paper Workflow**
```bash
# 1. Test setup
python test_setup.py

# 2. Multiple k-fold experiments
python train_cnn_kfold.py --num_folds 5 --epochs 100

# 3. Detailed analysis
python test_model.py --model_path best_model_kfold.pth --random_samples 100

# 4. Generate plots and metrics for paper
```

---

## **Script Comparison Table**

| Script | Speed | Robustness | Use Case | Output |
|--------|-------|------------|----------|---------|
| `test_setup.py` | Fast | N/A | Setup verification | Pass/Fail |
| `train_cnn.py` | Fast | Medium | Quick training | `best_model.pth` |
| `train_cnn_kfold.py` | Slow | High | Production training | `best_model_kfold.pth` |
| `test_model.py` | Medium | High | Model evaluation | Plots + metrics |
| `inference.py` | Fast | N/A | Real-world usage | Predictions |

---

## **When to Use Each Script**

### **Development Phase**
1. `test_setup.py` - Verify environment
2. `train_cnn.py` - Quick training experiments
3. `test_model.py` - Evaluate results
4. `inference.py` - Test on new data

### **Production Phase**
1. `test_setup.py` - Verify environment
2. `train_cnn_kfold.py` - Robust training
3. `test_model.py` - Comprehensive evaluation
4. `inference.py` - Deploy for real-time use

### **Research Phase**
1. `test_setup.py` - Verify environment
2. `train_cnn_kfold.py` - Multiple experiments
3. `test_model.py` - Detailed analysis
4. Generate plots and metrics

---

## **Common Commands Reference**

```bash
# Setup verification
python test_setup.py

# Quick training (30 minutes)
python train_cnn.py --epochs 50

# Robust training (2 hours)
python train_cnn_kfold.py --num_folds 5 --epochs 50

# Model evaluation
python test_model.py --model_path best_model.pth

# Real-time inference
python inference.py --model_path best_model.pth --image spectrogram.png

# Batch inference
python inference.py --model_path best_model.pth --batch *.png
```

---

## **Troubleshooting**

### **If `test_setup.py` fails:**
- Check if all packages are installed: `pip install -r requirements.txt`
- Verify data directory path is correct
- Check GPU/CPU availability

### **If training fails:**
- Run `test_setup.py` first
- Check if data directory exists and has images
- Verify batch size isn't too large for your GPU

### **If inference fails:**
- Make sure model file exists (`best_model.pth`)
- Check if image path is correct
- Verify image is in correct format (PNG)

---

## **File Outputs**

After running scripts, you'll get these files:

### **Training Outputs:**
- `best_model.pth` - Trained model weights
- `training_metrics.png` - Training/validation plots
- `confusion_matrix.png` - Model performance matrix

### **K-Fold Outputs:**
- `best_model_kfold.pth` - Best model from k-fold
- `kfold_results.png` - K-fold analysis plots

### **Testing Outputs:**
- `confusion_matrix_test.png` - Test confusion matrix
- `per_class_accuracy.png` - Per-class performance

---

This guide should help you understand when and how to use each script in your CNN project!
