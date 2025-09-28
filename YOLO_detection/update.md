# YOLO Detection System - Development Log

## Project Overview
Developing a YOLO-based object detection system for RF signal detection in spectrograms as part of NATO hackathon. The goal is to detect and classify different types of RF signals (WLAN, Bluetooth, Background) in spectrogram images.

## Phase 1: Initial CNN Approach (Completed)
- Started with CNN classification approach
- Realized fundamental flaw: CNN does image classification (one label per image) but dataset has YOLO-style bounding boxes (multiple objects per image)
- Documented problem in Problem.md
- Decided to switch to YOLO object detection approach

## Phase 2: YOLO System Development (Completed)

### Data Analysis
- Dataset: spectrogram_training_data_20220711 with 20,000+ labeled spectrograms
- File structure: Each spectrogram has 4 files (raw_data, .png, _marked.png, .txt)
- Class distribution analysis:
  - Background: 329,923 labels (71.3%)
  - WLAN: 73,946 labels (16.0%)
  - Bluetooth: 58,811 labels (12.7%)
  - BLE: 0 labels (0.0%)
- Key finding: BLE class is completely missing from dataset

### System Architecture
- Created YOLO_detection folder with complete system
- Implemented custom YOLO model (SimpleYOLO) with 3 classes
- Built data loader for YOLO format labels
- Created training script with proper YOLO loss function
- Developed testing and visualization tools

### Key Components Created
1. yolo_data_loader.py - Handles YOLO format data loading
2. yolo_model.py - Custom YOLO architecture
3. train_yolo.py - Training script with YOLO loss
4. test_yolo_model.py - Testing and visualization
5. labeling.py - Analysis and verification tools

## Phase 3: Training Results (Problem Identified)

### Training Execution
- Model trained for 2 epochs (stopped early due to time constraints)
- Model parameters: 2,123,547
- Training completed successfully without errors
- Best model saved to checkpoints/best_model.pth

### Training Behavior - CRITICAL ISSUE
- Loss immediately converged to near zero (0.0003)
- This was initially thought to be good performance
- Later analysis revealed this was actually a major problem

### Initial Hypothesis (INCORRECT)
- Thought the model was learning quickly
- Assumed low loss meant good performance
- Did not realize the model was learning to predict zeros

## Phase 4: Testing Results (Problem Confirmed)

### Test Setup
- Created comprehensive testing script with 3-panel visualization
- Tests original spectrogram, model predictions, and human-marked ground truth
- Organized results in yolo_testing_results folder

### Test Results - NO DETECTIONS
- Model consistently predicts "No RF signals detected"
- Tested with confidence thresholds from 0.5 down to 0.01
- Zero detections across all test images

### Debug Analysis - ROOT CAUSE IDENTIFIED
Debug mode revealed the actual problem:

```
Raw output shape: torch.Size([1, 3, 9])
Anchor 0: Sigmoid confidence: 0.000000 (raw: -48.1394)
Anchor 1: Sigmoid confidence: 0.000000 (raw: -48.0808) 
Anchor 2: Sigmoid confidence: 0.000000 (raw: -46.3272)
```

### Problem Analysis
1. **Model learned to predict extremely negative confidence values**
2. **Sigmoid of -48 ≈ 0.000000** - model thinks there are no objects
3. **All anchors predict "no object"** regardless of input
4. **Model essentially learned to say "I see nothing"**

## Root Cause Analysis

### Primary Issue: Loss Function
- Custom YOLO loss function was too simplified
- Model learned that predicting zeros minimizes loss
- No proper object detection learning occurred

### Secondary Issues
1. **Training Data Imbalance**: 71% Background vs 29% actual signals
2. **Model Architecture**: Simplified YOLO may be insufficient
3. **Training Duration**: Only 2 epochs may be insufficient
4. **Loss Convergence**: Immediate convergence to zero indicates learning failure

### What We Learned
- Low loss does not always mean good performance
- Model can learn to predict zeros to minimize loss
- YOLO loss function is critical for proper object detection
- Need to verify model is actually learning, not just minimizing loss

## Current Status
- YOLO system architecture is complete and functional
- Training infrastructure works correctly
- Model loads and runs without errors
- Problem is in the learning process, not the implementation

## Next Steps Required
1. **Implement proper YOLO loss function** (not simplified version)
2. **Use real YOLO implementation** (YOLOv8) instead of custom model
3. **Verify training data quality** and label accuracy
4. **Implement proper evaluation metrics** during training
5. **Consider data augmentation** to handle class imbalance

## Technical Notes
- Model correctly handles 4 classes (detected from saved checkpoint)
- Image preprocessing works correctly (no resizing, original resolution)
- Coordinate conversion from normalized to pixel coordinates is correct
- Visualization system works properly
- File organization and logging system is complete

## Files Created
- YOLO_detection/ folder with complete system
- yolo_testing_results/ folder for organized test outputs
- Comprehensive documentation and analysis tools
- Debug capabilities for model output analysis

## Key Lesson
The most important lesson learned: **A model that converges to zero loss immediately is not learning - it's failing to learn.** Proper object detection requires sophisticated loss functions that actually teach the model to detect objects, not just minimize a simple loss metric.

## Technical Insight from Team Discussion
**What we discovered about loss functions and overfitting:**

The loss functions (like cross entropy) are basically calculations that measure how "right" the AI is while it's doing its work. They're good indicators of certainty and how accurate the AI can be.

**The problem with our old model:**
- When loss converges too quickly to zero, something is wrong
- Loss is never supposed to be exactly 0
- This can be a sign of overfitting

**What happened in our case:**
The model was overfitting to predict "nothing" - it learned to say "okay this image has nothing, this image has nothing" so it converged to zero loss. Basically it was overfitting to not classify anything at all.

**The real issue:**
The marked zones in the data weren't strange - the strange part was that not everything was marked. The model learned to predict "no objects" for everything, which gave it a perfect loss of zero, but it wasn't actually learning to detect RF signals.

This is why we switched to using real YOLO instead of our custom implementation - the loss functions in real YOLO are much more sophisticated and actually teach the model to detect objects properly.

## How YOLO Validation Works
**Important clarification on how YOLO evaluates performance:**

YOLO validation compares **each predicted bounding box** to **each ground truth bounding box**, not per-image.

**Example:**
- Image has 5 RF signals (ground truth)
- Model detects 3 signals correctly
- Result: 3/5 = 60% accuracy for that image
- **NOT**: 0/1 (image completely wrong)

**Why this makes sense:**
- Partial detection is still valuable
- Finding 3 out of 5 signals is useful progress
- Model learns gradually to detect more signals
- Real-world scenario: you might miss some signals but detect others

**Key point:** YOLO evaluates per-object detection, not per-image classification. So if the model gets 3 out of 5 signals right, that's actually good progress - much better than detecting 0 out of 5 like our old model did.

## Phase 5: Ultralytics YOLOv8 Implementation (Current)

### New System Architecture
- Switched from custom YOLO to Ultralytics YOLOv8
- Created clean YOLO_detection folder with proper implementation
- Moved all custom implementation files to first_test_attempt/ folder
- Implemented standard YOLO training and testing workflow

### Training Results (20 Epochs)
- **Model**: YOLOv8 using Ultralytics framework
- **Training Duration**: 20 epochs (completed successfully)
- **Dataset**: 20,000 spectrogram images with YOLO labels
- **Classes**: 4 classes (Background, WLAN, Bluetooth, BLE)
- **Device**: CUDA (GPU acceleration enabled)
- **Model Location**: yolo_training/rf_detection/weights/best.pt

### Training Behavior Analysis
- **Initial Issue**: Model was identifying everything as Background
- **Root Cause**: Class imbalance (71% Background vs 29% actual signals)
- **Training Strategy**: Model learned to be conservative due to data imbalance
- **Result**: Model tends to predict Background class more frequently

### Enhanced Testing System
- **New Test Script**: Enhanced test_model.py with comprehensive analysis
- **Visualization**: 2x2 panel layout showing:
  - Original spectrogram
  - Model predictions with bounding boxes
  - YOLO labels (ground truth) with bounding boxes  
  - Human-marked reference image
- **Analysis Features**:
  - Detection counting and comparison
  - Detection ratio calculation
  - Numbered test results for tracking
  - Comprehensive test summary with matrix
  - Automatic analysis notes and recommendations

### Test Results Format
- **Output Directory**: yolo_testing_results/
- **Individual Results**: test_XX_filename.png (2x2 visualization)
- **Summary File**: test_summary.txt with detailed matrix
- **Analysis**: Automatic detection of model behavior issues

### Current Model Performance
- **Status**: Model trained and ready for testing
- **Expected Behavior**: May be conservative due to class imbalance
- **Testing Ready**: Enhanced test script available for comprehensive evaluation
- **Next Step**: Run test_model.py to evaluate actual performance

### Key Improvements Made
1. **Professional YOLO Implementation**: Using industry-standard Ultralytics YOLOv8
2. **Enhanced Testing**: Comprehensive visualization and analysis
3. **Better Documentation**: Clear tracking of training and testing results
4. **Automatic Analysis**: Script identifies potential issues automatically
5. **Organized Output**: Clean file structure with numbered results

### Training Notes
- Model completed 20 epochs successfully
- CUDA acceleration used throughout training
- Best model automatically saved during training
- Training can be resumed from existing model if needed
- Class imbalance remains a challenge but model should still detect signals
- Seems to be classifying mostly as background
- Need to test accuracy

## Phase 6: Model Testing Results and Analysis

### Test Results Summary
After comprehensive testing with the enhanced test_model.py script, the following conclusions were reached:

**Model Performance Issues Identified:**
- **Overfitting towards Background class**: Model has learned to classify most regions as Background
- **Difficulties detecting WLAN and Bluetooth signals**: Model struggles to identify actual RF signals
- **Conservative detection behavior**: Model tends to predict "no objects" or "Background only" as the safest strategy

### Root Cause Analysis
**Class Imbalance Impact:**
- Training data distribution: 71% Background vs 16% WLAN vs 13% Bluetooth
- Model learned that predicting Background minimizes loss
- This created a bias towards the majority class
- Model adopted the strategy of "when in doubt, predict Background"

**Detection Challenges:**
- Model has difficulties finding WLAN and Bluetooth signals in spectrograms
- **Critical Issue**: Model is not detecting Bluetooth at all (not just misclassifying)
- **Root Cause**: Model learned to ignore minority classes completely
- Tendency to classify signal regions as Background
- Learned to be overly conservative to avoid false positives
- This is a common problem in imbalanced datasets

**Specific Problem Identified:**
- **Bluetooth Detection**: Model is not finding Bluetooth signals at all
- **WLAN Detection**: Model is not finding WLAN signals at all
- **Not a classification problem**: Model isn't misclassifying them as Background
- **It's a detection problem**: Model is not detecting them in the first place

### Recommendations for Improvement

**For "Not Detecting At All" Problem:**
1. **Lower confidence thresholds** - Model might be too conservative
2. **Increase objectness weight** - Make model more sensitive to objects
3. **Use focal loss** - Focus on hard-to-detect classes
4. **Data augmentation** - Create more examples of minority classes
5. **Collect more training data** - Especially for WLAN and Bluetooth

**For "Misclassification" Problem:**
1. **Address class imbalance through weighted loss functions**
2. **Implement data augmentation to balance class representation**
3. **Use focal loss to focus on hard-to-detect classes**
4. **Consider collecting more WLAN and Bluetooth training samples**
5. **Adjust confidence thresholds for better detection sensitivity**

**Additional Solutions for Detection Issues:**
- **Test with very low confidence thresholds** (0.01, 0.05)
- **Check if model is learning objectness** (ability to detect objects)
- **Verify training data quality** - Are WLAN/Bluetooth labels correct?
- **Consider different model architecture** - Maybe YOLOv8s or YOLOv8m instead of nano

### Key Learning
The model's behavior demonstrates a classic case of class imbalance overfitting, where the model learns to predict the majority class (Background) as the safest strategy, leading to poor performance on the minority classes (WLAN, Bluetooth) that are actually the most important to detect.

## Phase 7: Overfitting Fix Implementation

### Problem Analysis
After identifying the overfitting issue, we implemented several strategies to address the class imbalance problem:

**Root Cause Confirmed:**
- Training data distribution: 71% Background vs 16% WLAN vs 13% Bluetooth
- Model learned to minimize loss by predicting the majority class
- Conservative behavior: "when in doubt, predict Background"

### Solution Implementation

#### **1. New Training Directory**
- **Change**: Created `yolo_training3/` folder for retraining
- **Reason**: Keep previous model (`yolo_training2/`) for comparison
- **Benefit**: Can compare old vs new model performance

#### **2. Enhanced Training Parameters**

**What These Parameters Mean (Simple Explanation):**

**Class Loss Weight (`cls=0.5`):**
- **What it does**: Tells the model "pay more attention to getting the class right"
- **Simple explanation**: Like telling a student "getting the answer right is more important than neat handwriting"
- **Why we increased it**: Model was ignoring WLAN/Bluetooth classes, so we make it focus more on classification
- **For detection issues**: This helps the model actually detect objects, not just classify them

**Box Loss Weight (`box=7.5`):**
- **What it does**: Tells the model "make sure the bounding boxes are in the right place"
- **Simple explanation**: Like telling a student "draw the box around the right object"
- **Why we kept it high**: We want accurate positioning of detected signals

**Distribution Focal Loss (`dfl=1.5`):**
- **What it does**: Helps the model be more confident about its predictions
- **Simple explanation**: Like telling a student "be more sure about your answers"
- **Why we kept it**: Helps the model give better confidence scores

#### **3. Data Augmentation Strategy**

**What Data Augmentation Means (Simple Explanation):**
Data augmentation is like creating more training examples by modifying existing images. It's like taking a photo and creating variations (flip it, make it brighter, etc.) to give the model more examples to learn from.

**Mosaic Augmentation (`mosaic=1.0`):**
- **What it does**: Takes 4 images and combines them into one big image
- **Simple explanation**: Like making a collage from 4 different photos
- **Why it helps**: Model sees more combinations of WLAN/Bluetooth signals together

**Mixup Augmentation (`mixup=0.1`):**
- **What it does**: Blends two images together (10% of the time)
- **Simple explanation**: Like mixing two paint colors to create a new color
- **Why it helps**: Creates new examples of WLAN/Bluetooth signals from existing data

**Copy-Paste Augmentation (`copy_paste=0.1`):**
- **What it does**: Copies objects from one image to another (10% of the time)
- **Simple explanation**: Like cutting out a person from one photo and pasting them into another
- **Why it helps**: Creates more WLAN/Bluetooth examples by copying them to different backgrounds

**General Augmentation (`augment=True`):**
- **What it does**: Applies standard modifications (rotation, brightness, etc.)
- **Simple explanation**: Like using photo filters to create variations
- **Why it helps**: Makes the model more robust to different image conditions


### Expected Improvements

#### **Training Behavior Changes:**
- **Slower convergence**: Model should take longer to converge (good sign)
- **Higher loss initially**: Should start with higher loss values
- **Gradual improvement**: Loss should decrease more gradually
- **Better class balance**: Model should learn to detect minority classes

#### **Detection Improvements:**
- **More WLAN detections**: Should find more WLAN signals
- **More Bluetooth detections**: Should find more Bluetooth signals
- **Better confidence scores**: Higher confidence for actual signals
- **Reduced Background bias**: Less tendency to predict everything as Background

### Implementation Details

**Modified Files:**
- `train_model.py`: Updated with new parameters and `yolo_training3` directory
- `test_model.py`: Already configured to test new model
- `generate_training_report.py`: Will generate report for new training

**Training Command:**
```bash
python3 train_model.py
```

**Testing Command:**
```bash
python3 test_model.py --model_path yolo_training3/rf_detection/weights/best.pt
```

**Testing for Detection Issues:**
```bash
# Test with very low confidence thresholds to see if model detects anything
python3 test_model.py --model_path yolo_training3/rf_detection/weights/best.pt --conf_threshold 0.01
python3 test_model.py --model_path yolo_training3/rf_detection/weights/best.pt --conf_threshold 0.05
python3 test_model.py --model_path yolo_training3/rf_detection/weights/best.pt --conf_threshold 0.1
```

### Monitoring Strategy

**During Training:**
- Watch for slower loss convergence (good sign)
- Monitor class-specific metrics if available
- Check for more balanced detection behavior

**After Training:**
- Compare with `yolo_training2` results
- Test with different confidence thresholds
- Analyze confusion matrix for class balance
- Check detection ratios for minority classes

### Expected Outcomes

**Success Indicators:**
- Model detects more WLAN and Bluetooth signals
- Higher confidence scores for actual signals
- Better balance between classes in predictions
- Improved detection ratios in testing

**If Still Overfitting:**
- May need to collect more minority class data
- Consider focal loss implementation
- Adjust confidence thresholds
- Try different model architectures

### Documentation Update
This phase represents a systematic approach to addressing class imbalance through:
1. **Parameter tuning** for better class focus
2. **Data augmentation** for minority class enhancement
3. **Systematic testing** to validate improvements
4. **Comparative analysis** between model versions
