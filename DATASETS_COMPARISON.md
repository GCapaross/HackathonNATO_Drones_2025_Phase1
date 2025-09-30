# Dataset Comparison: What You Have

You have **3 different drone detection datasets** in your project. Here's the breakdown:

---

## 1. DroneRF (Original) - `Data_aggregated/`

### Overview
- **Source**: Mohammad Al-Sa'd et al., 2018 research paper
- **Purpose**: Drone detection and identification using RF signatures
- **Location**: `/home/gabriel/Desktop/HackathonNATO_Drones_2025/Data_aggregated/`

### Data Format
- **Type**: Numerical frequency bins (NOT images)
- **Format**: `.mat` files + `RF_Data.csv` (483MB)
- **Features**: 2048 frequency bins per time segment
- **Size**: Each `.mat` file is 27-61MB

### Classes (10 total)
1. **Background** (00000): No drone, ambient RF
2. **Bebop Drone** (4 modes):
   - Flying (10000)
   - Hovering (10001)
   - Flying + Video (10010)
   - Hovering + Video (10011)
3. **AR Drone** (4 modes):
   - Flying (10100)
   - Hovering (10101)
   - Flying + Video (10110)
   - Hovering + Video (10111)
4. **Phantom Drone** (1 mode):
   - Flying (11000)

### Training Method
- **Model**: Dense Neural Network (fully connected layers)
- **Input**: 2047 numerical features (frequency bins)
- **Output**: Classification of drone presence, type, and flight mode
- **Scripts**: `Classification.py`, `Classification2.py`, `Classification3.py`
- **Saves**: Only predictions (CSV files), **NOT trained models**

### Use Case
- Fast classification on frequency domain data
- Good for real-time detection systems
- Lower computational requirements

---

## 2. DroneRFb-Spectra - `DroneRFb-Spectra/Data/`

### Overview
- **Source**: Newer dataset from Zhejiang University, China
- **Purpose**: Real-world drone recognition with modern drones
- **Location**: `/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb-Spectra/Data/`

### Data Format
- **Type**: Pre-computed spectrograms (2D arrays, already processed)
- **Format**: `.npy` files (NumPy arrays)
- **Size**: 543 × 512 (time × frequency)
- **Data type**: `float16`, normalized to [0, 1]
- **Each file**: ~500KB

### Classes (24 total)
**DJI Drones** (16 models):
1. Background (WiFi/Bluetooth)
2. DJI Phantom 3
3. DJI Phantom 4 Pro
4. DJI MATRICE 200
5. DJI MATRICE 100
6. DJI Air 2S
7. DJI Mini 3 Pro
8. DJI Inspire 2
9. DJI Mavic Pro
10. DJI Mini 2
11. DJI Mavic 3
12. DJI MATRICE 300
13. DJI Phantom 4 Pro RTK
14. DJI MATRICE 30T
15. DJI AVATA
16. DJI DIY
17. DJI MATRICE 600 Pro

**RC Controllers** (7 models):
18. VBar
19. FrSky X20
20. Futaba T16IZ
21. Taranis Plus
22. RadioLink AT9S
23. Futaba T14SG
24. Skydroid

### Frequency Bands
- 915 MHz (ISM band)
- 2.44 GHz (WiFi/Bluetooth)
- 5.80 GHz (WiFi/Video)
- Bandwidth: 100 MHz

### Training Method
- **Model**: CNN (Convolutional Neural Network)
- **Input**: 543 × 512 × 1 spectrogram images
- **Output**: Classification of 24 classes
- **Script**: `train_droneRFb_cnn.py`
- **Saves**: Both models (`.h5` files) AND results

### Use Case
- More modern drones (2020s models)
- Includes RC controller detection
- Better for image-based deep learning
- Requires more GPU/computation

---

## 3. MDPI Dataset - `spectrogram_training_data_20220711/`

### Overview
- **Source**: MDPI research dataset for RF signal classification
- **Purpose**: Multi-protocol RF signal detection (WLAN, BLE, Bluetooth)
- **Location**: `/home/gabriel/Desktop/HackathonNATO_Drones_2025/GitHubs/public-mdpi-dataset-helper-scripts-dataset_20220711/`

### Data Format
- **Type**: Raw complex IQ samples (complex64)
- **Format**: Binary files in `results/` directory
- **Generates**: PNG spectrograms with YOLO labels

### Classes (4-5 total)
1. **WLAN** (WiFi signals)
2. **BT_CLASSIC** (Bluetooth Classic)
3. **BLE** (Bluetooth Low Energy)
4. **Collisions** (overlapping signals)

### Features
- **Packet-based**: Individual RF packets
- **Merging**: Combines packets into frames with collisions
- **Augmentation**: Adds noise, channel effects, frequency offsets
- **Labels**: YOLO format bounding boxes
- **Spectrograms**: Generated on-the-fly from raw samples

### Training Method
- **Model**: YOLO (object detection) or CNN
- **Input**: 1024 × 192 spectrogram images (PNG)
- **Output**: Bounding boxes for signal detection
- **Script**: `complete_pipeline_generator.py`
- **Purpose**: Detect and localize multiple signals in RF spectrum

### Use Case
- General RF signal detection
- Not drone-specific
- Good for spectrum monitoring
- Bounding box detection (where + what)

---

## Quick Comparison Table

| Feature | DroneRF | DroneRFb-Spectra | MDPI Dataset |
|---------|---------|------------------|--------------|
| **Data Type** | Frequency bins (numbers) | Spectrograms (images) | Raw IQ samples |
| **Format** | `.mat` + CSV | `.npy` | Binary + generated PNG |
| **Classes** | 10 | 24 | 4-5 |
| **Drones** | 3 models (older) | 17 models (modern) | Not drone-specific |
| **Model** | Dense NN | CNN | CNN/YOLO |
| **Output** | Classification | Classification | Object detection |
| **Saves Model?** | No | Yes | Depends |
| **Year** | ~2018 | ~2022+ | ~2022 |
| **Best for** | Fast detection | Modern drones | Signal monitoring |

---

## Which Should You Use?

### Use **DroneRF** if:
- You need fast inference
- Working with older drones (Bebop, AR Drone, Phantom)
- Want to classify flight modes
- Have limited compute resources
- Need numerical features (not images)

### Use **DroneRFb-Spectra** if:
- Working with modern DJI drones
- Want to detect RC controllers too
- Have GPU available for CNN training
- Need visual spectrograms
- Want pre-processed, ready-to-train data

### Use **MDPI Dataset** if:
- Detecting general RF signals (not just drones)
- Need bounding boxes (localization)
- Working with WiFi/Bluetooth interference
- Want to generate custom spectrograms
- Need augmented training data

---

## Summary

**DroneRF**: Old but fast, numerical features, 3 drone types
**DroneRFb-Spectra**: Modern drones, visual spectrograms, 24 classes ✨ **RECOMMENDED for your hackathon**
**MDPI**: General RF signals, not drone-specific, YOLO-style detection

For a NATO drone detection hackathon, **DroneRFb-Spectra** is probably your best bet because:
- Most modern drones
- Already processed spectrograms
- Includes RC controllers
- Ready to train with CNN
