# 🌎 CDNeXt: Change Detection with ConvNeXt Architecture

A PyTorch implementation of change detection for remote sensing imagery using ConvNeXt backbone with various attention mechanisms.

## 🎯 Overview

This repository contains a complete implementation of a change detection model that leverages the power of ConvNeXt (A ConvNet for the 2020s) architecture for identifying changes in bi-temporal remote sensing images. The model incorporates multiple attention mechanisms including spatial-temporal attention, CBAM (Convolutional Block Attention Module), and non-local attention to enhance feature representation and change detection accuracy.

## 🏗️ Architecture

The CDNeXt model consists of:

- **Backbone**: ConvNeXt encoder (Tiny/Small/Base/Large/XLarge variants)
- **Attention Mechanisms**: 
  - Spatiotemporal Attention
  - CBAM (Channel & Spatial Attention)
  - Non-local Attention
  - DANet Module (Position & Channel Attention)
- **Feature Fusion**: Multi-scale feature fusion with upsampling
- **GPU Optimization**: Dual T4 GPU acceleration support

## 📊 Key Features

- ✅ Multiple ConvNeXt backbone variants (Tiny to XLarge)
- ✅ Comprehensive attention mechanism integration
- ✅ GPU-accelerated data processing and training
- ✅ Support for LEVIR-CD dataset
- ✅ Advanced data augmentation techniques
- ✅ Memory-efficient patch-based processing
- ✅ Multi-scale training and inference

## Usage
1)  Get the LEVIR_CD+ datset from the link: https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd-change-detection
2)  Data Preparation

    <img width="371" height="408" alt="image" src="https://github.com/user-attachments/assets/c747c3b2-418f-48af-9032-419ae786742c" />
3) Training
   In the train.py file, you can set the variable backboneName to use another backbone network:
   ```python
   backboneName = "tiny" #'tiny','small','base','resnet18'
   ```
4) Implementation of Notebook
   - Implement the code as per the instruction provided in the notebook
5) The model is trained upto 120 epoch due to limitation of Kaggle notebook.
6) Run this on high GPU machine to get the ultimate output as mentioned in the paper
7) Paper used to implement this project: https://www.sciencedirect.com/science/article/pii/S1569843224001213

## Results
<img width="5013" height="5005" alt="results (1)" src="https://github.com/user-attachments/assets/151c903d-3c5a-4db7-839b-6fc089d2a41d" />

<img width="800" height="245" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/2c7ce90a-ffed-46e3-ad98-21cfbb00bb7a" />

- When I did training upto 100 epoch, then the results are:
  
  <img width="630" height="253" alt="image (2)" src="https://github.com/user-attachments/assets/92083d21-d066-4e8d-9da2-cf3222b761a8" />



   
