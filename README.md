# Thesis_code_algorithms

# Thesis Code and Algorithms

This repository contains the MATLAB and Python code developed as part of our thesis: **"Pattern Recognition Towards Early Detection of Diabetic Small-Fiber Neuropathy Using Optical Coherence Tomography Angiography."**

The code implements algorithms for processing and analyzing OCTA data to identify patterns indicative of diabetic neuropathy.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Authors](#authors)


---

## Project Overview
This project focuses on using pattern recognition and machine learning techniques to analyze OCTA images for early detection of diabetic small-fiber neuropathy. The code includes:
- Training of a UNet CNN model
- Processing of OCTA maximum intensity projections(MIP) of the reticular plexus 
- Processing of OCTA MIPs of the papillary plexus

---

## Directory Structure


```
├── Papillary plexus algorithm/         # Scripts and data for papillary plexus analysis
│   ├── Training UNet/                  # Code and data for training the U-Net segmentation model
│   │   ├── Training data/              # Input training data for the model
│   │   └── epidermus_segmentation_CNN.ipynb  # Notebook for U-Net segmentation training
│   └── Papillary_plexus_MAIN.ipynb     # Main script for papillary plexus algorithm
│
├── Reticular plexus algorithm/         # Scripts for reticular plexus analysis
│   ├── Functions/                      # Helper functions for reticular plexus analysis
│   └── Reticular_plexus_MAIN.m         # Main MATLAB script for reticular plexus algorithm
│
└── README.md                           # Documentation and project overview

```
---

## Dependencies
### **MATLAB**
- Version: R2021a (or later)
- Required Toolboxes: Image Processing Toolbox, Statistics and Machine Learning Toolbox.

### **Python**
- Python 3.8 (or later)
- tensorflow: For deep learning and image processing.
- numpy: For numerical operations.
- matplotlib: For data visualization and plotting.
- opencv-python: For image processing.
- SimpleITK: For medical image processing.
- scikit-image: For image filtering, segmentation, and morphology operations.
- scipy: For scientific computations.
- Install dependencies using:
  pip install tensorflow numpy matplotlib opencv-python SimpleITK scikit-image scipy

---
## How to run 

For the Reticular plecus algorithm only the path to the OCTA data(in DICOM format) needs to be changed to the data you want to process
![image](https://github.com/user-attachments/assets/920854e4-056b-412d-91fb-0b23e0136ea2)

For the papillary plexus algorithm you firstly need to run the "import packages" and "functions" blocks
![image](https://github.com/user-attachments/assets/80617e25-241b-4b89-b8cc-eade7e03298a)

Then the path to the UNet model, OCTA data(in DICOM format) and OCTY data (also in DICOMO) needs to be changed to where you have stored them
Then run all sections 

---

## Authors 

Mads Leth Grønbeck
s194504@dtu.dk
