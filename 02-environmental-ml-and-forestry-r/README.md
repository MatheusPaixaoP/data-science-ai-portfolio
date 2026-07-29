# 🛰️ Environmental Machine Learning & Forestry Biometrics in R

![R](https://img.shields.io/badge/R-4.2%2B-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Caret](https://img.shields.io/badge/Caret-Ensemble-blue?style=for-the-badge)
![Random Forest](https://img.shields.io/badge/Random_Forest-ML-green?style=for-the-badge)
![SVM](https://img.shields.io/badge/SVM-Radial_Kernel-purple?style=for-the-badge)
![Biometrics](https://img.shields.io/badge/Domain-Forestry_%26_Remote_Sensing-emerald?style=for-the-badge)

---

## 📌 Project Overview

This repository demonstrates **R-based Machine Learning and Biometric Statistical Modeling** applied to two real-world environmental and remote sensing challenges:

1. **Multispectral Satellite Land Cover Classification:** Classifying multi-band satellite voxel neighborhoods into land cover categories using Landsat MSS data.
2. **Forestry Timber Volume Estimation:** Comparing domain-specific **Spurr Allometric Nonlinear Regression Models** against Machine Learning algorithms (Random Forest, SVM, ANN) for non-destructive tree timber volume prediction.

---

## 🛰️ Problem 1: Satellite Multispectral Land Cover Classification

### 🎯 Objective
Classify pixel neighborhoods (3x3 grid across 4 spectral bands: 2 visible, 2 near-infrared) into **6 land cover classes**: *Red Soil, Cotton Crop, Grey Soil, Damp Grey Soil, Vegetation Stubble, and Very Damp Grey Soil*.

### 📊 Benchmark Results

| Algorithm | Multi-Class Confusion Matrix Accuracy | Performance Summary |
| :--- | :---: | :--- |
| 🟢 **Random Forest Classifier** | **91.20% (0.9120)** | **Best Model:** Superior handling of multi-spectral feature interactions. |
| 🟡 **Support Vector Machine (SVM)** | **88.71% (0.8871)** | Strong linear & radial margin separation. |
| 🔴 **Artificial Neural Network (ANN)** | **41.90% (0.4190)** | Sub-optimal without heavy hyperparameter tuning. |

---

## 🌲 Problem 2: Forestry Timber Volume Estimation (Spurr Allometric Model vs. ML)

### 🎯 Objective
Estimate standing tree timber volume ($m^3$) using non-destructive physical measurements: **Diameter at Breast Height ($DAP / DBH$)** and **Total Tree Height ($HT$)**.

The classical **Spurr Allometric Model** is defined as:
$$\text{Volume} = \beta_0 + \beta_1 \cdot \text{DAP}^2 \cdot \text{HT}$$

### 📈 Model Performance & Metrics Matrix

Custom evaluation functions were implemented to evaluate **RMSE**, **Coefficient of Determination ($R^2$)**, **Standard Error of Estimate ($S_{yx}$)**, and **Relative Standard Error ($S_{yx}\%$)**:

| Model Architecture | RMSE | $R^2$ Score | $S_{yx}$ | $S_{yx}\%$ |
| :--- | :---: | :---: | :---: | :---: |
| 🟢 **Spurr Allometric Model (`nls`)** | **0.1297** | **0.9027 (90.27%)** | **0.1367** | **10.42%** |
| 🟡 **Random Forest Regressor** | 0.1503 | 0.8692 (86.92%) | 0.1584 | 12.27% |
| 🔵 **SVM (Radial Kernel)** | 0.1800 | 0.8125 (81.25%) | 0.1898 | 14.72% |
| 🔴 **Artificial Neural Network** | 0.5123 | -0.5191 | 0.5401 | 54.01% |

### 💡 Key Domain Takeaway
The **Spurr Allometric Model** outperformed all standard Machine Learning algorithms ($R^2 = 90.27\%$, $S_{yx}\% = 10.42\%$). In forestry biometrics, incorporating domain-specific physical principles (where volume scales proportionally with $DBH^2 \cdot Height$) yields better generalization than unconstrained black-box models.

---

## 📁 Repository Structure

```text
02-environmental-ml-and-forestry-r/
├── README.md                                    <-- Technical Overview Report
└── scripts/
    ├── 01_satellite_landcover_classification.R  <-- Satellite ML Classification Script
    └── 02_forestry_timber_volume_spurr.R       <-- Forestry Allometric & ML Script
```

---

## 🚀 How to Run in R

```r
# Install required packages
install.packages(c("mlbench", "caret", "randomForest", "e1071", "nnet"))

# Execute Problem 1 (Satellite Land Cover Classification)
source("scripts/01_satellite_landcover_classification.R")

# Execute Problem 2 (Forestry Timber Volume Estimation)
source("scripts/02_forestry_timber_volume_spurr.R")
```

---
**Author:** Matheus Paixão  
**LinkedIn / GitHub:** [MatheusPaixaoP](https://github.com/MatheusPaixaoP)
