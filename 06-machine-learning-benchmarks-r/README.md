# 🤖 End-to-End Machine Learning & Data Mining Benchmarks in R

![R](https://img.shields.io/badge/R-4.2%2B-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Caret](https://img.shields.io/badge/Caret-Benchmarking-blue?style=for-the-badge)
![Random Forest](https://img.shields.io/badge/Random_Forest-ML-green?style=for-the-badge)
![SVM](https://img.shields.io/badge/SVM-Radial-purple?style=for-the-badge)
![Apriori](https://img.shields.io/badge/Algorithm-Apriori_%26_KMeans-orange?style=for-the-badge)

---

## 📌 Project Overview

This repository provides a comprehensive **Machine Learning & Data Mining Benchmark Suite in R**, evaluating supervised classification, continuous regression, unsupervised clustering, and market basket association rules across real-world datasets:

1. **Multi-Class Vehicle Silhouette Classification:** Benchmarking KNN, ANN (`nnet`), SVM Radial, and Random Forest under 75/25 Hold-out and 10-Fold Cross-Validation.
2. **Diabetes Onset Diagnosis Classification:** Binary classification benchmark comparing sensitivity, specificity, and confusion matrices.
3. **Graduate Admission Chance Regression:** Continuous estimation evaluated via $R^2$, Standard Error of Estimate ($S_{yx}$), Pearson $r$, $RMSE$, and $MAE$.
4. **Forest Biomass Continuous Regression:** Environmental modeling predicting standing forest biomass ($t/ha$).
5. **Vehicle Silhouette Unsupervised Clustering:** K-Means algorithm ($k=10$) over Z-score normalized feature spaces.
6. **Gym Activity Association Rule Mining:** Apriori association rules identifying exercise patterns using Support, Confidence, and Lift.

---

## 📊 Benchmark Results Summary Matrix

### 1. 🚗 Multi-Class Vehicle Silhouette Classification

| Model Architecture | Parameter Configuration | Evaluation Strategy | Accuracy | Performance Summary |
| :--- | :---: | :---: | :---: | :--- |
| 🟢 **Random Forest Classifier** | `mtry = 4` | **10-Fold Cross-Validation** | **1.000 (100.0%)** | **Best Model:** Perfect class separation across vehicle profiles. |
| 🟢 **Random Forest Classifier** | `mtry = 6` | 75/25 Hold-Out | 0.7441 (74.41%) | High accuracy on split test partition. |
| 🟡 **SVM Radial Kernel** | `C = 10, sigma = 0.1` | **10-Fold Cross-Validation** | **0.9799 (97.99%)** | Excellent non-linear decision boundaries. |
| 🟡 **SVM Radial Kernel** | `C = 1, sigma = 0.1` | 75/25 Hold-Out | 0.7725 (77.25%) | Strong generalizability on test set. |
| 🔵 **Artificial Neural Network (ANN)** | `size = 7, decay = 0.5` | 10-Fold Cross-Validation | 0.6489 (64.89%) | Stable convergence. |
| 🔴 **KNN Classifier** | `k = 5` | 75/25 Hold-Out | 0.6777 (67.77%) | Sensitive to localized feature variance. |

---

### 2. 📈 Regression Performance Matrix (Admission & Forest Biomass)

| Benchmark Task | Best Model | $R^2$ Score | $S_{yx}$ | Pearson $r$ | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Graduate Admission Chance** | 🟢 **Random Forest (10-Fold CV)** | **0.8650** | **0.0541** | **0.9320** | **0.0550** | **0.0412** |
| **Forest Biomass ($t/ha$)** | 🟢 **Random Forest (10-Fold CV)** | **0.9120** | **10.45** | **0.9560** | **11.20** | **8.64** |

---

## 📁 Repository Structure

```text
06-machine-learning-benchmarks-r/
├── README.md                                           <-- Technical Benchmark Report
└── scripts/
    ├── 01_vehicle_multi_class_classification.R         <-- Multi-Class Classification Script
    ├── 02_diabetes_diagnosis_classification.R         <-- Binary Classification Script
    ├── 03_graduate_admission_regression.R              <-- Graduate Admission Regression Script
    ├── 04_forest_biomass_estimation_regression.R       <-- Environmental Biomass Regression Script
    ├── 05_vehicle_silhouette_kmeans_clustering.R       <-- K-Means Clustering Script
    └── 06_gym_activity_association_rules.R             <-- Apriori Association Rules Script
```

---

## 🚀 How to Run in R

```r
# Install required dependencies
install.packages(c("readr", "dplyr", "caTools", "caret", "class", "nnet", "e1071", "randomForest", "kernlab", "arules", "Metrics"))

# Run Classification Benchmarks
source("scripts/01_vehicle_multi_class_classification.R")
source("scripts/02_diabetes_diagnosis_classification.R")

# Run Regression Benchmarks
source("scripts/03_graduate_admission_regression.R")
source("scripts/04_forest_biomass_estimation_regression.R")

# Run Unsupervised Clustering & Association Rules
source("scripts/05_vehicle_silhouette_kmeans_clustering.R")
source("scripts/06_gym_activity_association_rules.R")
```

---
**Author:** Matheus Paixão  
**LinkedIn:** [Matheus Paixão](https://www.linkedin.com/in/matheus-paix%C3%A3o-5803b321b/?locale=pt)  
**GitHub:** [@MatheusPaixaoP](https://github.com/MatheusPaixaoP)
