# 📡 Sonar Signal Classification & Dataset Optimization (UCI Sonar)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline_%26_GridSearch-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SVM](https://img.shields.io/badge/SVM-Hyperparameter_Tuning-purple?style=for-the-badge)
![Signal Processing](https://img.shields.io/badge/Domain-Sonar_Signal_Optimization-emerald?style=for-the-badge)

---

## 📌 Project Overview
Binary classification of complex sonar signals is essential for naval defense, underwater robotics, and geological seabed mapping.

This project focuses on **dataset preprocessing, feature scaling, signal feature selection, and hyperparameter tuning** to boost classification performance on the benchmark **UCI Connectionist Bench Sonar Dataset** (60 continuous frequency return bands used to discriminate between metal cylinders / mines and rocks).

---

## 🛠️ Optimization Strategy & Pipeline

1. **Signal Preprocessing & Standard Scaling (`StandardScaler`):** Normalizing continuous frequency return energies across all 60 spectral channels to prevent high-amplitude channels from dominating distance metrics.
2. **Feature Selection (`SelectKBest` with ANOVA F-score):** Identifying optimal frequency band subsets to reduce noise and dimensionality.
3. **Hyperparameter Tuning (`GridSearchCV`):** Tuning Support Vector Classifier (SVC) parameters ($C$, $\gamma$, and Kernel function: RBF vs Linear) using Stratified 5-Fold Cross-Validation.

---

## 📊 Performance Metrics & Accuracy Boost

| Pipeline Stage | Model Architecture | Scaling / Selection | Test Accuracy | $F1$-Score (Mines) | $F1$-Score (Rocks) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🔴 **Baseline Model** | Raw `SVC()` | None | **~74.60%** | 0.74 | 0.75 |
| 🟢 **Optimized Pipeline** | `SVC(C=10, gamma='scale', kernel='rbf')` | `StandardScaler` + `SelectKBest` | **~85.71% - 88.89%** | **0.88** | **0.87** |

> 🎯 **Optimization Result:** Achieved a **+11.1% to +14.2% accuracy improvement** over unoptimized baseline models through systematic feature scaling, ANOVA feature selection, and grid search hyperparameter calibration.

---

## 📁 Repository Structure

```text
05-sonar-signal-ml-optimization/
├── README.md                                  <-- Technical Overview Report
├── notebooks/
│   └── sonar_signal_optimization.ipynb        <-- Interactive Jupyter Notebook
└── src/
    └── sonar_classifier.py                    <-- Modular Python ML Optimization Pipeline
```

---

## 🚀 How to Run

### 1. Execute ML Optimization Pipeline
```bash
python src/sonar_classifier.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebooks/sonar_signal_optimization.ipynb
```

---
**Author:** Matheus Paixão  
**LinkedIn:** [Matheus Paixão](https://www.linkedin.com/in/matheus-paix%C3%A3o-5803b321b/?locale=pt)  
**GitHub:** [@MatheusPaixaoP](https://github.com/MatheusPaixaoP)
