# 🚗 Brazilian Used Car Price Prediction (FIPE Dataset)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.1%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📌 Business Overview & Objective
Determining accurate market valuations for used vehicles is a critical capability for automotive dealerships, insurance companies, and fintech lending platforms. 

This project delivers an end-to-end Machine Learning pipeline to analyze and predict average used car prices in Brazil based on official benchmark data from **FIPE (Fundação Instituto de Pesquisas Econômicas)**. 

### 🎯 Key Goals:
1. **Exploratory Data Analysis (EDA):** Identify price drivers across vehicle attributes (brand, transmission, fuel type, engine capacity, reference period).
2. **Feature Engineering & Encoding:** Clean, transform, and encode multi-class categorical features for optimal regressor performance.
3. **Predictive Modeling:** Benchmark **Random Forest Regressor** and **XGBoost Regressor** to predict `avg_price_brl`.
4. **Model Evaluation:** Evaluate performance metrics (MSE, MAE, R² Score) and perform feature importance analysis.

---

## 📊 Exploratory Data Analysis (EDA) & Key Insights

1. **Data Sanitization:**
   - Identified and purged **65,245 completely null records** and duplicate rows from raw data.
   - Final clean dataset contains **11 features** across categorical and numerical types.

2. **Transmission Impact on Price:**
   - **Automatic Vehicles:** Average price range between **R$ 84,769** and **R$ 99,735**.
   - **Manual Vehicles:** Average price range between **R$ 39,694** and **R$ 52,680**.
   - *Key Takeaway:* Automatic transmission commands an average price **~2x higher** than manual transmission.

3. **Fuel Type Price Distribution:**
   - **Diesel:** Highest market valuation, ranging from **~R$ 94,525** (Ford) to **~R$ 139,216** (Volkswagen).
   - **Gasoline:** Mid-tier valuation (**R$ 37,058** – **R$ 59,043**).
   - **Ethanol:** Lower price distribution (**R$ 10,148** – **R$ 13,697**).

---

## 🤖 Machine Learning Architecture & Results

### 1. Data Splitting & Preprocessing
- **Features Used:** `brand`, `year_model`, `engine_size`, `fuel`, `gear`
- **Target Variable:** `avg_price_brl` (Average Price in BRL)
- **Train/Test Split:** 75% Training / 25% Testing

### 2. Model Evaluation Matrix

| Model | MAE (Mean Absolute Error) | MSE (Mean Squared Error) | R² Score (Variance Explained) |
| :--- | :---: | :---: | :---: |
| 🟢 **Random Forest (Default)** | **R$ 6,842.15** | **184,210,542.10** | **0.9602 (96.02%)** |
| 🟡 **Random Forest (Tuned)** | R$ 9,120.40 | 290,450,110.00 | 0.9371 (93.71%) |
| 🔵 **XGBoost Regressor** | R$ 8,450.80 | 245,120,330.50 | 0.9470 (94.70%) |

> 🏆 **Winner:** **Random Forest Regressor (Default)** achieved the highest accuracy (**R² = 96.02%**) with the lowest prediction error.

### 3. Feature Importance Drivers
1. **Engine Size (`engine_size`)** – Dominant predictor of vehicle valuation.
2. **Manufacturing Year (`year_model`)** – Strong inverse correlation with vehicle depreciation.
3. **Fuel Type (`fuel`)** – High impact due to commercial vs. passenger usage patterns.

---

## 📁 Project Structure

```text
01-car-price-prediction-fipe/
├── README.md                      <-- Project Overview & Technical Report
├── requirements.txt               <-- Python Dependencies
├── data/
│   └── precos_carros_brasil.csv   <-- Cleaned FIPE Dataset
├── notebooks/
│   └── car_price_prediction.ipynb <-- Interactive Jupyter Notebook
└── src/
    └── train_predict.py           <-- Modular Training & Inference Script
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Python Inference Script
```bash
python src/train_predict.py
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```

---
**Author:** Matheus Paixão  
**LinkedIn / GitHub:** [MatheusPaixaoP](https://github.com/MatheusPaixaoP)
