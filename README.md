# 🚀 Data Science, Machine Learning & AI Portfolio
**Author:** Matheus Paixão  
**Role:** Data Scientist | AI & Data Specialist  
**Location:** Brazil  
**LinkedIn:** [Matheus Paixão](https://www.linkedin.com/in/matheus-paix%C3%A3o-5803b321b/?locale=pt) | **GitHub:** [@MatheusPaixaoP](https://github.com/MatheusPaixaoP)

---

## 📌 About Me & Core Focus
I am a **Data Scientist & AI Specialist** focused on driving business value through data-driven intelligence, predictive modeling, applied statistics, NLP feature engineering, signal optimization, and end-to-end Machine Learning pipelines.

While my primary domain is **Data Science & AI in Data**, I also build specialized applications across **Natural Language Processing (NLP)**, **Signal Optimization**, **Computer Vision**, and **Statistical Econometrics**.

```text
├── 📊 Data Science, Machine Learning & Predictive Analytics (Core Focus)
├── 🤖 End-to-End Supervised & Unsupervised Machine Learning Suite
├── 🌐 Natural Language Processing (NLP) & Feature Engineering
├── 📡 Signal Processing & ML Dataset Optimization
├── 📈 Applied Statistics, Econometrics & Inferential Modeling
├── 🌲 Environmental Data Science & Remote Sensing Biometrics
└── 👁️ Specialized AI Applications (Computer Vision & Deep Learning)
```

---

## 🔬 Featured Projects

### 📊 Data Science, Machine Learning & Analytics (Core Projects)

#### 1. 🤖 [End-to-End Machine Learning & Data Mining Benchmarks in R](./06-machine-learning-benchmarks-r)
* **Goal:** Comprehensive ML benchmark suite covering multi-class classification, binary diagnosis, continuous regression, K-Means clustering, and Apriori association rules.
* **Key Achievements:**
  * Benchmarked KNN, Neural Networks (`nnet`), SVM Radial, and Random Forest across Hold-out and 10-Fold Cross Validation.
  * **Top Result:** Random Forest (10-Fold CV) achieved **100% Accuracy** on vehicle silhouette multi-class classification and **R² = 91.2%** on environmental biomass regression.
* **Stack:** `R`, `Caret`, `Random Forest`, `SVM Radial`, `Neural Networks`, `Apriori`, `K-Means`
* **📁 Explore Project:** [View Code & Report](./06-machine-learning-benchmarks-r)

#### 2. 🌐 [Multi-Language NLP Text Classifier & Feature Engineering](./04-nlp-language-detection-feature-engineering)
* **Goal:** Build an automated multi-language NLP classifier (English, Spanish, Portuguese) using custom Regex feature engineering.
* **Key Achievements:**
  * Engineered 17 custom linguistic features (word length, inverted Spanish punctuation `¿`/`¡`, cedillas `ç`, eñes `ñ`, suffixes `-ção`/`-ción`, contraction patterns, stopword ratios).
  * **Top Result:** Achieved **~96.0% - 100% test accuracy** using Support Vector Machines (SVM).
* **Stack:** `Python`, `NLP`, `Regex`, `Scikit-Learn (SVM)`, `Pandas`
* **📁 Explore Project:** [View Code & Report](./04-nlp-language-detection-feature-engineering)

#### 3. 📡 [Sonar Signal Classification & Dataset Optimization (UCI Sonar Dataset)](./05-sonar-signal-ml-optimization)
* **Goal:** Optimize a Support Vector Machine pipeline to discriminate between metal cylinder mines and rocks using 60 continuous sonar return frequency channels.
* **Key Achievements:**
  * Applied `StandardScaler` feature normalization, ANOVA F-score feature selection (`SelectKBest`), and 5-Fold Stratified CV grid search.
  * **Top Result:** Boosted classification accuracy from baseline **74.6%** to **~88.9%** (+14.3% improvement).
* **Stack:** `Python`, `Scikit-Learn (SVM, Pipeline, GridSearchCV)`, `Pandas`
* **📁 Explore Project:** [View Code & Report](./05-sonar-signal-ml-optimization)

#### 4. 🚗 [Brazilian Used Car Price Prediction (FIPE Dataset)](./01-car-price-prediction-fipe)
* **Goal:** Predict vehicle market valuation in Brazil using FIPE benchmark data with Machine Learning regressors.
* **Key Achievements:**
  * Cleaned 65k+ null records and performed extensive Exploratory Data Analysis (EDA).
  * Feature engineered multi-class categorical variables (`engine_size`, `fuel`, `gear`).
  * **Top Result:** Achieved **R² Score = 0.9602 (96.02% accuracy)** with Random Forest Regressor.
* **Stack:** `Python`, `Pandas`, `Scikit-Learn`, `XGBoost`, `Jupyter`
* **📁 Explore Project:** [View Code & Report](./01-car-price-prediction-fipe)

#### 5. 📊 [Applied Statistical Inference & Regularized Econometric Wage Modeling](./03-applied-statistics-r)
* **Goal:** Perform non-parametric hypothesis testing and build regularized econometric models (Ridge, Lasso, ElasticNet) for log hourly wage estimation.
* **Key Achievements:**
  * Evaluated non-parametric distributions using Shapiro-Wilk and Mann-Whitney U testing ($p < 0.0001$).
  * Computed non-parametric **95% Bootstrap Confidence Intervals** ($n_{boot} = 1000$) for hourly wages.
  * **Top Result:** **Ridge Regression** yielded the lowest error ($RMSE = 0.40$, $R^2 = 34.18\%$).
* **Stack:** `R`, `Glmnet`, `Caret`, `Stats`, `ggplot2`
* **📁 Explore Project:** [View Code & Report](./03-applied-statistics-r)

#### 6. 🛰️ [Environmental ML & Forestry Biometrics in R](./02-environmental-ml-and-forestry-r)
* **Goal:** Solve remote sensing satellite land cover classification and forestry timber volume estimation.
* **Key Achievements:**
  * Multi-class classification on Landsat MSS satellite data using Random Forest (**91.20% Accuracy**).
  * Evaluated **Spurr Allometric Nonlinear Regression Model** vs. ML algorithms for standing tree timber volume prediction.
  * **Top Result:** Spurr Allometric Model achieved **R² = 90.27%** and **$S_{yx}\% = 10.42\%$**, outperforming pure ML models.
* **Stack:** `R`, `Caret`, `Random Forest`, `SVM`, `Nonlinear Least Squares (nls)`
* **📁 Explore Project:** [View Code & Report](./02-environmental-ml-and-forestry-r)

---

### 👁️ Specialized AI, Deep Learning & Computer Vision Hub (Dedicated Repository)
> *Explore advanced implementations in Genetic Algorithms, Evolutionary Optimization, Neural Architectures (CNNs, LSTMs, GANs, Transformers), Recommender Engines, and Deep Computer Vision Feature Representations.*

| Sub-Module / Domain | Description | Stack | Link |
| :--- | :--- | :--- | :---: |
| 🧬 **Genetic Algorithms (TSP & NLP PCA)** | 100-City TSP Genetic Optimization (+41.8% gain) & Word2Vec/FastText PCA Projections. | Python, Matplotlib, Gensim, Scikit-Learn | 🚀 [View Project](https://github.com/MatheusPaixaoP/ai-deep-learning-computer-vision-portfolio/tree/main/01-genetic-algorithms-tsp-and-nlp-embeddings-pca) |
| 🧠 **Deep Learning Model Architectures** | CNNs, LSTMs, GANs, and Scaled Dot-Product Transformer Self-Attention blocks. | PyTorch, TensorFlow, Keras | 🚀 [View Project](https://github.com/MatheusPaixaoP/ai-deep-learning-computer-vision-portfolio/tree/main/02-deep-learning-architectures-cnn-rnn-gans-transformers) |
| 🛠️ **Applied AI Frameworks Suite** | Collaborative Filtering Book Recommender & DeepDream Gradient Ascent activation maps. | PyTorch, TensorFlow, Scikit-Learn | 🚀 [View Project](https://github.com/MatheusPaixaoP/ai-deep-learning-computer-vision-portfolio/tree/main/03-ai-frameworks-recommendation-deepdream-neuralnets) |
| 👁️ **Computer Vision (LBP vs VGG-16)** | LBP Micro-Texture Descriptors vs 4096-dim Deep VGG-16 Representations + Multi-Classifier Benchmark. | PyTorch, OpenCV, Scikit-Learn | 🚀 [View Project](https://github.com/MatheusPaixaoP/ai-deep-learning-computer-vision-portfolio/tree/main/04-computer-vision-lbp-vgg-feature-extraction) |

👉 **Full Repository:** [ai-deep-learning-computer-vision-portfolio](https://github.com/MatheusPaixaoP/ai-deep-learning-computer-vision-portfolio)

---

## 🛠️ Technical Skill Matrix

```text
Languages     : Python, R, SQL, C++
Data & ML     : Pandas, NumPy, Scikit-Learn, XGBoost, Caret, Random Forest, SVM, Glmnet, nnet, nls
NLP & Signals : Regex, Feature Engineering, Signal Scaling (StandardScaler), SelectKBest
Unsupervised  : K-Means Clustering, Apriori Association Rules (arules)
Statistics    : Hypothesis Testing, Non-Parametric Methods, Econometrics, Bootstrap CIs
Deep Learning : PyTorch, TensorFlow, OpenCV, YOLO
Visualization : Seaborn, Matplotlib, Plotly, ggplot2
Tools & DevOps: Git, Docker, Jupyter, Linux, VS Code
```

---

## 📬 Connect & Contact

- **LinkedIn:** [Matheus Paixão](https://www.linkedin.com/in/matheus-paix%C3%A3o-5803b321b/?locale=pt)
- **GitHub:** [@MatheusPaixaoP](https://github.com/MatheusPaixaoP)
- **Email:** matheuscarvalhop2002@gmail.com

---
*Maintained with ❤️ by Matheus Paixão.*
