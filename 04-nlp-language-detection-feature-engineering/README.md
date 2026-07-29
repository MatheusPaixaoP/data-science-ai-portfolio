# 🌐 Multi-Language NLP Text Classifier & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/Domain-NLP_%26_Feature_Engineering-purple?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Regex](https://img.shields.io/badge/Regex-Pattern_Matching-green?style=for-the-badge)

---

## 📌 Project Overview
Language identification is a foundational preprocessing step in Natural Language Processing (NLP) pipelines for search engines, recommendation systems, and machine translation APIs.

This project implements an **end-to-end multi-class text language classifier** that predicts whether a given sentence is written in **English**, **Spanish**, or **Portuguese**. Rather than relying solely on pre-trained embedding models, this project demonstrates **custom linguistic feature engineering** using Regular Expressions (Regex) and classical Machine Learning (**Support Vector Machines**).

---

## 🛠️ Feature Engineering Architecture

The feature extractor transforms raw text strings into **17 dense numerical features**:

1. **Sentence Morphological Metrics:** Average word length (`avg_word_len`).
2. **Inverted Punctuation Indicators:** Spanish upside-down question/exclamation marks (`¿`, `¡`).
3. **Orthographic Special Character Counters:**
   - Cedilla (`ç`) for Portuguese.
   - Eñe (`ñ`) for Spanish.
   - Tildes and diacritics (`ã`, `õ`, `â`, `ê`, `ô`, `á`, `é`, `í`, `ó`, `ú`).
4. **Morphological Suffix Pattern Matching:** Suffix matching (`-ção` for PT vs. `-ción` for ES).
5. **English Contractions:** Apostrophe contraction patterns (`\b\w+'\w+\b`).
6. **Language Article & Question Word Vectors:** Stopword frequencies for `el/la/los/las` (ES), `o/a/os/as` (PT), and `the/how/is/and` (EN).

---

## 📊 Model Benchmark & Accuracy Results

The dataset was partitioned using stratified sampling into **75% Training** and **25% Test** sets:

| Model Architecture | Training Accuracy | Test Accuracy | Precision (Macro) | Recall (Macro) | $F1$-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🟢 **Support Vector Machine (Linear SVM)** | **100.0%** | **96.0% - 100.0%** | **1.00** | **1.00** | **1.00** |

> 🎯 **Target Goal (>70% Accuracy):** Exceeded assignment benchmark by achieving **~96-100% test accuracy** across multi-language sentences.

---

## 📁 Repository Structure

```text
04-nlp-language-detection-feature-engineering/
├── README.md                                  <-- Technical Report & Documentation
├── notebooks/
│   └── nlp_language_classifier.ipynb          <-- Interactive Jupyter Notebook
└── src/
    └── language_detector.py                   <-- Modular Python NLP Classifier & Inference CLI
```

---

## 🚀 How to Run

### 1. Execute Inference CLI & Training Script
```bash
python src/language_detector.py
```

### 2. Run Interactive Notebook
```bash
jupyter notebook notebooks/nlp_language_classifier.ipynb
```

---
**Author:** Matheus Paixão  
**LinkedIn:** [Matheus Paixão](https://www.linkedin.com/in/matheus-paix%C3%A3o-5803b321b/?locale=pt)  
**GitHub:** [@MatheusPaixaoP](https://github.com/MatheusPaixaoP)
