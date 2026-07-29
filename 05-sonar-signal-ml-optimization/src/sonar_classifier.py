"""
Sonar Signal Classification & Dataset Optimization (UCI Sonar Dataset)
Author: Matheus Paixão
Domain: Machine Learning Engineering & Signal Processing

Optimizes Support Vector Machine (SVM) classifier performance on 60-band continuous
sonar return signals to distinguish between Metal Cylinders (Mines) and Rocks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

UCI_SONAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"


def load_sonar_data(url: str = UCI_SONAR_URL) -> tuple:
    """Loads UCI Sonar dataset and encodes target classes (R=0, M=1)."""
    print(f"[+] Fetching Sonar dataset from: {url}")
    df = pd.read_csv(url, header=None)
    
    X = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    y = y_raw.map({'R': 0, 'M': 1})
    
    print(f"[+] Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} frequency band features.")
    print(f"[+] Class Balance: {sum(y == 1)} Mines (M), {sum(y == 0)} Rocks (R).")
    return X, y


def build_feature_pipeline():
    """Constructs a Machine Learning pipeline with Scaling, Feature Selection, and SVM."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_select', SelectKBest(score_func=f_classif)),
        ('svm', SVC(random_state=42))
    ])
    
    param_grid = {
        'feature_select__k': [20, 30, 40, 50, 'all'],
        'svm__kernel': ['rbf', 'linear'],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    return pipeline, param_grid


def train_and_optimize():
    """Trains baseline SVM vs. Optimized Hyperparameter Grid Pipeline."""
    X, y = load_sonar_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 1. Baseline Unoptimized Model
    baseline_model = SVC(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"\n[!] Baseline Model Accuracy (Raw SVM): {baseline_acc * 100:.2f}%")
    
    # 2. Optimized Pipeline with Feature Selection & Grid Search CV
    pipeline, param_grid = build_feature_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("[+] Executing Grid Search Cross-Validation Tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    test_preds = best_model.predict(X_test)
    optimized_acc = accuracy_score(y_test, test_preds)
    
    print("\n" + "=" * 65)
    print(f"{'SONAR MODEL OPTIMIZATION PERFORMANCE SUMMARY':^65}")
    print("=" * 65)
    print(f"-> Baseline Accuracy : {baseline_acc * 100:.2f}%")
    print(f"-> Optimized Accuracy: {optimized_acc * 100:.2f}% (+{ (optimized_acc - baseline_acc) * 100:.2f}% Improvement)")
    print(f"-> Best Parameters   : {best_params}")
    print("-" * 65)
    print("\nClassification Report (Optimized Model):")
    print(classification_report(y_test, test_preds, target_names=['Rock (R)', 'Mine (M)']))
    print("=" * 65)


if __name__ == "__main__":
    train_and_optimize()
