# ==============================================================================
# Machine Learning Benchmark 04: Forest Biomass Continuous Regression
# Author: Matheus Paixão
# Domain: Environmental Machine Learning & Forestry Biometrics
# Stack: R, Caret, Random Forest, SVM Radial, Neural Networks (nnet)
# Metrics: R², RMSE, MAE, Syx, Pearson Correlation r
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(caTools)
  library(caret)
  library(e1071)
  library(randomForest)
  library(kernlab)
  library(Metrics)
})

set.seed(202545)

# 1. Load Dataset
data_path <- "7Biomassa.csv"
if (file.exists(data_path)) {
  df <- read_csv(data_path)
  cat("[+] Loaded dataset: 7Biomassa.csv\n")
} else {
  cat("[!] Dataset 7Biomassa.csv not found in local directory. Generating synthetic benchmark space...\n")
  df <- data.frame(
    dap = rnorm(100, 22.5, 5.0),
    ht = rnorm(100, 18.0, 3.5),
    volume = rnorm(100, 0.45, 0.15),
    biomass_ton = rnorm(100, 120.0, 35.0)
  )
}

target_col <- names(df)[ncol(df)]

# Metric Evaluation
eval_regression_metrics <- function(actual, predicted) {
  r2_val     <- 1 - (sum((actual - predicted)^2) / sum((actual - mean(actual))^2))
  rmse_val   <- rmse(actual, predicted)
  mae_val    <- mae(actual, predicted)
  pearson_r  <- cor(actual, predicted)
  syx_val    <- sd(actual - predicted)
  return(c(R2 = r2_val, Syx = syx_val, Pearson_r = pearson_r, RMSE = rmse_val, MAE = mae_val))
}

trainIndex <- createDataPartition(df[[target_col]], p = 0.75, list = FALSE)
train_df   <- df[trainIndex, ]
test_df    <- df[-trainIndex, ]

ctrl_cv <- trainControl(method = "cv", number = 10)

cat("\n==============================================================================\n")
cat("            BENCHMARK 4: ENVIRONMENTAL FOREST BIOMASS REGRESSION               \n")
cat("==============================================================================\n")

# 1. Random Forest Regressor (10-Fold CV)
cat("\n[1] Training Random Forest Regressor (10-Fold CV)...\n")
rf_cv <- train(as.formula(paste(target_col, "~ .")), data = df, method = "rf", trControl = ctrl_cv)
rf_preds <- predict(rf_cv, test_df)
m_rf <- eval_regression_metrics(test_df[[target_col]], rf_preds)

# 2. SVM Radial Regressor (10-Fold CV)
cat("\n[2] Training SVM Radial Regressor (10-Fold CV)...\n")
svm_cv <- train(as.formula(paste(target_col, "~ .")), data = df, method = "svmRadial", trControl = ctrl_cv)
svm_preds <- predict(svm_cv, test_df)
m_svm <- eval_regression_metrics(test_df[[target_col]], svm_preds)

# 3. Neural Network Regressor (10-Fold CV)
cat("\n[3] Training Neural Network Regressor (10-Fold CV)...\n")
ann_cv <- train(as.formula(paste(target_col, "~ .")), data = df, method = "nnet", linout = TRUE, trace = FALSE, trControl = ctrl_cv)
ann_preds <- predict(ann_cv, test_df)
m_ann <- eval_regression_metrics(test_df[[target_col]], ann_preds)

results_df <- data.frame(
  Model = c("Random Forest (10-Fold CV)", "SVM Radial (10-Fold CV)", "Neural Network (10-Fold CV)"),
  R2 = c(m_rf['R2'], m_svm['R2'], m_ann['R2']),
  RMSE = c(m_rf['RMSE'], m_svm['RMSE'], m_ann['RMSE']),
  MAE = c(m_rf['MAE'], m_svm['MAE'], m_ann['MAE']),
  Syx = c(m_rf['Syx'], m_svm['Syx'], m_ann['Syx']),
  Pearson_r = c(m_rf['Pearson_r'], m_svm['Pearson_r'], m_ann['Pearson_r'])
)

cat("\n==============================================================================\n")
cat("                      REGRESSION BENCHMARK SUMMARY                             \n")
cat("==============================================================================\n")
print(results_df)
cat("==============================================================================\n")
