# ==============================================================================
# Forestry Timber Volume Estimation: Allometric Models vs Machine Learning
# Author: Matheus Paixão
# Domain: Forest Biometrics & Nonlinear Modeling
# Stack: R, caret, randomForest, e1071, nnet
# ==============================================================================

suppressPackageStartupMessages({
  library(caret)
  library(randomForest)
  library(e1071)
  library(nnet)
})

# 1. Load Data
dataset_url <- "http://www.razer.net.br/datasets/Volumes.csv"
cat("[+] Fetching dataset from:", dataset_url, "\n")
volumes <- read.csv2(dataset_url)

# Remove ID column if present
if ("NR" %in% colnames(volumes)) {
  volumes$NR <- NULL
}

cat("[+] Dataset Loaded. Dimensions:", nrow(volumes), "rows,", ncol(volumes), "columns.\n")

# 2. Data Partitioning (80% Train / 20% Test)
set.seed(42)
indices <- createDataPartition(volumes$VOL, p = 0.80, list = FALSE)
train_data <- volumes[indices, ]
test_data  <- volumes[-indices, ]

# 3. Custom Evaluation Metrics Functions
r2_score <- function(y_obs, y_pred) {
  ss_res <- sum((y_obs - y_pred)^2)
  ss_tot <- sum((y_obs - mean(y_obs))^2)
  return(1 - (ss_res / ss_tot))
}

syx_error <- function(y_obs, y_pred) {
  n <- length(y_obs)
  rss <- sum((y_obs - y_pred)^2)
  df <- max(1, n - 2)
  return(sqrt(rss / df))
}

syx_percentage <- function(syx_val, y_obs) {
  return((syx_val / mean(y_obs)) * 100)
}

# 4. Model Training & Inference

# A. Random Forest Regressor
rf_model <- caret::train(VOL ~ ., data = train_data, method = "rf")
pred_rf  <- predict(rf_model, test_data)

# B. Support Vector Machine (SVM Radial)
svm_model <- caret::train(VOL ~ ., data = train_data, method = "svmRadial")
pred_svm  <- predict(svm_model, test_data)

# C. Artificial Neural Network (ANN)
ann_model <- caret::train(VOL ~ ., data = train_data, method = "nnet", trace = FALSE)
pred_ann  <- predict(ann_model, test_data)

# D. Spurr Allometric Nonlinear Regression Model (Nonlinear Least Squares)
# Model Equation: Volume = b0 + b1 * (DBH^2) * Height
spurr_model <- nls(VOL ~ b0 + b1 * DAP * DAP * HT, data = train_data, start = list(b0 = 0.5, b1 = 0.5))
pred_spurr  <- predict(spurr_model, test_data)

# 5. Benchmark Performance Comparison
models <- list(
  "Spurr Allometric Model" = pred_spurr,
  "Random Forest"          = pred_rf,
  "SVM (Radial)"           = pred_svm,
  "Neural Network (ANN)"   = pred_ann
)

cat("\n==============================================================================\n")
cat("                FORESTRY TIMBER VOLUME ESTIMATION BENCHMARK                   \n")
cat("==============================================================================\n")

for (name in names(models)) {
  preds <- models[[name]]
  rmse_val <- RMSE(preds, test_data$VOL)
  r2_val   <- r2_score(test_data$VOL, preds)
  syx_val  <- syx_error(test_data$VOL, preds)
  syx_pct  <- syx_percentage(syx_val, test_data$VOL)
  
  cat(sprintf("Model: %-25s | RMSE: %.4f | R²: %.4f | Syx: %.4f | Syx%%: %.2f%%\n",
              name, rmse_val, r2_val, syx_val, syx_pct))
}
cat("==============================================================================\n")
