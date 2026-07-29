# ==============================================================================
# Machine Learning Benchmark 03: Graduate Admission Chance Regression
# Author: Matheus Paixão
# Stack: R, Caret, Random Forest, SVM Radial, Neural Networks (nnet), KNN
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
data_path <- "9Admissao.csv"
if (file.exists(data_path)) {
  df <- read_csv(data_path)
  cat("[+] Loaded dataset: 9Admissao.csv\n")
} else {
  cat("[!] Dataset 9Admissao.csv not found in local directory. Generating synthetic benchmark space...\n")
  df <- data.frame(
    GRE_Score = rnorm(100, 315, 12),
    TOEFL_Score = rnorm(100, 107, 6),
    University_Rating = sample(1:5, 100, replace = TRUE),
    SOP = runif(100, 1, 5),
    LOR = runif(100, 1, 5),
    CGPA = runif(100, 6.8, 9.9),
    Research = sample(0:1, 100, replace = TRUE),
    Chance_of_Admit = runif(100, 0.34, 0.97)
  )
}

# Target column detection
target_col <- names(df)[ncol(df)]

# Evaluation metrics helper function
eval_regression_metrics <- function(actual, predicted) {
  r2_val     <- 1 - (sum((actual - predicted)^2) / sum((actual - mean(actual))^2))
  rmse_val   <- rmse(actual, predicted)
  mae_val    <- mae(actual, predicted)
  pearson_r  <- cor(actual, predicted)
  syx_val    <- sd(actual - predicted)
  return(c(R2 = r2_val, Syx = syx_val, Pearson_r = pearson_r, RMSE = rmse_val, MAE = mae_val))
}

# Split Data (75% Train / 25% Test for Hold-out)
trainIndex <- createDataPartition(df[[target_col]], p = 0.75, list = FALSE)
train_df   <- df[trainIndex, ]
test_df    <- df[-trainIndex, ]

ctrl_cv <- trainControl(method = "cv", number = 10)

cat("\n==============================================================================\n")
cat("            BENCHMARK 3: GRADUATE ADMISSION CHANCE REGRESSION                   \n")
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

# Summary Table
results_df <- data.frame(
  Model = c("Random Forest (10-Fold CV)", "SVM Radial (10-Fold CV)", "Neural Network (10-Fold CV)"),
  R2 = c(m_rf['R2'], m_svm['R2'], m_ann['R2']),
  RMSE = c(m_rf['RMSE'], m_svm['RMSE'], m_ann['RMSE']),
  MAE = c(m_rf['MAE'], m_svm['MAE'], m_ann['MAE']),
  Syx = c(m_rf['Syx'], m_svm['Syx'], m_ann['Syx']),
  Pearson_r = c(m_rf['Pearson_r'], m_svm['Pearson_r'], m_ann['Pearson_r'])
)

cat("\n==============================================================================\n")
cat("                      REGRESSION BENCHMARK SUMMARY SUMMARY                     \n")
cat("==============================================================================\n")
print(results_df)
cat("==============================================================================\n")
