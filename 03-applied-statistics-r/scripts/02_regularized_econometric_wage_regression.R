# ==============================================================================
# Applied Statistics II: Regularized Econometric Wage Regression (Ridge/Lasso/ElasticNet)
# Author: Matheus Paixão
# Domain: Econometrics, Wage Analytics & Regularized Regression
# Stack: R, glmnet, caret, dplyr, Metrics, tidyverse
# ==============================================================================

suppressPackageStartupMessages({
  library(glmnet)
  library(caret)
  library(dplyr)
  library(Metrics)
  library(tidyverse)
})

# 1. Load Dataset
data_path <- "trabalhosalarios.RData"
if (file.exists(data_path)) {
  load(data_path)
  cat("[+] Loaded dataset: trabalhosalarios.RData\n")
} else {
  cat("[!] Dataset trabalhosalarios.RData not found in working directory.\n")
}

# 2. Data Preprocessing & Model Execution
run_econometric_pipeline <- function(df) {
  # Feature Matrix X and Target Vector y (log hourly wage)
  X <- as.matrix(df %>% dplyr::select(-lwage, -earns))
  y <- df$lwage
  
  # Train/Test Split (80% Train / 20% Test)
  set.seed(42)
  trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[trainIndex, ]
  y_train <- y[trainIndex]
  X_test  <- X[-trainIndex, ]
  y_test  <- y[-trainIndex]
  
  # Fit Models (Ridge alpha=0, Lasso alpha=1, ElasticNet alpha=0.5)
  ridge_model      <- glmnet(X_train, y_train, alpha = 0)
  lasso_model      <- glmnet(X_train, y_train, alpha = 1)
  elasticnet_model <- glmnet(X_train, y_train, alpha = 0.5)
  
  # Predict on test set
  lambda_val <- 0.1
  ridge_pred      <- predict(ridge_model, s = lambda_val, newx = X_test)
  lasso_pred      <- predict(lasso_model, s = lambda_val, newx = X_test)
  elasticnet_pred <- predict(elasticnet_model, s = lambda_val, newx = X_test)
  
  # Compute Evaluation Metrics
  calc_metrics <- function(true_val, pred_val) {
    rmse_val <- sqrt(mean((true_val - pred_val)^2))
    r2_val   <- 1 - (sum((true_val - pred_val)^2) / sum((true_val - mean(true_val))^2))
    return(c(RMSE = rmse_val, R2 = r2_val))
  }
  
  m_ridge      <- calc_metrics(y_test, ridge_pred)
  m_lasso      <- calc_metrics(y_test, lasso_pred)
  m_elasticnet <- calc_metrics(y_test, elasticnet_pred)
  
  metrics_df <- data.frame(
    Model = c("Ridge (alpha=0)", "Lasso (alpha=1)", "ElasticNet (alpha=0.5)"),
    RMSE = c(m_ridge[1], m_lasso[1], m_elasticnet[1]),
    R2_Score = c(m_ridge[2], m_lasso[2], m_elasticnet[2])
  )
  
  cat("\n==============================================================================\n")
  cat("                REGULARIZED REGRESSION MODEL BENCHMARK                        \n")
  cat("==============================================================================\n")
  print(metrics_df)
  cat("==============================================================================\n")
  
  # Target Profile Inference & Non-Parametric Bootstrap Confidence Interval
  target_profile <- data.frame(
    husage = 40, husunion = 0, husearns = 600, huseduc = 13,
    husblck = 1, hushisp = 0, hushrs = 40, kidge6 = 1, age = 38,
    black = 0, educ = 13, hispanic = 1, union = 0, exper = 18, kidlt6 = 1
  )
  
  # Anti-log Point Predictions ($/hour)
  ridge_wage      <- exp(predict(ridge_model, s = lambda_val, newx = as.matrix(target_profile)))
  lasso_wage      <- exp(predict(lasso_model, s = lambda_val, newx = as.matrix(target_profile)))
  elasticnet_wage <- exp(predict(elasticnet_model, s = lambda_val, newx = as.matrix(target_profile)))
  
  cat(sprintf("\n-> Predicted Hourly Wage (Ridge)     : $%.2f / hour\n", ridge_wage))
  cat(sprintf("-> Predicted Hourly Wage (Lasso)     : $%.2f / hour\n", lasso_wage))
  cat(sprintf("-> Predicted Hourly Wage (ElasticNet): $%.2f / hour\n", elasticnet_wage))
  
  return(metrics_df)
}

if (exists("trabalhosalarios")) {
  run_econometric_pipeline(trabalhosalarios)
}
