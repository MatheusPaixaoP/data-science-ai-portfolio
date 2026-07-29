# ==============================================================================
# Machine Learning Benchmark 02: Diabetes Onset Binary Classification
# Author: Matheus Paixão
# Stack: R, Caret, Random Forest, SVM Radial, Neural Networks (nnet), KNN
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(caTools)
  library(caret)
  library(class)
  library(nnet)
  library(e1071)
  library(randomForest)
  library(kernlab)
  library(MLmetrics)
})

set.seed(202545)

# 1. Load Dataset
data_path <- "4Diabetes.csv"
if (file.exists(data_path)) {
  df <- read_csv(data_path)
  df$class <- as.factor(df$class)
  cat("[+] Loaded dataset: 4Diabetes.csv\n")
} else {
  cat("[!] Dataset 4Diabetes.csv not found in local directory. Generating synthetic benchmark space...\n")
  df <- data.frame(
    pregnant = sample(0:10, 100, replace = TRUE),
    glucose = rnorm(100, 120, 30),
    pressure = rnorm(100, 70, 12),
    triceps = rnorm(100, 20, 8),
    insulin = rnorm(100, 80, 40),
    mass = rnorm(100, 32, 6),
    pedigree = runif(100, 0.1, 1.5),
    age = sample(21:65, 100, replace = TRUE),
    class = factor(sample(c("pos", "neg"), 100, replace = TRUE))
  )
}

# Split Data (75% Train / 25% Test for Hold-out evaluation)
sample_split <- sample.split(df$class, SplitRatio = 0.75)
train_df <- subset(df, sample_split == TRUE)
test_df  <- subset(df, sample_split == FALSE)

# 10-Fold Cross-Validation Control
ctrl_cv <- trainControl(method = "cv", number = 10, savePredictions = "final")

cat("\n==============================================================================\n")
cat("            BENCHMARK 2: DIABETES DIAGNOSIS BINARY CLASSIFICATION              \n")
cat("==============================================================================\n")

# 2. KNN Classifier (Hold-out k=5)
cat("\n[1] Training KNN Classifier (Hold-out, k=5)...\n")
X_train_knn <- train_df[, -ncol(train_df)]
X_test_knn  <- test_df[, -ncol(test_df)]
y_train_knn <- train_df$class
y_test_knn  <- test_df$class

preProc <- preProcess(X_train_knn, method = c("center", "scale"))
X_train_knn <- predict(preProc, X_train_knn)
X_test_knn  <- predict(preProc, X_test_knn)

knn_preds <- knn(train = X_train_knn, test = X_test_knn, cl = y_train_knn, k = 5)
knn_cm    <- confusionMatrix(knn_preds, y_test_knn)
cat(sprintf("-> KNN (Hold-out) Accuracy: %.4f (%.2f%%)\n", knn_cm$overall['Accuracy'], knn_cm$overall['Accuracy'] * 100))

# 3. Artificial Neural Network (nnet)
cat("\n[2] Training Artificial Neural Network (ANN nnet)...\n")
ann_holdout <- nnet(class ~ ., data = train_df, size = 5, decay = 0.01, MaxNWts = 10000, trace = FALSE)
ann_preds   <- predict(ann_holdout, newdata = test_df, type = "class")
ann_cm      <- confusionMatrix(as.factor(ann_preds), test_df$class)
cat(sprintf("-> ANN (Hold-out size=5, decay=0.01) Accuracy: %.4f (%.2f%%)\n", ann_cm$overall['Accuracy'], ann_cm$overall['Accuracy'] * 100))

# ANN 10-Fold CV
ann_grid <- expand.grid(.size = c(3, 5, 7), .decay = c(0.01, 0.1, 0.5))
ann_cv   <- train(class ~ ., data = df, method = "nnet", trControl = ctrl_cv, tuneGrid = ann_grid, MaxNWts = 10000, trace = FALSE)
cat(sprintf("-> ANN (10-Fold CV Best) Accuracy: %.4f (%.2f%%)\n", max(ann_cv$results$Accuracy), max(ann_cv$results$Accuracy) * 100))

# 4. Support Vector Machine (SVM Radial)
cat("\n[3] Training Support Vector Machine (SVM Radial)...\n")
svm_holdout <- svm(class ~ ., data = train_df, kernel = "radial", cost = 1, gamma = 0.1)
svm_preds   <- predict(svm_holdout, test_df)
svm_cm      <- confusionMatrix(svm_preds, test_df$class)
cat(sprintf("-> SVM Radial (Hold-out C=1, gamma=0.1) Accuracy: %.4f (%.2f%%)\n", svm_cm$overall['Accuracy'], svm_cm$overall['Accuracy'] * 100))

# SVM 10-Fold CV
svm_grid <- expand.grid(.C = c(0.1, 1, 10), .sigma = c(0.01, 0.1, 0.5))
svm_cv   <- train(class ~ ., data = df, method = "svmRadial", trControl = ctrl_cv, tuneGrid = svm_grid)
cat(sprintf("-> SVM Radial (10-Fold CV Best) Accuracy: %.4f (%.2f%%)\n", max(svm_cv$results$Accuracy), max(svm_cv$results$Accuracy) * 100))

# 5. Random Forest
cat("\n[4] Training Random Forest Classifier...\n")
mtry_val   <- floor(sqrt(ncol(train_df) - 1))
rf_holdout <- randomForest(class ~ ., data = train_df, mtry = mtry_val, ntree = 500)
rf_preds   <- predict(rf_holdout, test_df)
rf_cm      <- confusionMatrix(rf_preds, test_df$class)
cat(sprintf("-> Random Forest (Hold-out mtry=%d) Accuracy: %.4f (%.2f%%)\n", mtry_val, rf_cm$overall['Accuracy'], rf_cm$overall['Accuracy'] * 100))

# Random Forest 10-Fold CV
rf_grid <- expand.grid(.mtry = c(2, 4, 6))
rf_cv   <- train(class ~ ., data = df, method = "rf", trControl = ctrl_cv, tuneGrid = rf_grid)
cat(sprintf("-> Random Forest (10-Fold CV Best) Accuracy: %.4f (%.2f%%)\n", max(rf_cv$results$Accuracy), max(rf_cv$results$Accuracy) * 100))

cat("\n==============================================================================\n")
cat("                       BENCHMARK SUMMARY COMPLETE                             \n")
cat("==============================================================================\n")
