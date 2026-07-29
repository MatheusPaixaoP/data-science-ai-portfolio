# ==============================================================================
# Multispectral Satellite Land Cover Classification
# Author: Matheus Paixão
# Domain: Remote Sensing & Environmental Machine Learning
# Stack: R, caret, randomForest, e1071, nnet, mlbench
# ==============================================================================

# 1. Load Required Libraries
suppressPackageStartupMessages({
  library(mlbench)
  library(caret)
  library(randomForest)
  library(e1071)
  library(nnet)
})

# 2. Load Satellite Dataset (Landsat MSS Voxel Neighborhoods)
data(Satellite)
cat("[+] Loaded Satellite Dataset: ", nrow(Satellite), "rows,", ncol(Satellite), "columns.\n")

# 3. Train/Test Data Partitioning (80% Train / 20% Test)
set.seed(42)
trainIndex <- createDataPartition(Satellite$classes, p = 0.8, list = FALSE)
trainData  <- Satellite[trainIndex, ]
testData   <- Satellite[-trainIndex, ]

cat("[+] Training set size: ", nrow(trainData), " | Test set size: ", nrow(testData), "\n")

# 4. Model Training & Evaluation

# --- Model 1: Random Forest Regressor/Classifier ---
cat("\n[+] Training Random Forest Classifier...\n")
rfModel <- randomForest(classes ~ ., data = trainData, importance = TRUE)
rfPred  <- predict(rfModel, testData)
rfConf  <- confusionMatrix(rfPred, testData$classes)

# --- Model 2: Support Vector Machine (SVM) ---
cat("[+] Training Support Vector Machine (SVM)...\n")
svmModel <- svm(classes ~ ., data = trainData)
svmPred  <- predict(svmModel, testData)
svmConf  <- confusionMatrix(svmPred, testData$classes)

# --- Model 3: Artificial Neural Network (ANN) ---
cat("[+] Training Artificial Neural Network (ANN)...\n")
nnModel <- nnet(classes ~ ., data = trainData, size = 10, maxit = 200, trace = FALSE)
nnPred  <- predict(nnModel, testData, type = "class")
nnPred  <- factor(nnPred, levels = levels(testData$classes))
nnConf  <- confusionMatrix(nnPred, testData$classes)

# 5. Benchmark Performance Comparison
cat("\n==============================================================================\n")
cat("                CLASSIFICATION ACCURACY BENCHMARK SUMMARY                     \n")
cat("==============================================================================\n")
cat(sprintf("-> Random Forest Accuracy: %.4f (%.2f%%)\n", rfConf$overall["Accuracy"], rfConf$overall["Accuracy"] * 100))
cat(sprintf("-> SVM Accuracy          : %.4f (%.2f%%)\n", svmConf$overall["Accuracy"], svmConf$overall["Accuracy"] * 100))
cat(sprintf("-> Neural Network (ANN)  : %.4f (%.2f%%)\n", nnConf$overall["Accuracy"], nnConf$overall["Accuracy"] * 100))
cat("==============================================================================\n")
