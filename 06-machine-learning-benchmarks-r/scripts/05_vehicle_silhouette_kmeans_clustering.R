# ==============================================================================
# Machine Learning Benchmark 05: Vehicle Silhouette K-Means Clustering
# Author: Matheus Paixão
# Domain: Unsupervised Learning & Feature Space Clustering
# Stack: R, stats, klaR
# ==============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

set.seed(202545)

# 1. Load Dataset
data_path <- "6Veiculos.csv"
if (file.exists(data_path)) {
  df <- read_csv(data_path)
  cat("[+] Loaded dataset: 6Veiculos.csv\n")
} else {
  cat("[!] Dataset 6Veiculos.csv not found in local directory. Generating synthetic cluster space...\n")
  df <- data.frame(
    comp = rnorm(100, 90, 10), circ = rnorm(100, 45, 5),
    d_circ = rnorm(100, 80, 8), rad_ratio = rnorm(100, 160, 20),
    pr_axis_ratio = rnorm(100, 60, 6), max_length_aspect = rnorm(100, 8, 2),
    tipo = factor(sample(c("opel", "saab", "bus", "van"), 100, replace = TRUE))
  )
}

# Remove index or label column if present
if ("tipo" %in% names(df)) {
  label_col <- df$tipo
  df_num <- df %>% dplyr::select(-tipo)
} else {
  df_num <- df
}

# 2. Z-Score Feature Standardization
df_scaled <- scale(df_num)

cat("\n==============================================================================\n")
cat("            BENCHMARK 5: UNSUPERVISED K-MEANS CLUSTERING (k=10)                \n")
cat("==============================================================================\n")

# 3. K-Means Algorithm (k=10 clusters, 25 random starts)
kmeans_res <- kmeans(df_scaled, centers = 10, nstart = 25)

clustered_df <- cbind(df, Cluster_ID = kmeans_res$cluster)

cat(sprintf("[+] Clustering Complete. Total Within-Cluster Sum of Squares: %.2f\n", kmeans_res$tot.withinss))
cat("\nFirst 10 Observations with Assigned Cluster Centroids:\n")
print(head(clustered_df, 10))
cat("==============================================================================\n")
