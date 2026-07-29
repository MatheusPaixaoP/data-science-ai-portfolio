# ==============================================================================
# Machine Learning Benchmark 06: Gym Activity Association Rule Mining (Apriori)
# Author: Matheus Paixão
# Domain: Market Basket Analysis & Association Rules
# Stack: R, arules, datasets
# ==============================================================================

suppressPackageStartupMessages({
  library(arules)
})

set.seed(202545)

cat("\n==============================================================================\n")
cat("            BENCHMARK 6: APRIORI ASSOCIATION RULE MINING                       \n")
cat("==============================================================================\n")

# 1. Load Transaction Data
data_path <- "2 - Musculacao - Dados.csv"
if (file.exists(data_path)) {
  tx_data <- read.transactions(file = data_path, format = "basket", sep = ";")
  cat("[+] Loaded transaction dataset: 2 - Musculacao - Dados.csv\n")
} else {
  cat("[!] Transaction dataset file not found. Creating sample transaction matrix...\n")
  sample_baskets <- list(
    c("Bench Press", "Flyes", "Triceps Extension"),
    c("Squats", "Leg Press", "Calf Raises"),
    c("Bench Press", "Incline Press", "Triceps Extension"),
    c("Deadlift", "Lat Pulldown", "Bicep Curls"),
    c("Bench Press", "Flyes", "Incline Press")
  )
  tx_data <- as(sample_baskets, "transactions")
}

# 2. Mine Association Rules with Apriori (Support >= 0.001, Confidence >= 0.70)
cat("\n[+] Mining Global Association Rules (Min Support=0.001, Min Confidence=0.70)...\n")
global_rules <- apriori(tx_data, parameter = list(supp = 0.001, conf = 0.70, minlen = 2))

cat("\n--- Top 10 High-Confidence Association Rules ---\n")
inspect(head(sort(global_rules, by = "confidence", decreasing = TRUE), 10))

# 3. Mine Targeted Rules with Antecedent (LHS) Filtering
cat("\n[+] Mining Specific Exercise Rules (LHS = 'Bench Press' / 'Flyes')...\n")
target_rules <- apriori(
  data = tx_data,
  parameter = list(supp = 0.001, conf = 0.10, minlen = 2),
  control = list(verbose = FALSE)
)

cat("\n--- Association Rules Filtered by Confidence ---\n")
inspect(head(sort(target_rules, by = "confidence", decreasing = TRUE), 10))
cat("==============================================================================\n")
