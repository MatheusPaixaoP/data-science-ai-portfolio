# ==============================================================================
# Applied Statistics I: Exploratory Analysis & Hypothesis Testing
# Author: Matheus Paixão
# Domain: Applied Statistics, Econometrics & Inferential Analysis
# Stack: R, ggplot2, stats
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
})

# 1. Load Dataset
data_path <- "salarios.RData"
if (file.exists(data_path)) {
  load(data_path)
  cat("[+] Loaded dataset: salarios.RData\n")
} else {
  cat("[!] Dataset salarios.RData not found in working directory.\n")
}

# Helper function for Mode calculation
calculate_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# 2. Exploratory Visualizations Function
generate_plots <- function(df) {
  cat("[+] Generating Boxplots and Histograms...\n")
  
  # Comparative Boxplot
  boxplot(df$age, df$husage,
          names = c("Wife Age", "Husband Age"),
          main = "Age Distribution Comparison: Wife vs. Husband",
          ylab = "Age (Years)",
          col = c("lightblue", "lightgreen"))
          
  # Histograms
  par(mfrow = c(1, 2))
  hist(df$age, main = "Wife Age Histogram", xlab = "Age", col = "lightblue", border = "black")
  hist(df$husage, main = "Husband Age Histogram", xlab = "Age", col = "lightgreen", border = "black")
  par(mfrow = c(1, 1))
}

# 3. Descriptive Measures (Central Tendency & Dispersion)
compute_descriptive_stats <- function(df) {
  stats <- data.frame(
    Variable = c("Wife Age (age)", "Husband Age (husage)"),
    Mean = c(mean(df$age, na.rm = TRUE), mean(df$husage, na.rm = TRUE)),
    Median = c(median(df$age, na.rm = TRUE), median(df$husage, na.rm = TRUE)),
    Mode = c(calculate_mode(df$age), calculate_mode(df$husage)),
    Variance = c(var(df$age, na.rm = TRUE), var(df$husage, na.rm = TRUE)),
    Std_Dev = c(sd(df$age, na.rm = TRUE), sd(df$husage, na.rm = TRUE)),
    CV_Percent = c(
      (sd(df$age, na.rm = TRUE) / mean(df$age, na.rm = TRUE)) * 100,
      (sd(df$husage, na.rm = TRUE) / mean(df$husage, na.rm = TRUE)) * 100
    )
  )
  
  cat("\n==============================================================================\n")
  cat("                    DESCRIPTIVE STATISTICS SUMMARY                             \n")
  cat("==============================================================================\n")
  print(stats)
  cat("==============================================================================\n")
  return(stats)
}

# 4. Inferential Hypothesis Testing (Normality & Non-Parametric Difference)
run_hypothesis_tests <- function(df) {
  cat("\n[+] Running Inferential Hypothesis Tests...\n")
  set.seed(123)
  
  # Sample 5,000 observations for Shapiro-Wilk Normality Test
  sample_wife <- sample(df$age, min(5000, length(df$age)))
  sample_husband <- sample(df$husage, min(5000, length(df$husage)))
  
  shapiro_wife <- shapiro.test(sample_wife)
  shapiro_husband <- shapiro.test(sample_husband)
  
  cat(sprintf("-> Shapiro-Wilk Normality Test (Wife Age)   : W = %.4f, p-value = %.4e\n", shapiro_wife$statistic, shapiro_wife$p.value))
  cat(sprintf("-> Shapiro-Wilk Normality Test (Husband Age): W = %.4f, p-value = %.4e\n", shapiro_husband$statistic, shapiro_husband$p.value))
  
  # Mann-Whitney U Test (Wilcoxon Rank-Sum Test)
  mann_whitney <- wilcox.test(df$age, df$husage, conf.int = TRUE)
  cat(sprintf("-> Mann-Whitney U Test (Wife vs Husband)    : W = %.0f, p-value = %.4e\n", mann_whitney$statistic, mann_whitney$p.value))
  
  if (mann_whitney$p.value < 0.05) {
    cat("   [Result] Statistically significant median age difference between wives and husbands (p < 0.05).\n")
  }
}

if (exists("salarios")) {
  compute_descriptive_stats(salarios)
  run_hypothesis_tests(salarios)
}
