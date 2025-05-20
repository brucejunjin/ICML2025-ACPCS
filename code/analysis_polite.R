# Load necessary libraries
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(MASS)
library(progress)

# load proposed
source('functions_ols.R')
source('functions_ppi.R')
source('functions_reppi.R')
library(SuperLearner)
library(ranger)
library(kernlab)
library(xgboost)
source('functions.R')
source('superlearner.R')


# Load dataset and shuffle
data <- fread("../data/Polite_data.csv") %>% na.omit() %>% sample_frac(1)
Yhat <- data$gpt_score
device <- "hedge"  # "hedge" or "1pp"
Y <- data[["Normalized Score"]]
n <- length(Y)

# Choose device-based feature
X_device <- if (device == "hedge") {
  as.matrix(data[, 5, with = FALSE])
} else {
  as.matrix(data[, 12, with = FALSE])
}
X <- as.matrix(cbind(1, X_device)) #data[, c(5, 13, 15, 18, 20, 21)]
colnames(X)[1] <- 'Intercept'

target_index <- 2

# Fit linear regression without intercept
model <- lm(Y ~ X - 1)
theta_true <- coef(model)[target_index]

# Simulation settings
num_trials <- 50
alpha <- 0.1
labeled_fracs <- seq(0.10, 0.30, length.out = 30)
n_all <- length(Y)

# assign the rate of 1 in labeled data
rt1 <- 0.7

for (j in seq_along(labeled_fracs)) {
  n <- as.integer(0.1 * n_all)
  cover_record <- matrix(0, nrow = 5, ncol = num_trials)
  bias_record <- matrix(0, nrow = 5, ncol = num_trials)
  
  for (i in 1:num_trials) {
    set.seed(2025 + i)
    print(c(i,j))
    ones_idx <- which(X[,2] == 1)
    zeros_idx <- which(X[,2] == 0)
    n_ones_labeled <- min(length(ones_idx), ceiling(n * rt1))  
    n_zeros_labeled <- n - n_ones_labeled 
    
    labeled <- c(sample(ones_idx, n_ones_labeled),sample(zeros_idx, n_zeros_labeled))
    unlabeled <- setdiff(1:n_all, labeled)
    
    X_lab <- X[labeled, , drop = FALSE]
    Y_lab <- as.matrix(Y[labeled])
    Yhat_lab <- as.matrix(Yhat[labeled])
    X_unlab <- X[unlabeled, , drop = FALSE]
    Yhat_unlab <- as.matrix(Yhat[unlabeled])
    
    # Classical OLS
    ci1 <- classical_ols_ci(X_lab, Y_lab, alpha = alpha)
    cover_record[1, i] <- (ci1$upper[target_index] - ci1$lower[target_index])
    bias_record[1, i] <- mean(c(ci1$lower[target_index], ci1$upper[target_index])) - theta_true
    
    # PPI
    ci2 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, lhat = 1)
    cover_record[2, i] <- (ci2$upper[target_index] - ci2$lower[target_index])
    bias_record[2, i] <- mean(c(ci2$lower[target_index], ci2$upper[target_index])) - theta_true
    
    # PPI++
    ci3 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha)
    cover_record[3, i] <- (ci3$upper[target_index] - ci3$lower[target_index])
    bias_record[3, i] <- mean(c(ci3$lower[target_index], ci3$upper[target_index])) - theta_true
    
    # REPPI
    ci4 <- ppi_opt_ols_ci_crossfit(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, method = "linreg")
    cover_record[4, i] <- (ci4$upper[target_index] - ci4$lower[target_index])
    bias_record[4, i] <- mean(c(ci4$lower[target_index], ci4$upper[target_index])) - theta_true
    
    # Proposed
    source <- list('x' = as.matrix(X_lab[,-1]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,-1]))
    bb <- PPI(source = source, target = target, sourcebb = as.vector(Yhat_lab), targetbb = as.vector(Yhat_unlab), 
              family = 'gaussian', bootstrap = TRUE)
    cover_record[5, i] <- (apply(bb$beta, 2, quantile, probs = 0.975)[target_index] - 
                             apply(bb$beta, 2, quantile, probs = 0.025)[target_index])
    bias_record[5, i] <- mean(c(apply(bb$beta, 2, quantile, probs = 0.975)[target_index], 
                                 apply(bb$beta, 2, quantile, probs = 0.025)[target_index])) - theta_true
  }
  rowMeans(cover_record)
  rowMeans(bias_record)
}

# Save or analyze results
results_summary <- results %>%
  group_by(estimator, n) %>%
  summarize(coverage = mean(coverage), avg_width = mean(width), avg_mse = mean(mse), .groups = "drop")

print(results_summary)