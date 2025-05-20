# Load libraries
library(data.table)
library(MASS)      
library(dplyr)
library(sandwich)
library(stats)
library(Matrix)

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


# Read and shuffle data
data <- read.csv("../data/Census.csv")
data$SEX <- ifelse(data$SEX == 1, 0, 1)

X <- as.matrix(cbind(1, data[,1:2]))
colnames(X)[1] <- 'Intercept'
Yhat <- as.vector(data[, 3])
Y <- as.vector(data[, 4])

n_all <- length(Y)

# Assign probability for cutting label and unlabel
eta <- c(0, 1, 0)
prob <- as.numeric(exp(X %*% eta)/(1+exp(X %*% eta)))

# Fit a full model using sklearn-style logic
ols_model <- lm(Y ~ X - 1)  # already full-rank with no intercept needed
coeffs <- coef(ols_model)
target_index <- 2  
theta_true <- coeffs[target_index]

# Prepare for simulation
num_trials <- 50
alpha <- 0.1
labeled_fracs <- seq(0.05,0.20, length.out = 30)

results <- list()
columns <- c("lb", "ub", "interval width", "coverage", "estimator", "n", "mse")

# Simulation loop
for (j in seq_along(labeled_fracs)) {
  n <- as.integer(labeled_fracs[j] * n_all)
  cover_record <- matrix(0, nrow = 5, ncol = num_trials)
  
  for (i in seq_len(num_trials)) {
    print(c(i,j))
    set.seed(2025 + i)
    labeled <- sample(order(prob,decreasing = T)[1:(2*n)], size = n)
    unlabeled <- setdiff(seq_len(n_all), labeled)
    
    X_lab <- X[labeled, , drop = FALSE]
    Y_lab <- as.matrix(Y[labeled])
    Yhat_lab <- as.matrix(Yhat[labeled])
    X_unlab <- X[unlabeled, , drop = FALSE]
    Yhat_unlab <- as.matrix(Yhat[unlabeled])
    
    # Classical OLS
    ci1 <- classical_ols_ci(X_lab, Y_lab, alpha = alpha)
    cover_record[1, i] <- (ci1$upper[target_index] - ci1$lower[target_index])
    
    # PPI
    ci2 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, lhat = 1)
    cover_record[2, i] <- (ci2$upper[target_index] - ci2$lower[target_index])
    
    # PPI++
    ci3 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha)
    cover_record[3, i] <- (ci3$upper[target_index] - ci3$lower[target_index])
    
    # REPPI
    ci4 <- ppi_opt_ols_ci_crossfit(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, method = "linreg")
    cover_record[4, i] <- (ci4$upper[target_index] - ci4$lower[target_index])
    
    # Proposed
    source <- list('x' = as.matrix(X_lab[,2:3]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,2:3]))
    bb <- PPI(source = source, target = target, sourcebb = as.vector(Yhat_lab), targetbb = as.vector(Yhat_unlab), 
              family = 'gaussian', bootstrap = TRUE)
    cover_record[5, i] <- (apply(bb$beta, 2, quantile, probs = 0.975)[target_index] - 
                             apply(bb$beta, 2, quantile, probs = 0.025)[target_index])
  }
  results[[j]] <- rowMeans(cover_record)
}

