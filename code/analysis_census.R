# Load libraries
library(data.table)
library(MASS)      
library(dplyr)
library(sandwich)
library(stats)
library(Matrix)

# load proposed
source('./helper/functions_ols.R')
source('./helper/functions_ppi.R')
source('./helper/functions_reppi.R')
library(SuperLearner)
library(ranger)
library(kernlab)
library(xgboost)
source('./helper/functions.R')
source('./helper/superlearner.R')


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

# Prepare for simulation
num_trials <- 50
alpha <- 0.05

for (target_index in 1:3){
  # Fit a full model using sklearn-style logic
  ols_model <- lm(Y ~ X - 1)  # already full-rank with no intercept needed
  coeffs <- coef(ols_model)
  theta_true <- coeffs[target_index]
  
  # Simulation loop
  n <- as.integer(0.2 * n_all)
  length_record <- matrix(0, nrow = 6, ncol = num_trials)
  bias_record <- matrix(0, nrow = 6, ncol = num_trials)
  
  for (i in seq_len(num_trials)) {
    print(paste0(i," out of ", num_trials))
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
    length_record[1, i] <- (ci1$upper[target_index] - ci1$lower[target_index])
    bias_record[1, i] <- mean(c(ci1$lower[target_index], ci1$upper[target_index])) - theta_true
    
    # PPI
    ci2 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, lhat = 1)
    length_record[2, i] <- (ci2$upper[target_index] - ci2$lower[target_index])
    bias_record[2, i] <- mean(c(ci2$lower[target_index], ci2$upper[target_index])) - theta_true
    
    # PPI++
    ci3 <- ppi_ols_ci(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha)
    length_record[3, i] <- (ci3$upper[target_index] - ci3$lower[target_index])
    bias_record[3, i] <- mean(c(ci3$lower[target_index], ci3$upper[target_index])) - theta_true
    
    # REPPI
    ci4 <- ppi_opt_ols_ci_crossfit(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab, alpha = alpha, method = "linreg")
    length_record[4, i] <- (ci4$upper[target_index] - ci4$lower[target_index])
    bias_record[4, i] <- mean(c(ci4$lower[target_index], ci4$upper[target_index])) - theta_true
    
    # Proposed without ACP 
    source <- list('x' = as.matrix(X_lab[,2:3]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,2:3]))
    bb <- PPI(source = source, target = target, family = 'gaussian', bootstrap = TRUE)
    length_record[5, i] <- (apply(bb$beta, 2, quantile, probs = 0.975)[target_index] - 
                              apply(bb$beta, 2, quantile, probs = 0.025)[target_index])
    bias_record[5, i] <- mean(c(apply(bb$beta, 2, quantile, probs = 0.975)[target_index], 
                                apply(bb$beta, 2, quantile, probs = 0.025)[target_index])) - theta_true
    
    # Proposed with ACP 
    source <- list('x' = as.matrix(X_lab[,2:3]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,2:3]))
    bb <- PPI(source = source, target = target, sourcebb = as.vector(Yhat_lab), targetbb = as.vector(Yhat_unlab), 
              family = 'gaussian', bootstrap = TRUE)
    length_record[6, i] <- (apply(bb$beta, 2, quantile, probs = 0.975)[target_index] - 
                              apply(bb$beta, 2, quantile, probs = 0.025)[target_index])
    bias_record[6, i] <- mean(c(apply(bb$beta, 2, quantile, probs = 0.975)[target_index], 
                                apply(bb$beta, 2, quantile, probs = 0.025)[target_index])) - theta_true
  }
  rowMeans(length_record)
  apply(X=length_record, MARGIN=1, FUN=sd)
  rowMeans(bias_record)
  apply(X=bias_record, MARGIN=1, FUN=sd)
  
  save(length_record, file = paste0('../output/realdata/Census/CIlength', target_index, '.rda'))
  save(bias_record, file = paste0('../output/realdata/Census/Biaslength', target_index, '.rda'))
}

