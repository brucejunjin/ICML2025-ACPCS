# Load required libraries
library(dplyr)
library(readr)
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
data <- read_csv("./data/US_wine.csv") %>% sample_frac(1)

# Extract prediction and ground truth
Yhat <- pmax(data$gpt_point, 80)
Y <- data$points
n_all <- length(Y)

# Add dummy variables for province
data <- data %>%
  mutate(
    is_ca = as.integer(province == "California"),
    is_wa = as.integer(province == "Washington"),
    is_or = as.integer(province == "Oregon"),
    is_ny = as.integer(province == "New York")
  )

# Build feature matrix
X <- cbind(1,
  log(data$price),
  data$is_ca,
  data$is_wa,
  data$is_or,
  data$is_ny
)

# Assign probability for cutting label and unlabel
eta <- c(0, 1, 0, 0, 0, 0)
prob <- as.numeric(exp(X %*% eta)/(1+exp(X %*% eta)))

# Prepare for simulation
num_trials <- 50
alpha <- 0.05

for (target_index in 1:6){
  
  # Fit a full model using sklearn-style logic
  ols_model <- lm(Y ~ X - 1)  # already full-rank with no intercept needed
  coeffs <- coef(ols_model)
  theta_true <- coeffs[target_index]
  
  # Simulation loop
  n <- as.integer(0.3 * n_all)
  length_record <- matrix(0, nrow = 6, ncol = num_trials)
  bias_record <- matrix(0, nrow = 6, ncol = num_trials)
  
  for (i in seq_len(num_trials)) {
    set.seed(2025 + i)
    print(paste0(i," out of ", num_trials))
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
    source <- list('x' = as.matrix(X_lab[,2:6]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,2:6]))
    bb <- PPI(source = source, target = target, family = 'gaussian', bootstrap = TRUE)
    length_record[5, i] <- (apply(bb$beta, 2, quantile, probs = 0.975)[target_index] - 
                              apply(bb$beta, 2, quantile, probs = 0.025)[target_index])
    bias_record[5, i] <- mean(c(apply(bb$beta, 2, quantile, probs = 0.975)[target_index], 
                                apply(bb$beta, 2, quantile, probs = 0.025)[target_index])) - theta_true
    
    # Proposed with ACP 
    source <- list('x' = as.matrix(X_lab[,2:6]), 'y' = as.vector(Y_lab))
    target <- list('x' = as.matrix(X_unlab[,2:6]))
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
  
  save(length_record, file = paste0('./output/realdata/Wine/CIlength', target_index, '.rda'))
  save(bias_record, file = paste0('./output/realdata/Wine/Biaslength', target_index, '.rda'))
}

