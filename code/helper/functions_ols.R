# Load required packages
library(sandwich)  # for robust SEs
library(lmtest)    # not strictly needed here, but often used with sandwich

# OLS with robust SE
ols <- function(X, Y, return_se = FALSE) {
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("x", seq_len(ncol(X)))
  }
  
  data <- data.frame(Y = Y, X)
  formula <- as.formula(paste("Y ~", paste(colnames(X), collapse = " + "), "-1"))
  model <- lm(formula, data = data)
  theta <- coef(model)
  theta <- as.numeric(theta)
  
  if (return_se) {
    se <- sqrt(diag(sandwich::vcovHC(model, type = "HC0")))
    return(list(theta = theta, se = se))
  } else {
    return(list(theta = theta))
  }
}

# WLS with robust SE
wls <- function(X, Y, w = NULL, return_se = FALSE) {
  if (is.null(w) || all(w == 1)) {
    return(ols(X, Y, return_se = return_se))
  }
  
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("x", seq_len(ncol(X)))
  }
  
  n <- length(Y)
  w <- w / sum(w) * n  # normalize weights
  
  data <- data.frame(Y = Y, X)
  formula <- as.formula(paste("Y ~", paste(colnames(X), collapse = " + "), "-1"))
  model <- lm(formula, data = data, weights = w)
  theta <- coef(model)
  theta <- as.numeric(theta)
  
  if (return_se) {
    se <- sqrt(diag(sandwich::vcovHC(model, type = "HC0")))
    return(list(theta = theta, se = se))
  } else {
    return(list(theta = theta))
  }
}

# Z-based confidence interval
zconfint_generic <- function(est, se, alpha = 0.1, alternative = "two-sided") {
  if (alternative == "two-sided") {
    z <- qnorm(1 - alpha / 2)
    lower <- est - z * se
    upper <- est + z * se
  } else if (alternative == "less") {
    z <- qnorm(alpha)
    lower <- rep(-Inf, length(est))
    upper <- est + z * se
  } else if (alternative == "greater") {
    z <- qnorm(1 - alpha)
    lower <- est - z * se
    upper <- rep(Inf, length(est))
  } else {
    stop("alternative must be one of 'two-sided', 'less', 'greater'")
  }
  return(list(lower = lower, upper = upper))
}


# Main function
classical_mean_ci <- function(Y, w = NULL, alpha = 0.1, alternative = "two-sided") {
  n <- length(Y)
  
  if (is.null(w)) {
    mu <- mean(Y)
    se <- sd(Y) / sqrt(n)
  } else {
    w <- w / sum(w) * n
    mu <- sum(w * Y) / n
    se <- sqrt(sum(w * (Y - mu)^2) / (n - 1)) / sqrt(n)
  }
  
  zconfint_generic(mu, se, alpha, alternative)
}


classical_ols_ci <- function(X, Y, w = NULL, alpha = 0.1, alternative = "two-sided") {
  n <- length(Y)
  if (is.null(w)) {
    res <- ols(X, Y, return_se = TRUE)
  } else {
    w <- w / sum(w) * n
    res <- wls(X, Y, w = w, return_se = TRUE)
  }
  return(zconfint_generic(res$theta, res$se, alpha, alternative))
}


classical_logistic_ci <- function(X, Y, alpha = 0.1, alternative = "two-sided") {
  # Fit logistic regression
  pointest <- logistic(X, Y)
  
  n <- nrow(X)
  d <- ncol(X)
  eta <- X %*% pointest
  mu <- 1 / (1 + exp(-eta))  # expit
  
  # Compute V (Fisher Information)
  V <- matrix(0, nrow = d, ncol = d)
  grads <- matrix(0, nrow = n, ncol = d)
  for (i in 1:n) {
    xi <- matrix(X[i, ], nrow = 1)
    V <- V + (mu[i] * (1 - mu[i])) * t(xi) %*% xi / n
    grads[i, ] <- (mu[i] - Y[i]) * X[i, ]
  }
  
  V_inv <- solve(V)
  cov_mat <- V_inv %*% cov(grads) %*% V_inv
  
  se <- sqrt(diag(cov_mat) / n)
  
  # Compute confidence intervals
  ci_bounds <- zconfint_generic(pointest, se, alpha, alternative)
  return(ci_bounds)
}

# Logistic regression helper
logistic <- function(X, Y) {
  df <- data.frame(Y = as.factor(Y), X)
  model <- glm(Y ~ . -1, data = df, family = binomial())  # -1 to exclude intercept
  return(coef(model))
}