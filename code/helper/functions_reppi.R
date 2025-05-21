# Load required packages
library(sandwich)  # for robust SEs
library(lmtest)    # not strictly needed here, but often used with sandwich

sample_split <- function(n) {
  index <- sample(n)
  third <- floor(n / 3)
  index1 <- index[1:third]
  index2 <- index[(third + 1):(2 * third)]
  index3 <- index[(2 * third + 1):n]
  return(list(index1 = index1, index2 = index2, index3 = index3))
}

sample_split_logistic <- function(n, Y) {
  stopifnot(length(Y) == n)
  
  # Create two strata: Y == 1 and Y == 0
  idx_pos <- which(Y == 1)
  idx_neg <- which(Y == 0)
  
  # Shuffle each group separately
  idx_pos <- sample(idx_pos)
  idx_neg <- sample(idx_neg)
  
  # Helper to divide indices into three roughly equal parts
  split_into_three <- function(indices) {
    total <- length(indices)
    third <- floor(total / 3)
    extra <- total %% 3
    sizes <- rep(third, 3)
    if (extra > 0) sizes[1:extra] <- sizes[1:extra] + 1
    split_indices <- split(indices, rep(1:3, times = sizes))
    return(split_indices)
  }
  
  # Split positive and negative separately
  pos_split <- split_into_three(idx_pos)
  neg_split <- split_into_three(idx_neg)
  
  # Combine each stratum
  index1 <- c(pos_split[[1]], neg_split[[1]])
  index2 <- c(pos_split[[2]], neg_split[[2]])
  index3 <- c(pos_split[[3]], neg_split[[3]])
  
  # Shuffle each combined group
  return(list(
    index1 = sample(index1),
    index2 = sample(index2),
    index3 = sample(index3)
  ))
}

grad_fit_ols <- function(X, Y, Yhat, theta, r, method = "linreg") {
  covariates <- cbind(X, Yhat)
  Y <- drop(Y)  # Ensure Y is a vector
  
  # Model fitting
  if (method == "linreg") {
    df <- data.frame(Y = Y, covariates)
    model <- lm(Y ~ . -1, data = df)
  } else if (method == "logistic") {
    df <- data.frame(Y = factor(Y), covariates)
    model <- glm(Y ~ . -1, data = df, family = binomial())
  } else {
    stop(paste("Method", method, "not yet supported in R version."))
  }
  
  # Return a gradient-generating function
  f <- function(X_new, Yhat_new) {
    new_covariates <- data.frame(cbind(X_new, Yhat_new))
    colnames(new_covariates) <- names(coef(model))
    
    if (method == "logistic") {
      pred <- predict(model, newdata = new_covariates, type = "response")  # probabilities
    } else {
      pred <- predict(model, newdata = new_covariates)
    }
    
    theta_no_intercept <- theta  # keep consistent with Python version
    res <- X_new %*% matrix(theta_no_intercept, ncol = 1) - pred
    return((1 / (1 + r)) * sweep(X_new, 1, res, `*`))
  }
  
  return(f)
}

ppi_opt_ols_pointestimate <- function(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    grad,         # gradient function from grad_fit_ols
    theta_0,      # initial OLS estimate
    w = NULL,
    w_unlabeled = NULL
) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  r <- n / N
  
  # Normalize weights
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  # Gradient matrices
  grad_unlabeled_g <- grad(X_unlabeled, Yhat_unlabeled)
  grad_labeled_g <- grad(X, Yhat)
  
  theta_0 <- as.vector(theta_0)
  theta_no_intercept <- theta_0  # drop intercept
  resid_labeled <- X %*% matrix(theta_no_intercept, ncol = 1) - Y
  grad_labeled_l <- sweep(X, 1, resid_labeled, `*`)
  
  # Covariance matrices
  cov_both <- cov(cbind(grad_labeled_l, grad_labeled_g))
  cov_label_unlabel <- as.matrix(cov_both)
  cov_cross <- cov_label_unlabel[1:d, (d + 1):(2 * d)]
  cov_model <- cov_label_unlabel[(d + 1):(2 * d), (d + 1):(2 * d)]
  
  # Correction matrix M
  M <- (1 / (1 + r)) * cov_cross %*% solve(cov_model)
  
  # Mean gradients (weighted)
  grad_unlabeled <- colMeans(grad_unlabeled_g * w_unlabeled)
  grad_labeled <- colMeans(grad_labeled_g * w)
  
  # Inverse (weighted) X'X
  W <- diag(w)
  Sigma_inv <- solve(t(X) %*% W %*% X / n)
  
  # Corrected estimator
  theta <- Sigma_inv %*% ((t(X) %*% W %*% Y) / n + M %*% matrix(grad_labeled - grad_unlabeled, ncol = 1))
  
  return(list(theta = theta, M = M))
}

ppi_opt_ols_pointestimate_crossfit <- function(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    w = NULL,
    w_unlabeled = NULL,
    method = "linreg",
    return_grad = FALSE
) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  r <- n / N
  
  # Normalize weights
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  # Make sure Y, Yhat, Yhat_unlabeled are 2D column vectors
  Y <- reshape_to_2d(Y)
  Yhat <- reshape_to_2d(Yhat)
  Yhat_unlabeled <- reshape_to_2d(Yhat_unlabeled)
  
  # Split indices
  if (method != 'logistic'){
    idx_split <- sample_split(n)
    index1 <- idx_split$index1
    index2 <- idx_split$index2
    index3 <- idx_split$index3
  } else {
    idx_split <- sample_split_logistic(n, Y)
    index1 <- idx_split$index1
    index2 <- idx_split$index2
    index3 <- idx_split$index3
  }
  
  
  # WLS on each subset
  theta_1 <- wls(X[index1, ], Y[index1, ], w = w[index1])$theta
  theta_2 <- wls(X[index2, ], Y[index2, ], w = w[index2])$theta
  theta_3 <- wls(X[index3, ], Y[index3, ], w = w[index3])$theta
  
  # Gradient estimators
  grad_g_1 <- grad_fit_ols(X[index2, ], Y[index2, ], Yhat[index2, ], theta_1, r, method = method)
  grad_g_2 <- grad_fit_ols(X[index3, ], Y[index3, ], Yhat[index3, ], theta_2, r, method = method)
  grad_g_3 <- grad_fit_ols(X[index1, ], Y[index1, ], Yhat[index1, ], theta_3, r, method = method)
  
  # Bias-corrected point estimates
  est_1 <- ppi_opt_ols_pointestimate(X[index3, ], Y[index3, ], Yhat[index3, ],
                                     X_unlabeled, Yhat_unlabeled, grad_g_1, theta_1,
                                     w = w[index3], w_unlabeled = w_unlabeled)
  est_2 <- ppi_opt_ols_pointestimate(X[index1, ], Y[index1, ], Yhat[index1, ],
                                     X_unlabeled, Yhat_unlabeled, grad_g_2, theta_2,
                                     w = w[index1], w_unlabeled = w_unlabeled)
  est_3 <- ppi_opt_ols_pointestimate(X[index2, ], Y[index2, ], Yhat[index2, ],
                                     X_unlabeled, Yhat_unlabeled, grad_g_3, theta_3,
                                     w = w[index2], w_unlabeled = w_unlabeled)
  
  theta_avg <- (est_1$theta + est_2$theta + est_3$theta) / 3
  
  if (return_grad) {
    # Gradients on unlabeled set
    grad_unlabeled_1 <- grad_g_1(X_unlabeled, Yhat_unlabeled) %*% t(est_1$M)
    grad_unlabeled_2 <- grad_g_2(X_unlabeled, Yhat_unlabeled) %*% t(est_2$M)
    grad_unlabeled_3 <- grad_g_3(X_unlabeled, Yhat_unlabeled) %*% t(est_3$M)
    grad_unlabeled <- (grad_unlabeled_1 + grad_unlabeled_2 + grad_unlabeled_3) / 3
    
    # Gradients on labeled sets
    grad_labeled_1 <- grad_g_1(X[index3, ], Yhat[index3, ]) %*% t(est_1$M)
    grad_labeled_2 <- grad_g_2(X[index1, ], Yhat[index1, ]) %*% t(est_2$M)
    grad_labeled_3 <- grad_g_3(X[index2, ], Yhat[index2, ]) %*% t(est_3$M)
    grad_labeled <- rbind(grad_labeled_1, grad_labeled_2, grad_labeled_3)
    
    index_labeled <- c(index3, index1, index2)
    
    return(list(
      theta = theta_avg,
      grad_labeled = grad_labeled,
      index_labeled = index_labeled,
      grad_unlabeled = grad_unlabeled
    ))
  } else {
    return(theta_avg)
  }
}

ppi_opt_ols_ci_crossfit <- function(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha = 0.1,
    alternative = "two-sided",
    w = NULL,
    w_unlabeled = NULL,
    method = "linreg"
) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  
  # Normalize weights
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  # Get cross-fitted estimate and gradients
  ppi_res <- ppi_opt_ols_pointestimate_crossfit(
    X = X,
    Y = Y,
    Yhat = Yhat,
    X_unlabeled = X_unlabeled,
    Yhat_unlabeled = Yhat_unlabeled,
    w = w,
    w_unlabeled = w_unlabeled,
    method = method,
    return_grad = TRUE
  )
  
  ppi_opt_pointest <- ppi_res$theta
  grads_g_labeled <- ppi_res$grad_labeled
  index_labeled <- ppi_res$index_labeled
  grads_g_unlabeled <- ppi_res$grad_unlabeled
  
  # Get standard OLS stats
  stats <- ols_get_stats(
    pointest = ppi_opt_pointest,
    X = X,
    Y = Y,
    Yhat = Yhat,
    X_unlabeled = X_unlabeled,
    Yhat_unlabeled = Yhat_unlabeled,
    w = w,
    w_unlabeled = w_unlabeled,
    use_unlabeled = TRUE
  )
  grads <- stats$grads
  inv_hessian <- stats$inv_hessian
  
  # Weighted covariance: grads_g_unlabeled
  var_unlabeled <- cov.wt(grads_g_unlabeled, wt = w_unlabeled)$cov
  
  # Weighted covariance of differences on labeled data
  diffs_labeled <- grads[index_labeled, ] - grads_g_labeled
  var_labeled <- cov.wt(diffs_labeled, wt = w)$cov
  
  # Combined covariance
  Sigma_hat <- inv_hessian %*% ((n / N) * var_unlabeled + var_labeled) %*% inv_hessian
  se <- sqrt(diag(Sigma_hat) / n)
  
  # Confidence interval
  return(zconfint_generic(
    est = ppi_opt_pointest,
    se = se,
    alpha = alpha,
    alternative = alternative
  ))
}