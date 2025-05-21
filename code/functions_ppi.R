# Load required packages
library(sandwich)  # for robust SEs
library(lmtest)    # not strictly needed here, but often used with sandwich

# Helper: reshape to 2D column vector or matrix
reshape_to_2d <- function(x) {
  if (is.null(dim(x))) {
    return(matrix(x, ncol = 1))
  } else {
    return(as.matrix(x))
  }
}

# Helper: construct normalized weight vector (sum = n_obs)
construct_weight_vector <- function(n_obs, existing_weight, vectorized = FALSE) {
  if (is.null(existing_weight)) {
    res <- rep(1, n_obs)
  } else {
    res <- existing_weight / sum(existing_weight) * n_obs
  }
  if (vectorized && is.null(dim(res))) {
    res <- matrix(res, ncol = 1)
  }
  return(res)
}

# Helper: calculate lhat
calc_lhat_glm <- function(grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord = NULL, clip = FALSE, optim_mode = "overall") {
  grads <- reshape_to_2d(grads)
  grads_hat <- reshape_to_2d(grads_hat)
  grads_hat_unlabeled <- reshape_to_2d(grads_hat_unlabeled)
  
  n <- nrow(grads)
  N <- nrow(grads_hat_unlabeled)
  d <- ncol(inv_hessian)
  
  if (ncol(grads) != d) stop("Dimension mismatch between gradients and inverse Hessian.")
  
  grads_cent <- scale(grads, center = TRUE, scale = FALSE)
  grads_hat_cent <- scale(grads_hat, center = TRUE, scale = FALSE)
  
  cov_grads <- (1 / n) * (t(grads_cent) %*% grads_hat_cent + t(grads_hat_cent) %*% grads_cent)
  
  all_grads_hat <- rbind(grads_hat, grads_hat_unlabeled)
  var_grads_hat <- cov(all_grads_hat)
  
  vhat <- if (is.null(coord)) inv_hessian else inv_hessian[coord, coord, drop = FALSE]
  
  if (optim_mode == "overall") {
    num <- if (is.null(coord)) {
      sum(diag(vhat %*% cov_grads %*% vhat))
    } else {
      as.numeric(vhat %*% cov_grads %*% vhat)
    }
    denom <- if (is.null(coord)) {
      2 * (1 + n / N) * sum(diag(vhat %*% var_grads_hat %*% vhat))
    } else {
      2 * (1 + n / N) * as.numeric(vhat %*% var_grads_hat %*% vhat)
    }
    lhat <- num / denom
  } else if (optim_mode == "element") {
    num <- diag(vhat %*% cov_grads %*% vhat)
    denom <- 2 * (1 + n / N) * diag(vhat %*% var_grads_hat %*% vhat)
    lhat <- num / denom
  } else {
    stop("Invalid value for optim_mode.")
  }
  
  if (clip) {
    lhat <- pmax(pmin(lhat, 1), 0)
  }
  return(lhat)
}


# PPI mean point estimate
ppi_mean_pointestimate <- function(Y, Yhat, Yhat_unlabeled, lhat = NULL, coord = NULL, w = NULL, w_unlabeled = NULL, lambd_optim_mode = "overall") {
  Y <- reshape_to_2d(Y)
  Yhat <- reshape_to_2d(Yhat)
  Yhat_unlabeled <- reshape_to_2d(Yhat_unlabeled)
  
  n <- length(Y)
  N <- length(Yhat_unlabeled)
  d <- 1
  
  w <- construct_weight_vector(n, w, vectorized = TRUE)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled, vectorized = TRUE)
  
  if (is.null(lhat)) {
    # First-stage point estimate (lambda = 1)
    ppi_pointest <- colMeans(w_unlabeled * Yhat_unlabeled) + colMeans(w * (Y - Yhat))
    
    grads <- w * (Y - matrix(rep(ppi_pointest, each = n), nrow = n))
    grads_hat <- w * (Yhat - matrix(rep(ppi_pointest, each = n), nrow = n))
    grads_hat_unlabeled <- w_unlabeled * (Yhat_unlabeled - matrix(rep(ppi_pointest, each = N), nrow = N))
    
    inv_hessian <- diag(d)  # Identity for OLS
    lhat <- calc_lhat_glm(grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord = coord, clip = TRUE, optim_mode = lambd_optim_mode)
    
    return(ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lhat = lhat, coord = coord, w = w, w_unlabeled = w_unlabeled))
  } else {
    # Final estimator
    term1 <- colMeans(w_unlabeled * lhat * Yhat_unlabeled)
    term2 <- colMeans(w * (Y - lhat * Yhat))
    return(as.vector(term1 + term2))
  }
}

ppi_mean_ci <- function(Y,
                        Yhat,
                        Yhat_unlabeled,
                        alpha = 0.1,
                        alternative = "two-sided",
                        lhat = NULL,
                        coord = NULL,
                        w = NULL,
                        w_unlabeled = NULL,
                        lambd_optim_mode = "overall") {
  
  n <- nrow(Y)
  N <- nrow(Yhat_unlabeled)
  d <- if (is.null(dim(Y))) 1 else ncol(Y)
  
  Y <- reshape_to_2d(Y)
  Yhat <- reshape_to_2d(Yhat)
  Yhat_unlabeled <- reshape_to_2d(Yhat_unlabeled)
  
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  if (is.null(lhat)) {
    ppi_pointest <- ppi_mean_pointestimate(
      Y = Y,
      Yhat = Yhat,
      Yhat_unlabeled = Yhat_unlabeled,
      lhat = 1,
      w = w,
      w_unlabeled = w_unlabeled
    )
    
    grads <- w * (Y - ppi_pointest)
    grads_hat <- w * (Yhat - ppi_pointest)
    grads_hat_unlabeled <- w_unlabeled * (Yhat_unlabeled - ppi_pointest)
    
    inv_hessian <- diag(d)
    
    lhat <- calc_lhat_glm(
      grads = grads,
      grads_hat = grads_hat,
      grads_hat_unlabeled = grads_hat_unlabeled,
      inv_hessian = inv_hessian,
      coord = NULL,
      clip = FALSE,
      optim_mode = lambd_optim_mode
    )
    
    return(ppi_mean_ci(
      Y = Y,
      Yhat = Yhat,
      Yhat_unlabeled = Yhat_unlabeled,
      alpha = alpha,
      lhat = lhat,
      coord = coord,
      w = w,
      w_unlabeled = w_unlabeled
    ))
  }
  
  ppi_pointest <- ppi_mean_pointestimate(
    Y = Y,
    Yhat = Yhat,
    Yhat_unlabeled = Yhat_unlabeled,
    lhat = lhat,
    coord = coord,
    w = w,
    w_unlabeled = w_unlabeled
  )
  
  imputed_std <- apply(w_unlabeled * (lhat * Yhat_unlabeled), 2, sd) / sqrt(N)
  rectifier_std <- apply(w * (Y - lhat * Yhat), 2, sd) / sqrt(n)
  
  total_se <- sqrt(imputed_std^2 + rectifier_std^2)
  
  return(zconfint_generic(ppi_pointest, total_se, alpha, alternative))
}


ols_get_stats <- function(
    pointest,
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    w = NULL,
    w_unlabeled = NULL,
    use_unlabeled = TRUE
) {
  # Sizes
  n <- nrow(Y)
  N <- nrow(Yhat_unlabeled)
  d <- ncol(X)
  
  # Normalize weights
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  # Initialize matrices
  hessian <- matrix(0, nrow = d, ncol = d)
  grads <- matrix(0, nrow = n, ncol = d)
  grads_hat <- matrix(0, nrow = n, ncol = d)
  grads_hat_unlabeled <- matrix(0, nrow = N, ncol = d)
  
  if (use_unlabeled) {
    for (i in 1:N) {
      xi <- X_unlabeled[i, ]
      pred <- sum(xi * pointest)
      resid <- pred - Yhat_unlabeled[i]
      hessian <- hessian + w_unlabeled[i] / (N + n) * tcrossprod(xi)
      grads_hat_unlabeled[i, ] <- w_unlabeled[i] * xi * resid
    }
  }
  
  for (i in 1:n) {
    xi <- X[i, ]
    pred <- sum(xi * pointest)
    resid_label <- pred - Y[i]
    resid_hat <- pred - Yhat[i]
    
    if (use_unlabeled) {
      hessian <- hessian + w[i] / (N + n) * tcrossprod(xi)
    } else {
      hessian <- hessian + w[i] / n * tcrossprod(xi)
    }
    
    grads[i, ] <- w[i] * xi * resid_label
    grads_hat[i, ] <- w[i] * xi * resid_hat
  }
  
  inv_hessian <- solve(hessian)
  return(list(
    grads = grads,
    grads_hat = grads_hat,
    grads_hat_unlabeled = grads_hat_unlabeled,
    inv_hessian = inv_hessian
  ))
}


ppi_ols_pointestimate <- function(X, Y, Yhat, X_unlabeled, Yhat_unlabeled,
                                  lhat = NULL, coord = NULL, w = NULL, w_unlabeled = NULL) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  
  if (is.null(w)) {
    w <- rep(1, n)
  } else {
    w <- w / sum(w) * n
  }
  
  if (is.null(w_unlabeled)) {
    w_unlabeled <- rep(1, N)
  } else {
    w_unlabeled <- w_unlabeled / sum(w_unlabeled) * N
  }
  
  use_unlabeled <- !is.null(lhat) && lhat != 0
  
  if (is.null(lhat)) {
    Sigma <- t(X) %*% X / n
    Sigma_inv <- solve(Sigma)
    ppi_pointest <- Sigma_inv %*% (
      t(X) %*% Y / n +
        t(X_unlabeled) %*% Yhat_unlabeled / N -
        t(X) %*% Yhat / n
    )
  } else {
    Sigma <- (1 - lhat) * t(X) %*% X / n + lhat * t(X_unlabeled) %*% X_unlabeled / N
    Sigma_inv <- solve(Sigma)
    ppi_pointest <- Sigma_inv %*% (
      t(X) %*% Y / n +
        lhat * t(X_unlabeled) %*% Yhat_unlabeled / N -
        lhat * t(X) %*% Yhat / n
    )
  }
  
  if (is.null(lhat)) {
    stats <- ols_get_stats(
      pointest = ppi_pointest,
      X = X,
      Y = Y,
      Yhat = Yhat,
      X_unlabeled = X_unlabeled,
      Yhat_unlabeled = Yhat_unlabeled,
      w = w,
      w_unlabeled = w_unlabeled,
      use_unlabeled = use_unlabeled
    )
    
    lhat <- calc_lhat_glm(
      grads = stats$grads,
      grads_hat = stats$grads_hat,
      grads_hat_unlabeled = stats$grads_hat_unlabeled,
      inv_hessian = stats$inv_hessian,
      coord = coord,
      clip = TRUE
    )
    
    return(ppi_ols_pointestimate(
      X, Y, Yhat, X_unlabeled, Yhat_unlabeled,
      lhat = lhat, coord = coord, w = w, w_unlabeled = w_unlabeled
    ))
  } else {
    return(ppi_pointest)
  }
}


ppi_ols_ci <- function(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha = 0.1,
    alternative = "two-sided",
    lhat = NULL,
    coord = NULL,
    w = NULL,
    w_unlabeled = NULL
) {
  n <- nrow(Y)
  N <- nrow(Yhat_unlabeled)
  d <- ncol(X)
  
  # Normalize weights
  w <- construct_weight_vector(n, w)
  w_unlabeled <- construct_weight_vector(N, w_unlabeled)
  
  use_unlabeled <- is.null(lhat) || lhat != 0
  
  # Compute PPI point estimate (OLS)
  ppi_pointest <- ppi_ols_pointestimate(
    X = X,
    Y = Y,
    Yhat = Yhat,
    X_unlabeled = X_unlabeled,
    Yhat_unlabeled = Yhat_unlabeled,
    lhat = lhat,
    coord = coord,
    w = w,
    w_unlabeled = w_unlabeled
  )
  
  # Compute statistics
  stats <- ols_get_stats(
    pointest = ppi_pointest,
    X = X,
    Y = Y,
    Yhat = Yhat,
    X_unlabeled = X_unlabeled,
    Yhat_unlabeled = Yhat_unlabeled,
    w = w,
    w_unlabeled = w_unlabeled,
    use_unlabeled = use_unlabeled
  )
  
  grads <- stats$grads
  grads_hat <- stats$grads_hat
  grads_hat_unlabeled <- stats$grads_hat_unlabeled
  inv_hessian <- stats$inv_hessian
  
  # Estimate lhat if not provided
  if (is.null(lhat)) {
    lhat <- calc_lhat_glm(
      grads = grads,
      grads_hat = grads_hat,
      grads_hat_unlabeled = grads_hat_unlabeled,
      inv_hessian = inv_hessian,
      coord = coord,
      clip = TRUE
    )
    
    # Recurse with estimated lhat
    return(ppi_ols_ci(
      X = X,
      Y = Y,
      Yhat = Yhat,
      X_unlabeled = X_unlabeled,
      Yhat_unlabeled = Yhat_unlabeled,
      alpha = alpha,
      alternative = alternative,
      lhat = lhat,
      coord = coord,
      w = w,
      w_unlabeled = w_unlabeled
    ))
  }
  
  # Compute variance
  var_unlabeled <- cov(lhat * grads_hat_unlabeled)
  var_labeled <- cov(grads - lhat * grads_hat)
  Sigma_hat <- inv_hessian %*% ((n / N) * var_unlabeled + var_labeled) %*% inv_hessian
  
  # Compute standard error
  se <- sqrt(diag(Sigma_hat) / n)
  
  # Final confidence interval
  return(zconfint_generic(
    est = ppi_pointest,
    se = se,
    alpha = alpha,
    alternative = alternative
  ))
}


######### Start for logistic 
logistic_get_stats <- function(pointest, 
                               X, Y, Yhat, 
                               X_unlabeled, Yhat_unlabeled, 
                               w = NULL, w_unlabeled = NULL, 
                               use_unlabeled = TRUE) {
  
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(Yhat_unlabeled)
  
  # Normalize weights
  if (is.null(w)) {
    w <- rep(1, n)
  } else {
    w <- w / sum(w) * n
  }
  
  if (is.null(w_unlabeled)) {
    w_unlabeled <- rep(1, N)
  } else {
    w_unlabeled <- w_unlabeled / sum(w_unlabeled) * N
  }
  
  # Predicted probabilities
  mu <- plogis(X %*% pointest)                  # labeled
  mu_til <- plogis(X_unlabeled %*% pointest)    # unlabeled
  
  hessian <- matrix(0, nrow = d, ncol = d)
  grads_hat_unlabeled <- matrix(0, nrow = N, ncol = d)
  
  # Compute hessian and unlabeled gradients
  if (use_unlabeled) {
    for (i in 1:N) {
      xi <- X_unlabeled[i, , drop = FALSE]
      hessian <- hessian + 
        (w_unlabeled[i] / (N + n)) * mu_til[i] * (1 - mu_til[i]) * t(xi) %*% xi
      grads_hat_unlabeled[i, ] <- 
        w_unlabeled[i] * X_unlabeled[i, ] * (mu_til[i] - Yhat_unlabeled[i])
    }
  }
  
  grads <- matrix(0, nrow = n, ncol = d)
  grads_hat <- matrix(0, nrow = n, ncol = d)
  
  # Compute gradients and add to hessian
  for (i in 1:n) {
    xi <- X[i, , drop = FALSE]
    weight <- if (use_unlabeled) w[i] / (N + n) else w[i] / n
    hessian <- hessian + weight * mu[i] * (1 - mu[i]) * t(xi) %*% xi
    
    grads[i, ] <- w[i] * X[i, ] * (mu[i] - Y[i])
    grads_hat[i, ] <- w[i] * X[i, ] * (mu[i] - Yhat[i])
  }
  
  inv_hessian <- solve(hessian)
  
  return(list(
    grads = grads,
    grads_hat = grads_hat,
    grads_hat_unlabeled = grads_hat_unlabeled,
    inv_hessian = inv_hessian
  ))
}


safe_log1pexp <- function(x) {
  out <- numeric(length(x))
  idxs <- x > 10
  out[idxs] <- x[idxs]
  out[!idxs] <- log1p(exp(x[!idxs]))
  return(out)
}


ppi_logistic_pointestimate <- function(X, Y, Yhat, 
                                       X_unlabeled, Yhat_unlabeled,
                                       lhat = NULL, coord = NULL,
                                       optimizer_options = list(), 
                                       w = NULL, w_unlabeled = NULL) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  
  # Normalize weights
  if (is.null(w)) {
    w <- rep(1, n)
  } else {
    w <- w / sum(w) * n
  }
  if (is.null(w_unlabeled)) {
    w_unlabeled <- rep(1, N)
  } else {
    w_unlabeled <- w_unlabeled / sum(w_unlabeled) * N
  }
  
  if (is.null(optimizer_options$ftol)) {
    optimizer_options$ftol <- 1e-15
  }
  
  # Initial theta: classical logistic regression
  df <- data.frame(Y = as.factor(Y), X)
  model <- glm(Y ~ . - 1, data = df, family = binomial())
  theta <- coef(model)
  if (is.null(dim(theta))) {
    theta <- as.numeric(theta)
  }
  
  lhat_curr <- if (is.null(lhat)) 1 else lhat
  
  # Loss function
  rectified_logistic_loss <- function(theta_vec) {
    theta_vec <- as.numeric(theta_vec)
    term1 <- lhat_curr / N * sum(
      w_unlabeled * (-Yhat_unlabeled * (X_unlabeled %*% theta_vec) + 
                       safe_log1pexp(X_unlabeled %*% theta_vec))
    )
    term2 <- -lhat_curr / n * sum(
      w * (-Yhat * (X %*% theta_vec) + 
             safe_log1pexp(X %*% theta_vec))
    )
    term3 <- 1 / n * sum(
      w * (-Y * (X %*% theta_vec) + 
             safe_log1pexp(X %*% theta_vec))
    )
    return(term1 + term2 + term3)
  }
  
  # Gradient function
  rectified_logistic_grad <- function(theta_vec) {
    theta_vec <- as.numeric(theta_vec)
    grad1 <- (lhat_curr / N) * t(X_unlabeled) %*% 
      (w_unlabeled * (plogis(X_unlabeled %*% theta_vec) - Yhat_unlabeled))
    
    grad2 <- (-lhat_curr / n) * t(X) %*% 
      (w * (plogis(X %*% theta_vec) - Yhat))
    
    grad3 <- (1 / n) * t(X) %*% 
      (w * (plogis(X %*% theta_vec) - Y))
    
    return(as.numeric(grad1 + grad2 + grad3))
  }
  
  # Optimization
  optim_result <- optim(par = theta,
                        fn = rectified_logistic_loss,
                        gr = rectified_logistic_grad,
                        method = "L-BFGS-B",
                        control = list(fnscale = 1, factr = optimizer_options$ftol))
  
  ppi_pointest <- optim_result$par
  
  # Re-estimate with optimal lhat if needed
  if (is.null(lhat)) {
    stats <- logistic_get_stats(ppi_pointest, X, Y, Yhat,
                                X_unlabeled, Yhat_unlabeled,
                                w = w, w_unlabeled = w_unlabeled)
    
    lhat <- calc_lhat_glm(stats$grads,
                          stats$grads_hat,
                          stats$grads_hat_unlabeled,
                          stats$inv_hessian,
                          coord = coord,
                          clip = TRUE)
    
    return(ppi_logistic_pointestimate(
      X, Y, Yhat, X_unlabeled, Yhat_unlabeled,
      lhat = lhat, coord = coord,
      optimizer_options = optimizer_options,
      w = w, w_unlabeled = w_unlabeled
    ))
  } else {
    return(ppi_pointest)
  }
}


ppi_logistic_ci <- function(X, Y, Yhat,
                            X_unlabeled, Yhat_unlabeled,
                            alpha = 0.1,
                            alternative = "two-sided",
                            lhat = NULL,
                            coord = NULL,
                            optimizer_options = list(),
                            w = NULL,
                            w_unlabeled = NULL) {
  n <- nrow(X)
  d <- ncol(X)
  N <- nrow(X_unlabeled)
  
  if (is.null(w)) {
    w <- rep(1, n)
  } else {
    w <- w / sum(w) * n
  }
  if (is.null(w_unlabeled)) {
    w_unlabeled <- rep(1, N)
  } else {
    w_unlabeled <- w_unlabeled / sum(w_unlabeled) * N
  }
  
  use_unlabeled <- !identical(lhat, 0)
  
  # Compute point estimate
  ppi_pointest <- ppi_logistic_pointestimate(
    X, Y, Yhat,
    X_unlabeled, Yhat_unlabeled,
    optimizer_options = optimizer_options,
    lhat = lhat,
    coord = coord,
    w = w,
    w_unlabeled = w_unlabeled
  )
  
  # Get gradients and Hessian
  stats <- logistic_get_stats(
    ppi_pointest,
    X, Y, Yhat,
    X_unlabeled, Yhat_unlabeled,
    w = w,
    w_unlabeled = w_unlabeled,
    use_unlabeled = use_unlabeled
  )
  
  grads <- stats$grads
  grads_hat <- stats$grads_hat
  grads_hat_unlabeled <- stats$grads_hat_unlabeled
  inv_hessian <- stats$inv_hessian
  
  # Estimate lhat if needed
  if (is.null(lhat)) {
    lhat <- calc_lhat_glm(
      grads,
      grads_hat,
      grads_hat_unlabeled,
      inv_hessian,
      coord = coord,
      clip = TRUE
    )
    
    return(ppi_logistic_ci(
      X, Y, Yhat,
      X_unlabeled, Yhat_unlabeled,
      alpha = alpha,
      alternative = alternative,
      lhat = lhat,
      coord = coord,
      optimizer_options = optimizer_options,
      w = w,
      w_unlabeled = w_unlabeled
    ))
  }
  
  # Variance computation
  var_unlabeled <- cov(lhat * grads_hat_unlabeled)
  var <- cov(grads - lhat * grads_hat)
  
  Sigma_hat <- inv_hessian %*% ((n / N) * var_unlabeled + var) %*% inv_hessian
  se <- sqrt(diag(Sigma_hat) / n)
  
  # Confidence intervals
  return(zconfint_generic(ppi_pointest, se, alpha = alpha, alternative = alternative))
}