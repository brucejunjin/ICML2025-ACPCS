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

