SL.ranger.tune <- function(Y, X, newX, family, obsWeights, id,
                           # Tuning grid. Adjust as needed.
                           tune_params = list(
                             mtry            = c(1, 2, 3), 
                             min.node.size   = c(1, 5, 10),
                             sample.fraction = c(0.7, 1.0)
                           ),
                           cv_folds = 5,
                           # Additional args for ranger
                           num.trees = 500,
                           seed = 1,
                           ...) {
  # For binomial family, convert Y to factor with levels 0/1
  # This ensures ranger can output class probabilities for class '1'.
  if (family$family == "binomial") {
    Y_factor <- factor(Y, levels = c(0, 1))
  } else {
    stop("SL.ranger.tune currently only implemented for binomial family.")
  }
  
  # Generate CV folds
  set.seed(seed)
  folds <- sample(rep(seq_len(cv_folds), length.out = length(Y_factor)))
  
  # Function to compute AUC quickly
  # (You could use other metrics, like accuracy, if you prefer.)
  auc_fun <- function(labels, probs) {
    # Simple AUC function
    # Requires 'pROC' or a custom AUC implementation
    # We'll do a quick manual calculation here
    # or you can do: pROC::auc(labels, probs, quiet = TRUE)
    # For a simple demonstration, let's do a rank-based approach:
    if (length(unique(labels)) < 2) {
      return(NA_real_)
    }
    ord <- order(probs, decreasing = TRUE)
    labels_sorted <- labels[ord]
    cum_pos_found <- cumsum(labels_sorted == 1)
    total_pos <- sum(labels_sorted == 1)
    total_neg <- sum(labels_sorted == 0)
    if (total_pos == 0 || total_neg == 0) {
      return(NA_real_)
    }
    # Mann–Whitney U statistic method
    auc_est <- sum(cum_pos_found[labels_sorted == 0]) / (total_pos * total_neg)
    return(1 - auc_est)
  }
  
  best_auc <- -Inf
  best_params <- list()
  
  # Grid Search
  for (mtry_val in tune_params$mtry) {
    for (min_node_val in tune_params$min.node.size) {
      for (sf_val in tune_params$sample.fraction) {
        # Perform simple CV
        auc_cv <- numeric(cv_folds)
        
        for (fold_idx in seq_len(cv_folds)) {
          train_idx <- which(folds != fold_idx)
          test_idx  <- which(folds == fold_idx)
          
          # Fit on training fold
          rf_cv <- ranger(
            x               = X[train_idx, , drop = FALSE],
            y               = Y_factor[train_idx],
            mtry            = mtry_val,
            min.node.size   = min_node_val,
            sample.fraction = sf_val,
            num.trees       = num.trees,
            probability     = TRUE,
            classification  = TRUE,
            seed            = seed
          )
          
          # Predict on test fold => Probability for class '1'
          preds_prob <- predict(rf_cv, data = X[test_idx, , drop = FALSE],
                                type = "response")$predictions[, "1"]
          
          # Compute AUC
          auc_cv[fold_idx] <- auc_fun(Y[test_idx], preds_prob)
        }
        
        mean_auc <- mean(auc_cv, na.rm = TRUE)
        if (mean_auc > best_auc) {
          best_auc <- mean_auc
          best_params <- list(mtry            = mtry_val,
                              min.node.size   = min_node_val,
                              sample.fraction = sf_val)
        }
      }
    }
  }
  
  # Fit final model on entire dataset using best hyperparameters
  final_rf <- ranger(
    x               = X,
    y               = Y_factor,
    mtry            = best_params$mtry,
    min.node.size   = best_params$min.node.size,
    sample.fraction = best_params$sample.fraction,
    num.trees       = num.trees,
    probability     = TRUE,
    classification  = TRUE,
    seed            = seed
  )
  
  # Generate predictions on newX
  pred <- predict(final_rf, data = newX, type = "response")$predictions[, "1"]
  
  # Return object for later predictions + the predictions
  fit <- list(object = final_rf, best_params = best_params)
  class(fit) <- "SL.ranger.tune"
  
  return(list(pred = pred, fit = fit))
}


predict.SL.ranger.tune <- function(object, newdata, family, X = NULL, Y = NULL, ...) {
  preds <- predict(object$object, data = newdata, type = "response")$predictions[, "1"]
  return(preds)
}

SL.ksvm.tune <- function(Y, X, newX, family, obsWeights, id,
                         kernel = "rbfdot",
                         # You can expand or refine these grids as you like
                         ranges = list(sigma = 2^(-5:0),  # e.g. 2^-5, 2^-4, ..., 2^0
                                       C     = 2^(-2:2)), 
                         cv_folds = 5,
                         ...) {
  # Determine type for ksvm
  # For binary outcomes with family=binomial, use "C-svc" + probability model
  type <- if (family$family == "binomial") "C-svc" else "eps-svr"
  
  best_acc <- -Inf
  best_params <- list(sigma = NULL, C = NULL)
  
  # A simple manual grid search + CV:
  folds <- sample(rep(seq_len(cv_folds), length.out = length(Y)))
  
  for (sig in ranges$sigma) {
    for (cc in ranges$C) {
      cv_acc <- numeric(cv_folds)
      
      for (fold_idx in seq_len(cv_folds)) {
        train_idx <- which(folds != fold_idx)
        test_idx  <- which(folds == fold_idx)
        
        # Fit on training fold
        mod_cv <- ksvm(
          x          = as.matrix(X[train_idx, ]),
          y          = Y[train_idx],
          type       = type,
          kernel     = kernel,
          kpar       = list(sigma = sig),
          C          = cc,
          prob.model = TRUE
        )
        
        # Predict on test fold
        preds_cv <- predict(mod_cv, newdata = as.matrix(X[test_idx, ]), type = "response")
        cv_acc[fold_idx] <- mean(preds_cv == Y[test_idx])
      }
      
      mean_acc <- mean(cv_acc)
      if (mean_acc > best_acc) {
        best_acc <- mean_acc
        best_params$sigma <- sig
        best_params$C     <- cc
      }
    }
  }
  
  # Fit final model on entire dataset using best hyperparameters
  final_model <- ksvm(
    x          = as.matrix(X),
    y          = Y,
    type       = type,
    kernel     = kernel,
    kpar       = list(sigma = best_params$sigma),
    C          = best_params$C,
    prob.model = TRUE
  )
  
  # Generate predictions on newX (the SuperLearner "test" set)
  # For binary classification, predict probabilities in column 2
  pred <- predict(final_model, newdata = as.matrix(newX), type = "probabilities")[, 2]
  
  # Return object for later predictions + the predictions
  fit <- list(object = final_model, best_params = best_params)
  class(fit) <- "SL.ksvm.tune"
  
  return(list(pred = pred, fit = fit))
}

predict.SL.ksvm.tune <- function(object, newdata, family, X = NULL, Y = NULL, ...) {
  preds <- predict(object$object, newdata = as.matrix(newdata), type = "probabilities")[, 2]
  return(preds)
}

SL.xgboost.tune <- function(Y, X, newX, family, obsWeights, id,
                            # Define the hyperparameter grid (customize as desired)
                            tune_params = expand.grid(
                              eta        = c(0.01, 0.1),
                              max_depth  = c(2, 4, 6),
                              gamma      = c(0, 1)
                            ),
                            cv_folds = 5,      # number of CV folds in manual tuning
                            nrounds   = 100,   # total number of boosting rounds
                            early_stopping_rounds = 10,
                            verbose = 0,
                            ...) {
  # Check family
  if (family$family != "binomial") {
    stop("SL.xgboost.tune only implemented for binomial family.")
  }
  
  # For xgboost, we need the label as numeric 0/1
  Y_vec <- as.numeric(Y)
  X_mat <- as.matrix(X)
  
  # Create folds for manual CV
  set.seed(1)
  folds <- sample(rep(seq_len(cv_folds), length.out = length(Y_vec)))
  
  # We'll use AUC for hyperparameter selection 
  # (You can use accuracy or logloss or any metric you prefer.)
  # For AUC calculation, we’ll rely on either a custom function or pROC, etc.
  # Here, let's define a simple custom AUC function for illustration:
  get_auc <- function(labels, preds) {
    # If all labels are the same, return NA
    if (length(unique(labels)) < 2) return(NA_real_)
    # This is a quick rank-based approximation of AUC
    ord <- order(preds, decreasing = TRUE)
    labels_ord <- labels[ord]
    cum_pos <- cumsum(labels_ord == 1)
    total_pos <- sum(labels_ord == 1)
    total_neg <- sum(labels_ord == 0)
    if (total_pos == 0 | total_neg == 0) return(NA_real_)
    auc_val <- sum(cum_pos[labels_ord == 0]) / (total_pos * total_neg)
    return(1 - auc_val)
  }
  
  best_auc <- -Inf
  best_params <- NULL
  best_nrounds <- nrounds  # We'll store the best iteration if we want to do early stopping
  
  #---------------------------------------------------------------------
  # Grid Search
  #---------------------------------------------------------------------
  for (i in seq_len(nrow(tune_params))) {
    # Extract hyperparameters for this iteration
    param_grid <- tune_params[i, ]
    
    # We'll do a simple K-fold CV
    auc_cv <- numeric(cv_folds)
    for (k in seq_len(cv_folds)) {
      # train / test indices
      train_idx <- which(folds != k)
      test_idx  <- which(folds == k)
      
      dtrain <- xgb.DMatrix(data = X_mat[train_idx, ], label = Y_vec[train_idx])
      dtest  <- xgb.DMatrix(data = X_mat[test_idx, ],  label = Y_vec[test_idx])
      
      # xgboost params
      params_xgb <- list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta        = param_grid$eta,
        max_depth  = param_grid$max_depth,
        gamma      = param_grid$gamma
      )
      
      # train xgboost
      xgb_model_cv <- xgb.train(
        params                    = params_xgb,
        data                      = dtrain,
        nrounds                   = nrounds,
        watchlist                 = list(val = dtest),  # for early stopping
        early_stopping_rounds     = early_stopping_rounds,
        verbose                   = verbose
      )
      
      # best iteration from CV
      best_iter <- xgb_model_cv$best_iteration
      
      # predict on test fold
      preds_cv <- predict(xgb_model_cv, X_mat[test_idx, ], ntreelimit = best_iter)
      # compute AUC
      auc_cv[k] <- get_auc(Y_vec[test_idx], preds_cv)
    }
    mean_auc <- mean(auc_cv, na.rm = TRUE)
    
    # If we got a better AUC, store results
    if (!is.na(mean_auc) && mean_auc > best_auc) {
      best_auc <- mean_auc
      best_params <- param_grid
      # optional: store best_nrounds => might want to do a second pass,
      # but for simplicity, let's leave it as nrounds or store a typical best_iter
    }
  }
  
  #---------------------------------------------------------------------
  # Fit final model on entire dataset using best hyperparameters
  #---------------------------------------------------------------------
  if (is.null(best_params)) {
    # fallback if everything is NA for some reason
    best_params <- tune_params[1, ]
  }
  
  # We can do a final pass with early stopping on the entire data 
  # but let's just train with all data. 
  dtrain_full <- xgb.DMatrix(data = X_mat, label = Y_vec)
  
  final_params <- list(
    objective  = "binary:logistic",
    eval_metric= "auc",
    eta        = best_params$eta,
    max_depth  = best_params$max_depth,
    gamma      = best_params$gamma
  )
  
  final_model <- xgb.train(
    params    = final_params,
    data      = dtrain_full,
    nrounds   = nrounds,
    verbose   = verbose
  )
  
  # Generate predictions on newX
  newX_mat <- as.matrix(newX)
  pred <- predict(final_model, newX_mat)
  
  # Return object for later predictions + the predictions
  fit <- list(object = final_model, best_params = best_params)
  class(fit) <- "SL.xgboost.tune"
  
  return(list(pred = pred, fit = fit))
}

predict.SL.xgboost.tune <- function(object, newdata, family, X = NULL, Y = NULL, ...) {
  newX_mat <- as.matrix(newdata)
  preds <- predict(object$object, newX_mat)
  return(preds)
}