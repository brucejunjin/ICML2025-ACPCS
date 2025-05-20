predict_probability <- function(model, new_data_matrix, col_names) {
  colnames(new_data_matrix) <- col_names
  new_data_df <- as.data.frame(new_data_matrix)
  predicted_probs <- predict(model, newdata = new_data_df, type = "response")
  return(predicted_probs)
}

logit <- function(betavec, x){
  # x is n * p
  return(exp(x %*% betavec) / (1 + exp(x %*% betavec)))
}

unpenalized_loss <- function(avec, betavec, xtg, xsc, yhattg, pihat) {
  Xtg_full <- cbind(rep(1, nrow(xtg)), xtg)
  diff_tg <- (yhattg - logit(betavec, Xtg_full)) / (1 - pihat)
  part <- (avec - as.vector(diff_tg) %*% Xtg_full) / (nrow(xtg) + nrow(xsc))
  sum(part^2)
}

unpenalized_loss_boot <- function(avec, betavec, xtg, xsc, yhattg, pihat, ksitg) {
  Xtg_full <- cbind(rep(1, nrow(xtg)), xtg)
  diff_tg <- (yhattg - logit(betavec, Xtg_full)) * ksitg / (1 - pihat)
  part <- (avec - as.vector(diff_tg) %*% Xtg_full) / (nrow(xtg) + nrow(xsc))
  sum(part^2)
}

colSd <- function (x, na.rm=FALSE) apply(X=x, MARGIN=2, FUN=sd, na.rm=na.rm)

PPIwcf <- function(source, target, sourcebb = NULL, targetbb = NULL, family = 'gaussian',
                SL.library = c("SL.glm", "SL.gam", "SL.glmnet", "SL.ranger", "SL.ksvm", 
                               "SL.mean", "SL.randomForest", "SL.xgboost"),
                bootstrap = FALSE){
  set.seed(2025)
  xtg <- target$x
  xsc <- source$x
  ysc <- source$y
  ## estimations
  ## estimate source
  pihat <- nrow(xtg)/(nrow(xtg) + nrow(xsc))
  xcomb <- as.data.frame(rbind(xtg, xsc))
  rind <- c(rep(0, nrow(xtg)), rep(1, nrow(xsc)))
  ## estimate pi(x) with super learner
  if ((family == 'gaussian')|!(FALSE %in% (SL.library == c('SL.glm')))){
    data <- data.frame(y = rind, xcomb)
    gmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
    col_names <- names(gmodel$coefficients)[-1]
    phat <- predict_probability(gmodel, xsc, col_names)
    phatall <- predict_probability(gmodel, rbind(xtg, xsc), col_names)
  } else {
    sl_stacked <- SuperLearner(Y = rind, X = xcomb, family = binomial(),
                               SL.library = SL.library, verbose = F, method = "method.NNloglik")
    phat <- sl_stacked$SL.predict[(nrow(xtg)+1):(nrow(xtg) + nrow(xsc))]
    phatall <- sl_stacked$SL.predict
  }
  what <- pihat / (1 - pihat) * (1 / phat - 1)
  if (is.null(sourcebb) & is.null(targetbb)){
    ## estimate E(Y|X)
    if (family == 'gaussian'){
      data <- data.frame(y = ysc, xsc)
      lmodel <- lm(y ~ ., data = data)
      col_names <- names(lmodel$coefficients)[-1]
      yhat <- predict_probability(lmodel, xcomb, col_names)
      yhattg <- yhat[1:nrow(xtg)]
      yhatsc <- yhat[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))] 
    } else if (family == 'binomial'){
      if (!(FALSE %in% (SL.library == c('SL.glm')))){
        data <- data.frame(y = ysc, xsc)
        glmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
        col_names <- names(glmodel$coefficients)[-1]
        yhat <- predict_probability(glmodel, xcomb, col_names)
      } else {
        sl_stacked <- SuperLearner(Y = ysc, X = as.data.frame(xsc), family = binomial(),
                                   SL.library = SL.library, verbose = F, method = "method.NNloglik")
        yhat <- predict(sl_stacked, newdata = as.data.frame(rbind(xtg, xsc)))$pred
      }
      yhattg <- yhat[1:nrow(xtg)]
      yhatsc <- yhat[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))] 
    } else {
      stop('Please provide a correct family!')
    }
    if (bootstrap == FALSE){
      ## solve mean equation
      scpart <- sum(what * (ysc - yhatsc) / pihat)
      tgpart <- sum(yhattg / (1 - pihat))
      meanhat <- 1 / (nrow(xtg) / (1 - pihat)) * (scpart + tgpart)
      ## solve param equation
      if (family == 'gaussian'){
        scpart <- t(what * (ysc - yhatsc) / pihat)  %*% cbind(rep(1, nrow(xsc)), xsc) 
        tgpart <- t(yhattg / (1 - pihat)) %*% cbind(rep(1, nrow(xtg)), xtg)
        leftpart <- t(cbind(rep(1, nrow(xtg)), xtg)) %*% cbind(rep(1, nrow(xtg)), xtg) / (1 - pihat)
        betahat <- as.vector(solve(leftpart, diag(nrow(leftpart))) %*% t(scpart + tgpart))
        success <- 1
      } else if (family == 'binomial') {
        avec <- t(what * (ysc - yhatsc) / pihat)  %*% cbind(rep(1, nrow(xsc)), xsc) 
        penalized_objective <- function(betavec, lambda) {
          unpenalized_loss(avec, betavec, xtg, xsc, yhattg, pihat) + lambda * sum(betavec^2)
        }
        fit_for_lambda <- function(lambda, start_par = NULL) {
          if (is.null(start_par)) {
            start_par <- rep(0, ncol(xtg) + 1)
          }
          solution <- tryCatch(
            expr = {
              optim(
                par    = start_par,
                fn     = function(b) penalized_objective(b, lambda),
                method = "BFGS"
              )
            },
            warning = function(w) {
              message("Warning in optim: ", w$message)
              list('par' = start_par, 'convergence' = 1)
            },
            error = function(e) {
              message("Error in optim: ", e$message)
              list('par' = start_par, 'convergence' = 1)
            }
          )
          return(solution)
        }
        lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
        results_list <- vector("list", length(lambda_grid))
        names(results_list) <- paste0("lambda=", lambda_grid)
        for (i in seq_along(lambda_grid)) {
          this_lambda <- lambda_grid[i]
          fit <- fit_for_lambda(this_lambda)
          val_unpenalized <- unpenalized_loss(avec, fit$par, xtg, xsc, yhattg, pihat)
          results_list[[i]] <- list(
            lambda           = this_lambda,
            betavec_solution = fit$par,
            penalized_value  = fit$value,  
            unpenalized_loss = val_unpenalized,
            convergence      = fit$convergence
          )
        }
        converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
        if (length(converged_indices) == 0) {
          message("No model converged. Handle this case as you see fit...")
          best_lambda <- NA
          betahat     <- NA
          success     <- 0
        } else {
          converged_models <- results_list[converged_indices]
          lambdas_converged <- sapply(converged_models, function(x) x$lambda)
          best_idx_local    <- which.min(lambdas_converged)
          best_model  <- converged_models[[best_idx_local]]
          best_lambda <- best_model$lambda
          betahat     <- as.vector(best_model$betavec_solution)
          success     <- 1
        }
      }
    } else {
      meanhat <- c()
      betahat <- matrix(NA, nrow = 500, ncol = ncol(xtg) + 1)
      for (bt in 1:500){
        # generate ksi
        ksivec <- rexp(n = nrow(xtg) + nrow(xsc), 1)
        ksitg <- ksivec[1:nrow(xtg)]
        ksisc <- ksivec[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
        ## solve mean equation
        scpart <- sum(what * (ysc - yhatsc) / pihat * ksisc)
        tgpart <- sum(yhattg / (1 - pihat) * ksitg)
        meanhat[bt] <- 1 / (sum(ksitg)/ (1 - pihat)) * (scpart + tgpart)
        ## solve param equation
        if (family == 'gaussian'){
          scpart <- t(what * (ysc - yhatsc) / pihat * ksisc)  %*% cbind(rep(1, nrow(xsc)), xsc) 
          tgpart <- t(yhattg / (1 - pihat) * ksitg) %*% cbind(rep(1, nrow(xtg)), xtg)
          leftpart <- t(cbind(rep(1, nrow(xtg)), xtg)) %*% (cbind(rep(1, nrow(xtg)), xtg) * ksitg) / (1 - pihat)
          betahat[bt,] <- as.vector(solve(leftpart, diag(nrow(leftpart))) %*% t(scpart + tgpart))
        } else if (family == 'binomial') {
          avec <- t(what * (ysc - yhatsc) / pihat * ksisc)  %*% cbind(rep(1, nrow(xsc)), xsc) 
          penalized_objective <- function(betavec, lambda) {
            unpenalized_loss_boot(avec, betavec, xtg, xsc, yhattg, pihat, ksitg) + lambda * sum(betavec^2)
          }
          fit_for_lambda <- function(lambda, start_par = NULL) {
            if (is.null(start_par)) {
              start_par <- rep(0, ncol(xtg) + 1)
            }
            solution <- tryCatch(
              expr = {
                optim(
                  par    = start_par,
                  fn     = function(b) penalized_objective(b, lambda),
                  method = "BFGS"
                )
              },
              warning = function(w) {
                message("Warning in optim: ", w$message)
                list('par' = start_par, 'convergence' = 1)
              },
              error = function(e) {
                message("Error in optim: ", e$message)
                list('par' = start_par, 'convergence' = 1)
              }
            )
            return(solution)
          }
          lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
          results_list <- vector("list", length(lambda_grid))
          names(results_list) <- paste0("lambda=", lambda_grid)
          for (i in seq_along(lambda_grid)) {
            this_lambda <- lambda_grid[i]
            fit <- fit_for_lambda(this_lambda)
            val_unpenalized <- unpenalized_loss_boot(avec, fit$par, xtg, xsc, yhattg, pihat, ksitg)
            results_list[[i]] <- list(
              lambda           = this_lambda,
              betavec_solution = fit$par,
              penalized_value  = fit$value,  
              unpenalized_loss = val_unpenalized,
              convergence      = fit$convergence
            )
          }
          converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
          if (length(converged_indices) == 0) {
            message("No model converged. Handle this case as you see fit...")
            best_lambda <- NA
            betaest     <- NA
            success     <- 0
          } else {
            converged_models <- results_list[converged_indices]
            lambdas_converged <- sapply(converged_models, function(x) x$lambda)
            best_idx_local    <- which.min(lambdas_converged)
            best_model  <- converged_models[[best_idx_local]]
            best_lambda <- best_model$lambda
            betahat[bt,]     <- as.vector(best_model$betavec_solution)
          }
        }
      }
      if (sum(is.na(betahat)) == 0){
        success <- 1
      } else {
        success <- 0
      }
    }
  } else {
    if (is.null(sourcebb) | is.null(targetbb)){
      stop('The black box model for source and target should be simotaneously zero or nonzero!')
    } else{
      if (family == 'gaussian'){
        ## estimate E(Y|X,Yhat)
        xscb <- cbind(xsc, sourcebb)
        xtgb <- cbind(xtg, targetbb)
        xcombb <- rbind(xtgb, xscb)
        data <- data.frame(y = ysc, xscb)
        l1model <- lm(y ~ ., data = data)
        col_names <- names(l1model$coefficients)[-1]
        yhatb <- predict_probability(l1model, xcombb, col_names)
        yhatbtg <- yhatb[1:nrow(xtg)]
        yhatbsc <- yhatb[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
        ## estimate E(Y|X)
        data <- data.frame(y = yhatbsc, xsc)
        l2model <- lm(y ~ ., data = data)
        col_names <- names(l2model$coefficients)[-1]
        yhat <- predict_probability(l2model, xcomb, col_names)
        yhattg <- yhat[1:nrow(xtg)]
        yhatsc <- yhat[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
      } else if (family == 'binomial'){
        ## estimate E(Y|X,Yhat)
        xscb <- cbind(xsc, sourcebb)
        xtgb <- cbind(xtg, targetbb)
        xcombb <- rbind(xtgb, xscb)
        if (!(FALSE %in% (SL.library == c('SL.glm')))){
          data <- data.frame(y = ysc, xscb)
          gmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
          col_names <- names(gmodel$coefficients)[-1]
          yhatb <- predict_probability(gmodel, xcombb, col_names)
        } else {
          sl_stacked <- SuperLearner(Y = ysc, X = as.data.frame(xscb), family = binomial(),
                                     SL.library = SL.library, verbose = F, method = "method.NNloglik")
          newx <- as.data.frame(xcombb)
          names(newx) <- names(as.data.frame(xscb))
          yhatb <- predict(sl_stacked, newdata = newx)$pred
        }
        yhatbtg <- yhatb[1:nrow(xtg)]
        yhatbsc <- yhatb[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
        ## estimate E(Y|X)
        if (!(FALSE %in% (SL.library == c('SL.glm')))){
          data <- data.frame(y = ysc, xsc)
          gmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
          col_names <- names(gmodel$coefficients)[-1]
          yhat <- predict_probability(gmodel, xcomb, col_names)
        } else{
          sl_stacked <- SuperLearner(Y = ysc, X = as.data.frame(xsc), family = binomial(),
                                     SL.library = SL.library, verbose = F, method = "method.NNloglik")
          yhat <- predict(sl_stacked, newdata = as.data.frame(xcomb))$pred
        }
        yhattg <- yhat[1:nrow(xtg)]
        yhatsc <- yhat[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
      } else {
        stop('Please provide a correct family!')
      }
      if (bootstrap == FALSE) {
        ## solve mean equation
        scpart <- sum(what * (ysc - yhatbsc) / pihat)
        mutpart <- sum((1 - phatall) * (yhatb - yhat) / (1 - pihat))
        tgpart <- sum(yhattg / (1 - pihat))
        meanhat <- 1 / (nrow(xtg) / (1 - pihat)) * (scpart + mutpart + tgpart) 
        ## solve equation
        if (family == 'gaussian'){
          scpart <- t(what * (ysc - yhatbsc) / pihat) %*% cbind(rep(1, nrow(xsc)), xsc) 
          mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat)) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb))
          tgpart <- t(yhattg / (1 - pihat)) %*% cbind(rep(1, nrow(xtg)), xtg)
          leftpart <- t(cbind(rep(1, nrow(xtg)), xtg)) %*% cbind(rep(1, nrow(xtg)), xtg) / (1 - pihat)
          betahat <- as.vector(solve(leftpart, diag(nrow(leftpart))) %*% t(scpart + mutpart + tgpart))
          success <- 1
        } else if (family == 'binomial'){
          scpart <- t(what * (ysc - yhatbsc) / pihat) %*% cbind(rep(1, nrow(xsc)), xsc) 
          mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat)) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb)) 
          avec <- scpart + mutpart
          penalized_objective <- function(betavec, lambda) {
            unpenalized_loss(avec, betavec, xtg, xsc, yhattg, pihat) + lambda * sum(betavec^2)
          }
          fit_for_lambda <- function(lambda, start_par = NULL) {
            if (is.null(start_par)) {
              start_par <- rep(0, ncol(xtg) + 1)
            }
            solution <- tryCatch(
              expr = {
                optim(
                  par    = start_par,
                  fn     = function(b) penalized_objective(b, lambda),
                  method = "BFGS"
                )
              },
              warning = function(w) {
                message("Warning in optim: ", w$message)
                list('par' = start_par, 'convergence' = 1)
              },
              error = function(e) {
                message("Error in optim: ", e$message)
                list('par' = start_par, 'convergence' = 1)
              }
            )
            return(solution)
          }
          lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
          results_list <- vector("list", length(lambda_grid))
          names(results_list) <- paste0("lambda=", lambda_grid)
          for (i in seq_along(lambda_grid)) {
            this_lambda <- lambda_grid[i]
            fit <- fit_for_lambda(this_lambda)
            val_unpenalized <- unpenalized_loss(avec, fit$par, xtg, xsc, yhattg, pihat)
            results_list[[i]] <- list(
              lambda           = this_lambda,
              betavec_solution = fit$par,
              penalized_value  = fit$value,  
              unpenalized_loss = val_unpenalized,
              convergence      = fit$convergence
            )
          }
          converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
          if (length(converged_indices) == 0) {
            message("No model converged. Handle this case as you see fit...")
            best_lambda <- NA
            betahat     <- NA
            success     <- 0
          } else {
            converged_models <- results_list[converged_indices]
            lambdas_converged <- sapply(converged_models, function(x) x$lambda)
            best_idx_local    <- which.min(lambdas_converged)
            best_model  <- converged_models[[best_idx_local]]
            best_lambda <- best_model$lambda
            betahat     <- as.vector(best_model$betavec_solution)
            success     <- 1
          }
        } else {
          stop('Please provide a correct family!')
        }
      } else {
        meanhat <- c()
        betahat <- matrix(NA, nrow = 500, ncol = ncol(xtg) + 1)
        for (bt in 1:500){
          # generate ksi
          ksivec <- rexp(n = nrow(xtg) + nrow(xsc), 1)
          ksitg <- ksivec[1:nrow(xtg)]
          ksisc <- ksivec[(nrow(xtg) + 1): (nrow(xtg) + nrow(xsc))]
          ## solve mean equation
          scpart <- sum(what * (ysc - yhatbsc) / pihat * ksisc)
          mutpart <- sum((1 - phatall) * (yhatb - yhat) / (1 - pihat) * ksivec)
          tgpart <- sum(yhattg / (1 - pihat) * ksitg)
          meanhat[bt] <- 1 / ((sum(ksitg))/ (1 - pihat)) * (scpart + mutpart + tgpart) 
          ## solve equation
          if (family == 'gaussian'){
            scpart <- t(what * (ysc - yhatbsc) / pihat * ksisc) %*% cbind(rep(1, nrow(xsc)), xsc) 
            mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat) * ksivec) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb))
            tgpart <- t(yhattg / (1 - pihat) * ksitg) %*% cbind(rep(1, nrow(xtg)), xtg)
            leftpart <- t(cbind(rep(1, nrow(xtg)), xtg)) %*% (cbind(rep(1, nrow(xtg)), xtg) * ksitg) / (1 - pihat)
            betahat[bt,] <- as.vector(solve(leftpart, diag(nrow(leftpart))) %*% t(scpart + mutpart + tgpart))
          } else if (family == 'binomial'){
            scpart <- t(what * (ysc - yhatbsc) / pihat * ksisc) %*% cbind(rep(1, nrow(xsc)), xsc) 
            mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat) * ksivec) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb)) 
            avec <- scpart + mutpart
            penalized_objective <- function(betavec, lambda) {
              unpenalized_loss_boot(avec, betavec, xtg, xsc, yhattg, pihat, ksitg) + lambda * sum(betavec^2)
            }
            fit_for_lambda <- function(lambda, start_par = NULL) {
              if (is.null(start_par)) {
                start_par <- rep(0, ncol(xtg) + 1)
              }
              solution <- tryCatch(
                expr = {
                  optim(
                    par    = start_par,
                    fn     = function(b) penalized_objective(b, lambda),
                    method = "BFGS"
                  )
                },
                warning = function(w) {
                  message("Warning in optim: ", w$message)
                  list('par' = start_par, 'convergence' = 1)
                },
                error = function(e) {
                  message("Error in optim: ", e$message)
                  list('par' = start_par, 'convergence' = 1)
                }
              )
              return(solution)
            }
            lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
            results_list <- vector("list", length(lambda_grid))
            names(results_list) <- paste0("lambda=", lambda_grid)
            for (i in seq_along(lambda_grid)) {
              this_lambda <- lambda_grid[i]
              fit <- fit_for_lambda(this_lambda)
              val_unpenalized <- unpenalized_loss_boot(avec, fit$par, xtg, xsc, yhattg, pihat, ksitg)
              results_list[[i]] <- list(
                lambda           = this_lambda,
                betavec_solution = fit$par,
                penalized_value  = fit$value,  
                unpenalized_loss = val_unpenalized,
                convergence      = fit$convergence
              )
            }
            converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
            if (length(converged_indices) == 0) {
              message("No model converged. Handle this case as you see fit...")
              best_lambda <- NA
              betaest     <- NA
            } else {
              converged_models <- results_list[converged_indices]
              lambdas_converged <- sapply(converged_models, function(x) x$lambda)
              best_idx_local    <- which.min(lambdas_converged)
              best_model  <- converged_models[[best_idx_local]]
              best_lambda <- best_model$lambda
              betahat[bt,] <- as.vector(best_model$betavec_solution)
            }
          } else {
            stop('Please provide a correct family!')
          }
        }
        if (sum(is.na(betahat)) == 0){
          success <- 1
        } else {
          success <- 0
        }
      }
    }
  }
  return(list('mean'= meanhat, 'beta' = betahat, 'success' = success))
}

PPI <- function(source, target, sourcebb = NULL, targetbb = NULL, family = 'gaussian',
                SL.library = c("SL.glm", "SL.gam", "SL.glmnet", "SL.ranger", "SL.ksvm", 
                               "SL.mean", "SL.randomForest", "SL.xgboost"),
                bootstrap = FALSE, CF = FALSE){
  if (CF == FALSE){
    return(PPIwcf(source, target, sourcebb, targetbb, family = family,
                  SL.library = SL.library, bootstrap = bootstrap))
  } else {
    set.seed(2025)
    # generate CF index
    tcf.idx <- sample(1:nrow(target$x), size = ceiling(0.5 * nrow(target$x)), replace = FALSE)
    ntcf.idx <- (1:nrow(target$x))[-tcf.idx]
    scf.idx <- sample(1:nrow(source$x), size = ceiling(0.5 * nrow(source$x)), replace = FALSE)
    nscf.idx <- (1:nrow(source$x))[-scf.idx]
    xtg <- target$x
    xsc <- source$x
    ysc <- source$y
    ## estimations
    ## estimate source
    pihat <- nrow(xtg)/(nrow(xtg) + nrow(xsc))
    xcomb <- as.data.frame(rbind(xtg, xsc))
    rind <- c(rep(0, length(tcf.idx)), rep(1, length(scf.idx)))
    ## estimate pi(x) with super learner
    if ((family == 'gaussian')|!(FALSE %in% (SL.library == c('SL.glm')))){
      sl_stacked <- SuperLearner(Y = rind, X = as.data.frame(rbind(xtg[tcf.idx, ], xsc[scf.idx, ])), 
                                 family = binomial(),
                                 SL.library = SL.library, verbose = F, method = "method.NNloglik")
      predval <- predict(sl_stacked, newdata = as.data.frame(rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ])))$pred
      phat <- predval[(length(ntcf.idx)+1):(length(ntcf.idx) + length(nscf.idx))]
      phatall <- predval
    } else {
      sl_stacked <- SuperLearner(Y = rind, X = as.data.frame(rbind(xtg[tcf.idx, ], xsc[scf.idx, ])), 
                                 family = binomial(),
                                 SL.library = SL.library, verbose = F, method = "method.NNloglik")
      predval <- predict(sl_stacked, newdata = as.data.frame(rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ])))$pred
      phat <- predval[(length(ntcf.idx)+1):(length(ntcf.idx) + length(nscf.idx))]
      phatall <- predval
    }
    what <- pihat / (1 - pihat) * (1 / phat - 1)
    if (is.null(sourcebb) & is.null(targetbb)){
      ## estimate E(Y|X)
      if (family == 'gaussian'){
        sl_stacked <- glm(Y = ysc[scf.idx], X = as.data.frame(xsc[scf.idx, ]), family = gaussian(),
                                   SL.library = SL.library, verbose = F, method = "method.NNloglik")
        yhat <- predict(sl_stacked, newdata = as.data.frame(rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ])))$pred
      } else if (family == 'binomial'){
        if (!(FALSE %in% (SL.library == c('SL.glm')))){
          stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
        } else {
          sl_stacked <- SuperLearner(Y = ysc[scf.idx], X = as.data.frame(xsc[scf.idx, ]), family = binomial(),
                                     SL.library = SL.library, verbose = F, method = "method.NNloglik")
          yhat <- predict(sl_stacked, newdata = as.data.frame(rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ])))$pred
        }
        yhattg <- yhat[1:length(ntcf.idx)]
        yhatsc <- yhat[(length(ntcf.idx) + 1): (length(ntcf.idx) + length(nscf.idx))] 
      } else {
        stop('Please provide a correct family!')
      }
      if (bootstrap == FALSE){
        ## solve mean equation
        scpart <- sum(what * (ysc[nscf.idx] - yhatsc) / pihat)
        tgpart <- sum(yhattg / (1 - pihat))
        meanhat <- 1 / (nrow(xtg[ntcf.idx, ]) / (1 - pihat)) * (scpart + tgpart)
        ## solve param equation
        if (family == 'gaussian'){
          stop('Please provide a correct family!')
        } else if (family == 'binomial') {
          avec <- t(what * (ysc[nscf.idx] - yhatsc) / pihat)  %*% cbind(rep(1, nrow(xsc[nscf.idx, ])), xsc[nscf.idx, ]) 
          penalized_objective <- function(betavec, lambda) {
            unpenalized_loss(avec, betavec, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat) + lambda * sum(betavec^2)
          }
          fit_for_lambda <- function(lambda, start_par = NULL) {
            if (is.null(start_par)) {
              start_par <- rep(0, ncol(xtg) + 1)
            }
            solution <- tryCatch(
              expr = {
                optim(
                  par    = start_par,
                  fn     = function(b) penalized_objective(b, lambda),
                  method = "BFGS"
                )
              },
              warning = function(w) {
                message("Warning in optim: ", w$message)
                list('par' = start_par, 'convergence' = 1)
              },
              error = function(e) {
                message("Error in optim: ", e$message)
                list('par' = start_par, 'convergence' = 1)
              }
            )
            return(solution)
          }
          lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
          results_list <- vector("list", length(lambda_grid))
          names(results_list) <- paste0("lambda=", lambda_grid)
          for (i in seq_along(lambda_grid)) {
            this_lambda <- lambda_grid[i]
            fit <- fit_for_lambda(this_lambda)
            val_unpenalized <- unpenalized_loss(avec, fit$par, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat)
            results_list[[i]] <- list(
              lambda           = this_lambda,
              betavec_solution = fit$par,
              penalized_value  = fit$value,  
              unpenalized_loss = val_unpenalized,
              convergence      = fit$convergence
            )
          }
          converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
          if (length(converged_indices) == 0) {
            message("No model converged. Handle this case as you see fit...")
            best_lambda <- NA
            betahat     <- NA
            success     <- 0
          } else {
            converged_models <- results_list[converged_indices]
            lambdas_converged <- sapply(converged_models, function(x) x$lambda)
            best_idx_local    <- which.min(lambdas_converged)
            best_model  <- converged_models[[best_idx_local]]
            best_lambda <- best_model$lambda
            betahat     <- as.vector(best_model$betavec_solution)
            success     <- 1
          }
        }
      } else {
        meanhat <- c()
        betahat <- matrix(NA, nrow = 500, ncol = ncol(xtg) + 1)
        for (bt in 1:500){
          # generate ksi
          ksivec <- rexp(n = length(ntcf.idx) + length(nscf.idx), 1)
          ksitg <- ksivec[1:length(ntcf.idx)]
          ksisc <- ksivec[(length(ntcf.idx) + 1): (length(ntcf.idx) + length(nscf.idx))]
          ## solve mean equation
          scpart <- sum(what * (ysc[nscf.idx] - yhatsc) / pihat * ksisc)
          tgpart <- sum(yhattg / (1 - pihat) * ksitg)
          meanhat[bt] <- 1 / (sum(ksitg)/ (1 - pihat)) * (scpart + tgpart)
          ## solve param equation
          if (family == 'gaussian'){
            stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
          } else if (family == 'binomial') {
            avec <- t(what * (ysc[nscf.idx] - yhatsc) / pihat * ksisc)  %*% cbind(rep(1, nrow(xsc[nscf.idx, ])), xsc[nscf.idx, ]) 
            penalized_objective <- function(betavec, lambda) {
              unpenalized_loss_boot(avec, betavec, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat, ksitg) + lambda * sum(betavec^2)
            }
            fit_for_lambda <- function(lambda, start_par = NULL) {
              if (is.null(start_par)) {
                start_par <- rep(0, ncol(xtg) + 1)
              }
              solution <- tryCatch(
                expr = {
                  optim(
                    par    = start_par,
                    fn     = function(b) penalized_objective(b, lambda),
                    method = "BFGS"
                  )
                },
                warning = function(w) {
                  message("Warning in optim: ", w$message)
                  list('par' = start_par, 'convergence' = 1)
                },
                error = function(e) {
                  message("Error in optim: ", e$message)
                  list('par' = start_par, 'convergence' = 1)
                }
              )
              return(solution)
            }
            lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
            results_list <- vector("list", length(lambda_grid))
            names(results_list) <- paste0("lambda=", lambda_grid)
            for (i in seq_along(lambda_grid)) {
              this_lambda <- lambda_grid[i]
              fit <- fit_for_lambda(this_lambda)
              val_unpenalized <- unpenalized_loss_boot(avec, fit$par, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat, ksitg)
              results_list[[i]] <- list(
                lambda           = this_lambda,
                betavec_solution = fit$par,
                penalized_value  = fit$value,  
                unpenalized_loss = val_unpenalized,
                convergence      = fit$convergence
              )
            }
            converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
            if (length(converged_indices) == 0) {
              message("No model converged. Handle this case as you see fit...")
              best_lambda <- NA
              betaest     <- NA
              success     <- 0
            } else {
              converged_models <- results_list[converged_indices]
              lambdas_converged <- sapply(converged_models, function(x) x$lambda)
              best_idx_local    <- which.min(lambdas_converged)
              best_model  <- converged_models[[best_idx_local]]
              best_lambda <- best_model$lambda
              betahat[bt,]     <- as.vector(best_model$betavec_solution)
            }
          }
        }
        if (sum(is.na(betahat)) == 0){
          success <- 1
        } else {
          success <- 0
        }
      }
    } else {
      if (is.null(sourcebb) | is.null(targetbb)){
        stop('The black box model for source and target should be simotaneously zero or nonzero!')
      } else{
        if (family == 'gaussian'){
          stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
        } else if (family == 'binomial'){
          ## estimate E(Y|X,Yhat)
          xscb <- cbind(xsc, sourcebb)
          xtgb <- cbind(xtg, targetbb)
          xcomb <- rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ])
          if (!(FALSE %in% (SL.library == c('SL.glm')))){
            stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
          } else {
            sl_stacked <- SuperLearner(Y = ysc[scf.idx], X = as.data.frame(xscb[scf.idx, ]), family = binomial(),
                                       SL.library = SL.library, verbose = F, method = "method.NNloglik")
            newx <- as.data.frame(rbind(xtgb[ntcf.idx, ], xscb[nscf.idx, ]))
            names(newx) <- names(as.data.frame(xscb[scf.idx, ]))
            yhatb <- predict(sl_stacked, newdata = newx)$pred
          }
          yhatbtg <- yhatb[1:length(ntcf.idx)]
          yhatbsc <- yhatb[(length(ntcf.idx) + 1): (length(ntcf.idx) + length(nscf.idx))]
          ## estimate E(Y|X)
          if (!(FALSE %in% (SL.library == c('SL.glm')))){
            stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
          } else{
            sl_stacked <- SuperLearner(Y = ysc[scf.idx], X = as.data.frame(xsc[scf.idx, ]), family = binomial(),
                                       SL.library = SL.library, verbose = F, method = "method.NNloglik")
            newx <- as.data.frame(rbind(xtg[ntcf.idx, ], xsc[nscf.idx, ]))
            names(newx) <- names(as.data.frame(xsc[scf.idx, ]))
            yhat <- predict(sl_stacked, newdata = newx)$pred
          }
          yhattg <- yhat[1:length(ntcf.idx)]
          yhatsc <- yhat[(length(ntcf.idx) + 1): (length(ntcf.idx) + length(nscf.idx))]
        } else {
          stop('Please provide a correct family!')
        }
        if (bootstrap == FALSE) {
          ## solve mean equation
          scpart <- sum(what * (ysc[nscf.idx] - yhatbsc) / pihat)
          mutpart <- sum((1 - phatall) * (yhatb - yhat) / (1 - pihat))
          tgpart <- sum(yhattg / (1 - pihat))
          meanhat <- 1 / (length(ntcf.idx) / (1 - pihat)) * (scpart + mutpart + tgpart) 
          ## solve equation
          if (family == 'gaussian'){
            stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
          } else if (family == 'binomial'){
            scpart <- t(what * (ysc[nscf.idx] - yhatbsc) / pihat) %*% cbind(rep(1, nrow(xsc[nscf.idx, ])), xsc[nscf.idx, ]) 
            mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat)) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb)) 
            avec <- scpart + mutpart
            penalized_objective <- function(betavec, lambda) {
              unpenalized_loss(avec, betavec, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat) + lambda * sum(betavec^2)
            }
            fit_for_lambda <- function(lambda, start_par = NULL) {
              if (is.null(start_par)) {
                start_par <- rep(0, ncol(xtg) + 1)
              }
              solution <- tryCatch(
                expr = {
                  optim(
                    par    = start_par,
                    fn     = function(b) penalized_objective(b, lambda),
                    method = "BFGS"
                  )
                },
                warning = function(w) {
                  message("Warning in optim: ", w$message)
                  list('par' = start_par, 'convergence' = 1)
                },
                error = function(e) {
                  message("Error in optim: ", e$message)
                  list('par' = start_par, 'convergence' = 1)
                }
              )
              return(solution)
            }
            lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
            results_list <- vector("list", length(lambda_grid))
            names(results_list) <- paste0("lambda=", lambda_grid)
            for (i in seq_along(lambda_grid)) {
              this_lambda <- lambda_grid[i]
              fit <- fit_for_lambda(this_lambda)
              val_unpenalized <- unpenalized_loss(avec, fit$par, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat)
              results_list[[i]] <- list(
                lambda           = this_lambda,
                betavec_solution = fit$par,
                penalized_value  = fit$value,  
                unpenalized_loss = val_unpenalized,
                convergence      = fit$convergence
              )
            }
            converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
            if (length(converged_indices) == 0) {
              message("No model converged. Handle this case as you see fit...")
              best_lambda <- NA
              betahat     <- NA
              success     <- 0
            } else {
              converged_models <- results_list[converged_indices]
              lambdas_converged <- sapply(converged_models, function(x) x$lambda)
              best_idx_local    <- which.min(lambdas_converged)
              best_model  <- converged_models[[best_idx_local]]
              best_lambda <- best_model$lambda
              betahat     <- as.vector(best_model$betavec_solution)
              success     <- 1
            }
          } else {
            stop('Please provide a correct family!')
          }
        } else {
          meanhat <- c()
          betahat <- matrix(NA, nrow = 500, ncol = ncol(xtg) + 1)
          for (bt in 1:500){
            # generate ksi
            ksivec <- rexp(n = length(ntcf.idx) + length(nscf.idx), 1)
            ksitg <- ksivec[1:length(ntcf.idx)]
            ksisc <- ksivec[(length(ntcf.idx) + 1): (length(ntcf.idx) + length(nscf.idx))]
            ## solve mean equation
            scpart <- sum(what * (ysc[nscf.idx] - yhatbsc) / pihat * ksisc)
            mutpart <- sum((1 - phatall) * (yhatb - yhat) / (1 - pihat) * ksivec)
            tgpart <- sum(yhattg / (1 - pihat) * ksitg)
            meanhat[bt] <- 1 / ((sum(ksitg))/ (1 - pihat)) * (scpart + mutpart + tgpart) 
            ## solve equation
            if (family == 'gaussian'){
              stop('You dont need CF=TRUE, please let CF=FALSE and run again!')
            } else if (family == 'binomial'){
              scpart <- t(what * (ysc[nscf.idx] - yhatbsc) / pihat * ksisc) %*% cbind(rep(1, nrow(xsc[nscf.idx, ])), xsc[nscf.idx, ]) 
              mutpart <- t((1 - phatall) * (yhatb - yhat) / (1 - pihat) * ksivec) %*% cbind(rep(1, nrow(xcomb)), as.matrix(xcomb)) 
              avec <- scpart + mutpart
              penalized_objective <- function(betavec, lambda) {
                unpenalized_loss_boot(avec, betavec, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat, ksitg) + lambda * sum(betavec^2)
              }
              fit_for_lambda <- function(lambda, start_par = NULL) {
                if (is.null(start_par)) {
                  start_par <- rep(0, ncol(xtg) + 1)
                }
                solution <- tryCatch(
                  expr = {
                    optim(
                      par    = start_par,
                      fn     = function(b) penalized_objective(b, lambda),
                      method = "BFGS"
                    )
                  },
                  warning = function(w) {
                    message("Warning in optim: ", w$message)
                    list('par' = start_par, 'convergence' = 1)
                  },
                  error = function(e) {
                    message("Error in optim: ", e$message)
                    list('par' = start_par, 'convergence' = 1)
                  }
                )
                return(solution)
              }
              lambda_grid <- c(0, 1e-4, 1e-3, 1e-2, 1e-1, 1)
              results_list <- vector("list", length(lambda_grid))
              names(results_list) <- paste0("lambda=", lambda_grid)
              for (i in seq_along(lambda_grid)) {
                this_lambda <- lambda_grid[i]
                fit <- fit_for_lambda(this_lambda)
                val_unpenalized <- unpenalized_loss_boot(avec, fit$par, xtg[ntcf.idx, ], xsc[nscf.idx, ], yhattg, pihat, ksitg)
                results_list[[i]] <- list(
                  lambda           = this_lambda,
                  betavec_solution = fit$par,
                  penalized_value  = fit$value,  
                  unpenalized_loss = val_unpenalized,
                  convergence      = fit$convergence
                )
              }
              converged_indices <- which(sapply(results_list, function(x) x$convergence) == 0)
              if (length(converged_indices) == 0) {
                message("No model converged. Handle this case as you see fit...")
                best_lambda <- NA
                betaest     <- NA
              } else {
                converged_models <- results_list[converged_indices]
                lambdas_converged <- sapply(converged_models, function(x) x$lambda)
                best_idx_local    <- which.min(lambdas_converged)
                best_model  <- converged_models[[best_idx_local]]
                best_lambda <- best_model$lambda
                betahat[bt,] <- as.vector(best_model$betavec_solution)
              }
            } else {
              stop('Please provide a correct family!')
            }
          }
          if (sum(is.na(betahat)) == 0){
            success <- 1
          } else {
            success <- 0
          }
        }
      }
    }
    return(list('mean'= meanhat, 'beta' = betahat, 'success' = success))
  }
}
