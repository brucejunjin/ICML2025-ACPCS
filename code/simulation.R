# Load dependency
library(MASS)
library(SuperLearner)
library(caret)
library(parallel)
source('functions.R')
source('superlearner.R')
source('functions_ols.R')
source('functions_ppi.R')
source('functions_reppi.R')

# Define the parameters in the study
seed <- 2025
n <-  40000
SL.library <- c("SL.glm")
eta <- c(0, rep(1, 5))

for (family in c('gaussian', 'binomial')){
  for (alpha in 0:5){
    for (rho in c(0, 0.3, 0.6, 0.9, 1)){
      if ((alpha != 5) & (rho != 0)){
        next
      }
      print(paste0('Family: ', family, '; alpha: ', alpha, 
                   '; rho: ', rho))
      for (ntarget in c(300, 600, 900, 1200, 1500)){
        for (nsource in c(300, 600, 900, 1200, 1500)){
          if ((ntarget != 300) & (nsource != 300)){
            next
          }
          beta <- c(1, 1, rep(0.5, 4), alpha)
          ite <- 500 
          
          # Placeholders about mean
          ytrue.mean <- c()
          ytg.mean <- c()
          wbb.mean <- c()
          bb.mean <- c()
          xy.mean <- c()
          ppi.mean <- c()
          ppipp.mean <- c()
          reppi.mean <- c()
          
          # Placeholders about beta
          wbb.list <- list()
          bb.list <- list()
          betatrue.list <- list()
          betatg.list <- list()
          xy.list <- list()
          ppi.list <- list()
          ppipp.list <- list()
          reppi.list <- list()
          
          # Placeholders about y
          ytg.list <- list()
          ywbb.list <- list()
          ybb.list <- list()
          ytgest.list <- list()
          yxy.list <- list()
          yppi.list <- list()
          yppipp.list <- list()
          yreppi.list <- list()
          
          seedcount <- 1
          while (ite != 0){
            #print(500 - ite + 1)
            set.seed(seed + seedcount)
            seedcount = seedcount + 1
            # Covariate + response generation
            meanvec <- rep(0, 6)
            covmat <- diag(6)
            covmat[1, 6] <- rho
            covmat[6, 1] <- rho
            XZ <- mvrnorm(n = n, mu = meanvec, Sigma = covmat)
            XZ1 <- cbind(rep(1, n), XZ)
            if (family == 'gaussian'){
              eps <- mvrnorm(n = n, mu = c(0), Sigma = c(1))
              Y <- XZ1 %*% beta + eps
            } else {
              prob <- exp(XZ1 %*% beta)/ (1 + exp(XZ1 %*% beta))
              Y <- rbinom(n = n, size = 1, prob = prob)
              XZ1[, 7] <- exp(XZ1[, 7] * alpha)/(1 + exp(XZ1[, 7] * alpha))
            }
            
            # R generation for group 
            prob <- exp(XZ1[, 1:6] %*% eta)/ (1 + exp(XZ1[, 1:6] %*% eta))
            r <- rbinom(n = n, size = 1, prob = prob)
            index0 <- which(r == 0)[1:nsource]
            index1 <- which(r == 1)[1:ntarget]
            indexselect <- c(index0, index1)
            
            # First store Mont Carlo
            yall.mean <- mean(Y[which(r == 1)])
            ytrue.mean[ite] <- yall.mean
            if (family == 'gaussian'){
              data <- data.frame(y = Y[which(r == 1)], XZ1[which(r == 1), 2:6])
              names(data)[1] <- 'y'
              lmodel <- lm(y ~ ., data = data)
              betatrue.list[[ite]] <- as.vector(lmodel$coefficients[-1])
            } else if (family == 'binomial'){
              data <- data.frame(y = Y[which(r == 1)], XZ1[which(r == 1), 2:6])
              names(data)[1] <- 'y'
              glmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
              betatrue.list[[ite]] <- as.vector(glmodel$coefficients[-1])
            } else {
              stop('Please provide a correct family!')
            }
            
            # limit to select and split covariate and group
            XZ1 <- XZ1[indexselect, ]
            Y <- Y[indexselect]
            r <- r[indexselect]
            Xtg <- XZ1[which(r == 1), 2:6]
            Xsc <- XZ1[which(r == 0), 2:6]
            Ytg <- Y[which(r == 1)]
            Ysc <- Y[which(r == 0)]
            Yhattg <- XZ1[which(r == 1), 7]
            Yhatsc <- XZ1[which(r == 0), 7]
            
            target <- list('x' = Xtg)
            source <- list('x' = Xsc, 'y' = Ysc)
            
            # store target only 
            if (family == 'gaussian'){
              data <- data.frame(y = Ytg, Xtg)
              names(data)[1] <- 'y'
              lmodel <- lm(y ~ ., data = data)
              betatg.list[[ite]] <- as.vector(lmodel$coefficients)
            } else if (family == 'binomial'){
              data <- data.frame(y = Ytg, Xtg)
              names(data)[1] <- 'y'
              glmodel <- glm(y ~ ., data = data, family = binomial(link = "logit"))
              betatg.list[[ite]] <- as.vector(glmodel$coefficients)
            } else {
              stop('Please provide a correct family!')
            }
            
            # store xy only, ppi, ppi++, reppi
            Xtg1 <- as.matrix(cbind(1, Xtg))
            Xsc1 <- as.matrix(cbind(1, Xsc))
            if (family == 'gaussian'){
              # xy only
              fit.xy <- classical_ols_ci(Xsc1, as.matrix(Ysc), alpha = 0.05)
              xy.list[[ite]] <- (fit.xy$lower + fit.xy$upper)/2
              fit.xy <- classical_mean_ci(as.matrix(Ysc), alpha = 0.05)
              xy.mean[ite] <- (fit.xy$lower + fit.xy$upper)/2
              # ppi
              fit.ppi <- ppi_ols_ci(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05, lhat = 1)
              ppi.list[[ite]] <- as.vector((fit.ppi$lower + fit.ppi$upper)/2)
              fit.ppi <- ppi_mean_ci(as.matrix(Ysc), as.matrix(Yhatsc), as.matrix(Yhattg), alpha = 0.05, lhat = 1)
              ppi.mean[ite] <- (fit.ppi$lower + fit.ppi$upper)/2
              # ppi++
              fit.ppipp <- ppi_ols_ci(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05)
              ppipp.list[[ite]] <- as.vector((fit.ppipp$lower + fit.ppipp$upper)/2)
              fit.ppipp <- ppi_mean_ci(as.matrix(Ysc), as.matrix(Yhatsc), as.matrix(Yhattg), alpha = 0.05)
              ppipp.mean[ite] <- (fit.ppipp$lower + fit.ppipp$upper)/2
              # reppi
              fit.reppi <- ppi_opt_ols_ci_crossfit(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05, method = 'linreg')
              reppi.list[[ite]] <- as.vector((fit.reppi$lower + fit.reppi$upper)/2)
            } else if (family == 'binomial'){
              # xy only
              fit.xy <- classical_logistic_ci(Xsc1, as.matrix(Ysc), alpha = 0.05)
              xy.list[[ite]] <- (fit.xy$lower + fit.xy$upper)/2
              fit.xy <- classical_mean_ci(as.matrix(Ysc), alpha = 0.05)
              xy.mean[ite] <- (fit.xy$lower + fit.xy$upper)/2
              # ppi
              fit.ppi <- ppi_logistic_ci(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05, lhat = 1)
              ppi.list[[ite]] <- as.vector((fit.ppi$lower + fit.ppi$upper)/2)
              fit.ppi <- ppi_mean_ci(as.matrix(Ysc), as.matrix(Yhatsc), as.matrix(Yhattg), alpha = 0.05, lhat = 1)
              ppi.mean[ite] <- (fit.ppi$lower + fit.ppi$upper)/2
              # ppi++
              fit.ppipp <- ppi_logistic_ci(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05)
              ppipp.list[[ite]] <- as.vector((fit.ppipp$lower + fit.ppipp$upper)/2)
              fit.ppipp <- ppi_mean_ci(as.matrix(Ysc), as.matrix(Yhatsc), as.matrix(Yhattg), alpha = 0.05)
              ppipp.mean[ite] <- (fit.ppipp$lower + fit.ppipp$upper)/2
              # reppi
              fit.reppi <- ppi_opt_ols_ci_crossfit(Xsc1, as.matrix(Ysc), as.matrix(Yhatsc), Xtg1, as.matrix(Yhattg), alpha = 0.05, method = 'logistic')
              reppi.list[[ite]] <- as.vector((fit.reppi$lower + fit.reppi$upper)/2)
            } else {
              stop('Please provide a correct family!')
            }
            
            wbb.est <- PPI(source = source, target = target, family = family, SL.library = SL.library)
            bb.est <- PPI(source = source, target = target, sourcebb = Yhatsc, targetbb = Yhattg, 
                          family = family, SL.library = SL.library)
            
            if (family == 'gaussian'){
              wbb.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% wbb.est$beta
              bb.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% bb.est$beta
              ytg.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% betatg.list[[ite]]
              xy.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% xy.list[[ite]]
              ppi.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% ppi.list[[ite]]
              ppipp.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% ppipp.list[[ite]]
              reppi.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% reppi.list[[ite]]
              reppi.mean[ite] <- mean(reppi.pred)
            } else if (family == 'binomial'){
              wbblinear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% wbb.est$beta
              bblinear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% bb.est$beta
              ytg.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% betatg.list[[ite]]
              xy.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% xy.list[[ite]]
              ppi.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% ppi.list[[ite]]
              ppipp.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% ppipp.list[[ite]]
              reppi.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% reppi.list[[ite]]
              wbb.pred <- exp(wbblinear) / (1 + exp(wbblinear))
              bb.pred <- exp(bblinear) / (1 + exp(bblinear))
              ytg.pred <- exp(ytg.linear) / (1 + exp(ytg.linear))
              xy.pred <- exp(xy.linear) / (1 + exp(xy.linear))
              ppi.pred <- exp(ppi.linear) / (1 + exp(ppi.linear))
              ppipp.pred <- exp(ppipp.linear) / (1 + exp(ppipp.linear))
              reppi.pred <- exp(reppi.linear) / (1 + exp(reppi.linear))
              reppi.mean[ite] <- mean(reppi.pred)
            }
            if ((wbb.est$success == 0)|(bb.est$success == 0)){
              next
            } else{
              ytg.mean[ite] <- mean(Ytg)
              wbb.mean[ite] <- wbb.est$mean
              bb.mean[ite] <- bb.est$mean
              wbb.list[[ite]] <- wbb.est$beta
              bb.list[[ite]] <- bb.est$beta
              ytg.list[[ite]] <- Ytg
              ywbb.list[[ite]] <- as.vector(wbb.pred)
              ybb.list[[ite]] <- as.vector(bb.pred)
              ytgest.list[[ite]] <- as.vector(ytg.pred)
              yxy.list[[ite]] <- as.vector(xy.pred)
              yppi.list[[ite]] <- as.vector(ppi.pred)
              yppipp.list[[ite]] <- as.vector(ppipp.pred)
              yreppi.list[[ite]] <- as.vector(reppi.pred)
              ite <- ite - 1
            }
          }
          
          if (family == 'gaussian'){
            famname <- 'Gaussian'
          } else{
            famname <- 'Binomial'
          }
          # about mean
          save(ytrue.mean, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytrue.rda'))
          save(wbb.mean, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(bb.mean, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(ytg.mean, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytg.rda'))
          # add 4 compares
          save(xy.mean, file = paste0("./output/xyonly/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(ppi.mean, file = paste0("./output/ppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(ppipp.mean, file = paste0("./output/ppipp/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(reppi.mean, file = paste0("./output/reppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          
          # about beta
          save(wbb.list, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(bb.list, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(betatrue.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_betatrue.rda'))
          save(betatg.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_betatg.rda'))
          # add 4 compares
          save(xy.list, file = paste0("./output/xyonly/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(ppi.list, file = paste0("./output/ppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(ppipp.list, file = paste0("./output/ppipp/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(reppi.list, file = paste0("./output/reppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          
          # abount y
          save(ytg.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytruelist.rda'))
          save(ywbb.list, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(ybb.list, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(ytgest.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          # add 4 compares
          save(yxy.list, file = paste0("./output/xyonly/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(yppi.list, file = paste0("./output/ppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(yppipp.list, file = paste0("./output/ppipp/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(yreppi.list, file = paste0("./output/reppi/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
        }
      }
    }
  }
}


#gapmat <- matrix(unlist(wbb.list), ncol = 6, byrow = T) - beta[1:6] 
#gapmatbb <- matrix(unlist(bb.list), ncol = 6, byrow = T) - beta[1:6]

#apply(gapmat, 2, trimmed_mean) 
#apply(gapmatbb, 2, trimmed_mean) 
#apply(gapmat, 2, trimmed_sd) 
#apply(gapmatbb, 2, trimmed_sd) 
#apply(gapmat^2, 2, trimmed_mean)  
#apply(gapmatbb^2, 2, trimmed_mean) 

#trimmed_mean((wbb.mean - ytrue.vec))
#trimmed_mean((bb.mean - ytrue.vec))
#trimmed_sd((wbb.mean - ytrue.vec))
#trimmed_sd((bb.mean - ytrue.vec))
#trimmed_mean((wbb.mean - ytrue.vec)^2)
#trimmed_mean((bb.mean - ytrue.vec)^2)
