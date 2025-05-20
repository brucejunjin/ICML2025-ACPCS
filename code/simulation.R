# Load dependency
library(MASS)
library(SuperLearner)
library(caret)
library(parallel)
source('functions.R')
source('superlearner.R')

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
      for (ntarget in c(300, 600, 900, 1200, 1500)){
        for (nsource in c(300, 600, 900, 1200, 1500)){
          if ((ntarget != 300) & (nsource != 300)){
            next
          }
          beta <- c(1, 1, rep(0.5, 4), alpha)
          ite <- 500 
          
          # Placeholders
          ytrue.mean <- c()
          ytg.mean <- c()
          wbb.mean <- c()
          bb.mean <- c()
          wbb.list <- list()
          bb.list <- list()
          ytg.list <- list()
          betatrue.list <- list()
          betatg.list <- list()
          ywbb.list <- list()
          ybb.list <- list()
          ytgest.list <- list()
          
          seedcount <- 1
          while (ite != 0){
            print(500 - ite + 1)
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
            
            wbb.est <- PPI(source = source, target = target, family = family, SL.library = SL.library)
            bb.est <- PPI(source = source, target = target, sourcebb = Yhatsc, targetbb = Yhattg, 
                          family = family, SL.library = SL.library)
            
            if (family == 'gaussian'){
              wbb.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% wbb.est$beta
              bb.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% bb.est$beta
              ytg.pred <- cbind(rep(1, nrow(Xtg)), Xtg) %*% betatg.list[[ite]]
            } else if (family == 'binomial'){
              wbblinear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% wbb.est$beta
              bblinear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% bb.est$beta
              ytg.linear <- cbind(rep(1, nrow(Xtg)), Xtg) %*% betatg.list[[ite]]
              wbb.pred <- exp(wbblinear) / (1 + exp(wbblinear))
              bb.pred <- exp(bblinear) / (1 + exp(bblinear))
              ytg.pred <- exp(ytg.linear) / (1 + exp(ytg.linear))
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
              ite <- ite - 1
            }
          }
          
          if (family == 'gaussian'){
            famname <- 'Gaussian'
          } else{
            famname <- 'Binomial'
          }
          
          save(wbb.mean, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(bb.mean, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_mean.rda'))
          save(ytg.mean, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytg.rda'))
          save(ytrue.mean, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytrue.rda'))
          
          save(wbb.list, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(bb.list, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_beta.rda'))
          save(ytg.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_ytruelist.rda'))
          save(ywbb.list, file = paste0("./output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(ybb.list, file = paste0("./output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          save(ytgest.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_predict.rda'))
          
          save(betatrue.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_betatrue.rda'))
          save(betatg.list, file = paste0("./output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, '_ns', nsource, '_nt', ntarget, '_betatg.rda'))
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
