### load pack
library(ggplot2)
library(dplyr)
library(ggrepel) 
library(pROC)
library(patchwork)
### Prelim
tmrate <- 0.05
trimmed_mean <- function(x) {
  mean(x, trim = tmrate)
}
trimmed_sd <- function(x) {
  trim = tmrate
  trimmed_values <- sort(x)[floor(trim * length(x)) + 1 : ceiling((1 - trim) * length(x))]
  trimmed_mean <- mean(trimmed_values)
  sqrt(mean((trimmed_values - trimmed_mean)^2))
}

for (famname in c('Gaussian', 'Binomial')){
  for (measure in c('mean', 'beta1', 'beta2', 'pred')){
    
    ################# Template for alpha + nsource (Fig 1) ##################
    # Fix param
    ntarget <- 300
    rho <- 0
    famname <- famname
    # Range param
    nsource.list <- c(300, 600, 900, 1200, 1500)
    alpha.list <- c(0, 1, 2, 3, 4, 5)
    # Load data
    df1 <- data.frame(
      nsource = numeric(0),
      alpha  = character(0),
      ARE   = numeric(0)
    )
    for (nsource in nsource.list){
      for (alpha in alpha.list){
        wbb.mean <- paste0("../output/wbb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                           "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = wbb.mean)  
        bb.mean <- paste0("../output/bb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                          "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = bb.mean) 
        ytg.mean <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_ytg.rda')
        load(file = ytg.mean) 
        ytrue.mean <- paste0("../output/true/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                             "_ns", nsource, "_nt", ntarget, "_ytrue.rda")
        load(file = ytrue.mean)
        wbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = wbb.list)
        bb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                          '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = bb.list)
        ytg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                           '_ns', nsource, '_nt', ntarget, '_ytruelist.rda')
        load(file = ytg.list)
        ywbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                            '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ywbb.list)
        ybb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ybb.list)
        betatg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                              '_ns', nsource, '_nt', ntarget, '_betatg.rda')
        load(file = betatg.list)
        betatrue.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                                '_ns', nsource, '_nt', ntarget, '_betatrue.rda')
        load(file = betatrue.list)
        ytgest.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                              '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ytgest.list)
        if (measure == 'mean'){
          wbbvalue <- trimmed_mean((wbb.mean - ytrue.mean)^2)
          bbvalue <- trimmed_mean((bb.mean - ytrue.mean)^2)
        } else if (measure == 'beta1'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
        } else if (measure == 'beta2'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
        } else if (measure == 'pred'){
          if (famname == 'Gaussian'){
            wbbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ywbb.list), function(x) mean(x^2)))
            bbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ybb.list), function(x) mean(x^2)))
          } else if (famname == 'Binomial'){
            wbbvalue <- numeric(length(ytg.list))
            bbvalue <- numeric(length(ytg.list))
            for (i in seq_along(ytg.list)) {
              true_labels <- ytg.list[[i]]      
              predicted_bb <- ybb.list[[i]] 
              predicted_wbb <- ywbb.list[[i]]
              wbbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_wbb, direction = ">")))
              bbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_bb, direction = ">")))
            }
            wbbvalue <- 1/trimmed_mean(wbbvalue)
            bbvalue <- 1/trimmed_mean(bbvalue)
          }
        }
        # input data
        df1 <- rbind(df1,
                    data.frame(nsource = nsource,
                               alpha  = alpha,
                               ARE   = wbbvalue/bbvalue))
      }
    }
    df1 <- df1 %>%
      mutate(
        rowfacet   = "ntarget = 300, nsource changes",  # top row
        rowfacet   = "N = 300, n changes",  # top row
        colfacet   = "alpha",             # left column
        xvar       = nsource,             # x-axis is nsource
        groupvar   = alpha                # color/shape/linetype by alpha
      )
    
    ################# Template for alpha + ntarget (Fig 2) ##################
    # Fix param
    nsource <- 300
    rho <- 0
    famname <- famname
    # Range param
    ntarget.list <- c(300, 600, 900, 1200, 1500)
    alpha.list <- c(0, 1, 2, 3, 4, 5)
    # Load data
    df2 <- data.frame(
      ntarget = numeric(0),
      alpha  = character(0),
      ARE   = numeric(0)
    )
    for (ntarget in ntarget.list){
      for (alpha in alpha.list){
        wbb.mean <- paste0("../output/wbb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                           "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = wbb.mean)  
        bb.mean <- paste0("../output/bb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                          "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = bb.mean) 
        ytg.mean <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_ytg.rda')
        load(file = ytg.mean) 
        ytrue.mean <- paste0("../output/true/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                             "_ns", nsource, "_nt", ntarget, "_ytrue.rda")
        load(file = ytrue.mean)
        wbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = wbb.list)
        bb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                          '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = bb.list)
        ytg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                           '_ns', nsource, '_nt', ntarget, '_ytruelist.rda')
        load(file = ytg.list)
        ywbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                            '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ywbb.list)
        ybb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ybb.list)
        betatg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                              '_ns', nsource, '_nt', ntarget, '_betatg.rda')
        load(file = betatg.list)
        betatrue.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                                '_ns', nsource, '_nt', ntarget, '_betatrue.rda')
        load(file = betatrue.list)
        ytgest.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                              '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ytgest.list)
        if (measure == 'mean'){
          wbbvalue <- trimmed_mean((wbb.mean - ytrue.mean)^2)
          bbvalue <- trimmed_mean((bb.mean - ytrue.mean)^2)
        } else if (measure == 'beta1'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
        } else if (measure == 'beta2'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
        } else if (measure == 'pred'){
          if (famname == 'Gaussian'){
            wbbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ywbb.list), function(x) mean(x^2)))
            bbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ybb.list), function(x) mean(x^2)))
          } else if (famname == 'Binomial'){
            wbbvalue <- numeric(length(ytg.list))
            bbvalue <- numeric(length(ytg.list))
            for (i in seq_along(ytg.list)) {
              true_labels <- ytg.list[[i]]      
              predicted_bb <- ybb.list[[i]] 
              predicted_wbb <- ywbb.list[[i]]
              wbbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_wbb, direction = ">")))
              bbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_bb, direction = ">")))
            }
            wbbvalue <- 1/trimmed_mean(wbbvalue)
            bbvalue <- 1/trimmed_mean(bbvalue)
          }
        }
        # input data
        df2 <- rbind(df2,
                    data.frame(ntarget = ntarget,
                               alpha  = alpha,
                               ARE   = wbbvalue/bbvalue))
      }
    }
    df2 <- df2 %>%
      mutate(
        rowfacet   = "nsource = 300, ntarget changes", # bottom row
        rowfacet   = "n = 300, N changes", # bottom row
        colfacet   = "alpha",            # left column
        xvar       = ntarget,            # x-axis is ntarget
        groupvar   = alpha
      )
    
    ################# Template for rho + nsource (Fig 3) ##################
    # Fix param
    ntarget <- 300
    alpha <- 5
    famname <- famname
    # Range param
    nsource.list <- c(300, 600, 900, 1200, 1500)
    rho.list <- c(0, 0.3, 0.6, 0.9, 1)
    # Load data
    df3 <- data.frame(
      nsource = numeric(0),
      rho  = character(0),
      ARE   = numeric(0)
    )
    for (nsource in nsource.list){
      for (rho in rho.list){
        wbb.mean <- paste0("../output/wbb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                           "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = wbb.mean)  
        bb.mean <- paste0("../output/bb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                          "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = bb.mean) 
        ytg.mean <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_ytg.rda')
        load(file = ytg.mean) 
        ytrue.mean <- paste0("../output/true/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                             "_ns", nsource, "_nt", ntarget, "_ytrue.rda")
        load(file = ytrue.mean)
        wbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = wbb.list)
        bb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                          '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = bb.list)
        ytg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                           '_ns', nsource, '_nt', ntarget, '_ytruelist.rda')
        load(file = ytg.list)
        ywbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                            '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ywbb.list)
        ybb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ybb.list)
        betatg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                              '_ns', nsource, '_nt', ntarget, '_betatg.rda')
        load(file = betatg.list)
        betatrue.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                                '_ns', nsource, '_nt', ntarget, '_betatrue.rda')
        load(file = betatrue.list)
        ytgest.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                              '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ytgest.list)
        if (measure == 'mean'){
          wbbvalue <- trimmed_mean((wbb.mean - ytrue.mean)^2)
          bbvalue <- trimmed_mean((bb.mean - ytrue.mean)^2)
        } else if (measure == 'beta1'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
        } else if (measure == 'beta2'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
        } else if (measure == 'pred'){
          if (famname == 'Gaussian'){
            wbbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ywbb.list), function(x) mean(x^2)))
            bbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ybb.list), function(x) mean(x^2)))
          } else if (famname == 'Binomial'){
            wbbvalue <- numeric(length(ytg.list))
            bbvalue <- numeric(length(ytg.list))
            for (i in seq_along(ytg.list)) {
              true_labels <- ytg.list[[i]]      
              predicted_bb <- ybb.list[[i]] 
              predicted_wbb <- ywbb.list[[i]]
              wbbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_wbb, direction = ">")))
              bbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_bb, direction = ">")))
            }
            wbbvalue <- 1/trimmed_mean(wbbvalue)
            bbvalue <- 1/trimmed_mean(bbvalue)
          }
        }
        # input data
        df3 <- rbind(df3,
                    data.frame(nsource = nsource,
                               rho  = rho,
                               ARE   = wbbvalue/bbvalue))
      }
    }
    df3 <- df3 %>%
      mutate(
        rowfacet   = "ntarget = 300, nsource changes", # top row
        rowfacet   = "N = 300, n changes", # top row
        colfacet   = "rho",              # right column
        xvar       = nsource,
        groupvar   = rho
      )
    
    ################# Template for rho + ntarget (Fig 4) ##################
    # Fix param
    nsource <- 300
    alpha <- 5
    famname <- famname
    # Range param
    ntarget.list <- c(300, 600, 900, 1200, 1500)
    rho.list <- c(0, 0.3, 0.6, 0.9, 1)
    # Load data
    df4 <- data.frame(
      ntarget = numeric(0),
      rho  = character(0),
      ARE   = numeric(0)
    )
    for (ntarget in ntarget.list){
      for (rho in rho.list){
        wbb.mean <- paste0("../output/wbb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                           "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = wbb.mean)  
        bb.mean <- paste0("../output/bb/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                          "_ns", nsource, "_nt", ntarget, "_mean.rda")
        load(file = bb.mean) 
        ytg.mean <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_ytg.rda')
        load(file = ytg.mean) 
        ytrue.mean <- paste0("../output/true/", famname, "_alpha", alpha*10, "_rho", rho*10, 
                             "_ns", nsource, "_nt", ntarget, "_ytrue.rda")
        load(file = ytrue.mean)
        wbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = wbb.list)
        bb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                          '_ns', nsource, '_nt', ntarget, '_beta.rda')
        load(file = bb.list)
        ytg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                           '_ns', nsource, '_nt', ntarget, '_ytruelist.rda')
        load(file = ytg.list)
        ywbb.list <- paste0("../output/wbb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                            '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ywbb.list)
        ybb.list <- paste0("../output/bb/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                           '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ybb.list)
        betatg.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10,
                              '_ns', nsource, '_nt', ntarget, '_betatg.rda')
        load(file = betatg.list)
        betatrue.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                                '_ns', nsource, '_nt', ntarget, '_betatrue.rda')
        load(file = betatrue.list)
        ytgest.list <- paste0("../output/true/", famname, '_alpha', alpha*10, '_rho', rho*10, 
                              '_ns', nsource, '_nt', ntarget, '_predict.rda')
        load(file = ytgest.list)
        if (measure == 'mean'){
          wbbvalue <- trimmed_mean((wbb.mean - ytrue.mean)^2)
          bbvalue <- trimmed_mean((bb.mean - ytrue.mean)^2)
        } else if (measure == 'beta1'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[2]) - sapply(betatrue.list, function(x) x[1]))^2)
        } else if (measure == 'beta2'){
          wbbvalue <- trimmed_mean((sapply(wbb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
          bbvalue <- trimmed_mean((sapply(bb.list, function(x) x[3]) - sapply(betatrue.list, function(x) x[2]))^2)
        } else if (measure == 'pred'){
          if (famname == 'Gaussian'){
            wbbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ywbb.list), function(x) mean(x^2)))
            bbvalue <- trimmed_mean(sapply(Map(`-`,  ytg.list, ybb.list), function(x) mean(x^2)))
          } else if (famname == 'Binomial'){
            wbbvalue <- numeric(length(ytg.list))
            bbvalue <- numeric(length(ytg.list))
            for (i in seq_along(ytg.list)) {
              true_labels <- ytg.list[[i]]      
              predicted_bb <- ybb.list[[i]] 
              predicted_wbb <- ywbb.list[[i]]
              wbbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_wbb, direction = ">")))
              bbvalue[i] <- suppressMessages(auc(roc(true_labels, predicted_bb, direction = ">")))
            }
            wbbvalue <- 1/trimmed_mean(wbbvalue)
            bbvalue <- 1/trimmed_mean(bbvalue)
          }
        }
        # input data
        df4 <- rbind(df4,
                    data.frame(ntarget = ntarget,
                               rho  = rho,
                               ARE   = wbbvalue/bbvalue))
      }
    }
    df4 <- df4 %>%
      mutate(
        rowfacet   = "nsource = 300, ntarget changes", # bottom row
        rowfacet   = "n = 300, N changes", # bottom row
        colfacet   = "rho",              # right column
        xvar       = ntarget,
        groupvar   = rho
      )
    
    df_alpha <- bind_rows(df1, df2)
    df_alpha <- df_alpha[,c('rowfacet', 'colfacet', 'xvar', 'alpha', 'ARE')]
    df_alpha$alpha <- as.factor(df_alpha$alpha)
    df_rho <- bind_rows(df3, df4)
    df_rho <- df_rho[,c('rowfacet', 'colfacet', 'xvar', 'rho', 'ARE')]
    df_rho$rho <- as.factor(df_rho$rho)
    df_alpha <- df_alpha %>% mutate(colfacet = "alpha")
    df_rho   <- df_rho   %>% mutate(colfacet = "rho")
    
    # plot for alpha
    df_label1 <- df_alpha[df_alpha$rowfacet == 'N = 300, n changes', ] %>%
      group_by(alpha) %>%
      slice(which.max(xvar))
    df_label2 <- df_alpha[df_alpha$rowfacet == 'n = 300, N changes', ] %>%
      group_by(alpha) %>%
      slice(which.max(xvar))
    df_label <- rbind(df_label1, df_label2)
    p_alpha <- ggplot(df_alpha, aes(x = xvar, y = ARE,
                                color = alpha,
                                shape = alpha,
                                linetype = alpha)) +
      geom_line(size = 1) +
      geom_point(size = 2, stroke = 1) +
      scale_x_continuous(breaks = c(300, 600, 900, 1200, 1500)) +
      scale_y_continuous(
        breaks = function(x) unique(c(scales::pretty_breaks(n = 5)(x), 1)) 
      ) +
      geom_label_repel(
        data = df_label,
        aes(label = paste("alpha ==", alpha)),  
        nudge_x = 30,          
        show.legend = FALSE,   
        size = 4,              
        min.segment.length = 0,
        parse = TRUE
      ) + 
      facet_wrap(~ rowfacet, ncol = 2, scales = "free_x") +
      scale_color_manual(
        name = expression(alpha),
        values = scales::hue_pal()(length(unique(df_alpha$alpha))),
        labels = as.expression(lapply(as.vector(unique(df_alpha$alpha)), function(x) bquote(alpha == .(x))))
      ) +
      scale_shape_manual(
        name = expression(alpha),
        values = sort(unique(df_alpha$alpha)),  
        labels = as.expression(lapply(as.vector(unique(df_alpha$alpha)), function(x) bquote(alpha == .(x))))
      ) +
      scale_linetype_manual(
        name = expression(alpha),
        values = sort(unique(df_alpha$alpha)),  
        labels = as.expression(lapply(as.vector(unique(df_alpha$alpha)), function(x) bquote(alpha == .(x))))
      ) +
      labs(x = "Sample size", y = "ARE") +
      theme_bw(base_size = 14) +
      theme(
        legend.position = "right",
        strip.text      = element_text(size = 14, face = "bold")
      )
    
    # plot for rho
    df_label1 <- df_rho[df_rho$rowfacet == 'N = 300, n changes', ] %>%
      group_by(rho) %>%
      slice(which.max(xvar))
    df_label2 <- df_rho[df_rho$rowfacet == 'n = 300, N changes', ] %>%
      group_by(rho) %>%
      slice(which.max(xvar))
    df_label <- rbind(df_label1, df_label2)
    p_rho <- ggplot(df_rho, aes(x = xvar, y = ARE,
                                color = rho,
                                shape = rho,
                                linetype = rho)) +
      geom_line(size = 1) +
      geom_point(size = 2, stroke = 1) +
      scale_x_continuous(breaks = c(300, 600, 900, 1200, 1500)) +
      scale_y_continuous(
        breaks = function(x) unique(c(scales::pretty_breaks(n = 5)(x), 1)) 
      ) +
      geom_label_repel(
        data = df_label,
        aes(label = paste("zeta ==", rho)),  
        nudge_x = 30,          
        show.legend = FALSE,   
        size = 4,              
        min.segment.length = 0,
        parse = TRUE
      ) + 
      facet_wrap(~ rowfacet, ncol = 2, scales = "free_x") +
      scale_color_manual(
        name = expression(zeta),
        values = scales::hue_pal()(length(unique(df_rho$rho))),
        labels = as.expression(lapply(as.vector(unique(df_rho$rho)), function(x) bquote(zeta == .(x))))
      ) +
      scale_shape_manual(
        name = expression(zeta),
        values = sort(unique(df_rho$rho)),  
        labels = as.expression(lapply(as.vector(unique(df_rho$rho)), function(x) bquote(zeta == .(x))))
      ) +
      scale_linetype_manual(
        name = expression(zeta),
        values = sort(unique(df_rho$rho)),  
        labels = as.expression(lapply(as.vector(unique(df_rho$rho)), function(x) bquote(zeta == .(x))))
      ) +
      labs(x = "Sample size", y = "ARE") +
      theme_bw(base_size = 14) +
      theme(
        legend.position = "right",
        strip.text      = element_text(size = 14, face = "bold")
      )
    
    p_combined <- p_alpha / p_rho
    p_combined
    ggsave(paste0("../figures/", famname, '_', measure, '.pdf'), 
           plot = p_combined, 
           width = 14, height = 15, dpi = 300)
  }
}


