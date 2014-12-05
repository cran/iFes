LOOCV <- function(features, X, Y){
  X <- X[,features]
  data <- as.data.frame(cbind(X,Y))
  names(data)[ncol(data)] <- "label"
  error <- 0
  prob <- c()
  for(i in 1:nrow(data)){
    model.lr <- glm(label ~ ., data = data[-i,], family = binomial(logit))
    pr.lr <- predict(model.lr,data[i,], type = "response")
    prob <- c(prob, pr.lr)
    pr.lr <- ifelse(pr.lr >= 0.5, 1, 0)
    error <- error + length(which(pr.lr != data[i, ncol(data)]))
  }
  pred <- prediction(prob, Y)
  perf <- performance(pred, "auc")
  auc <- perf@y.values[[1]]
  acc <- (length(Y) - error) / length(Y)
  return(c(acc, auc))   
}


CV <- function(y, fold = 10){
  zero.index <- which(y == 0)
  zero.index <- sample(zero.index, length(zero.index))
  one.index <- which(y == 1)
  one.index <- sample(one.index, length(one.index))
  
  cv.zero.test <- split(zero.index, 1:fold)
  cv.one.test <- split(one.index, 1:fold)
  cv.test <- list()
  for(i in 1:fold){
    cv.test[[i]] <- c(cv.zero.test[[i]], cv.one.test[[i]])
  }
  
  cv.train <- list()
  for(i in 1:fold){
    cv.train[[i]] <- setdiff(1:length(y), cv.test[[i]])
  }
  cv <- list()
  cv$test <- cv.test
  cv$train <- cv.train
  return(cv)
}



KFold <- function(features, X, Y, fold, n){
  X <- as.matrix(X[,features])
  data <- as.data.frame(cbind(X,Y))
  names(data)[ncol(data)] <- "label"
  #error <- numeric(n)
  acc <- numeric(n)
  auc <- numeric(n)
  for(t in 1:n){
    set.seed(t)
    cv <- CV(Y, fold)
    predY <- numeric(length(Y))
    for(i in 1:fold){
      model <- glm(label ~ ., data = data[cv$train[[i]],], family = binomial(logit))
      pred <- c()
      if(length(cv$test[[i]]) != 0){
        pred <- predict(model, data[cv$test[[i]],], type = "response")
      }
      predY[cv$test[[i]]] <- pred
    }
    
    pred.roc <- prediction(predY, Y)  
    perf <- performance(pred.roc, "auc")
    auc[t] <- perf@y.values[[1]]
    predY <- ifelse(predY >= 0.5, 1, 0)
    acc[t] <- length(which(predY == Y)) / length(Y)

  }
  
  
  return(c(mean(acc),sd(acc),mean(auc), sd(auc)))
}


rm.bias <- function(result, x, y, fold, times, parallel = 1){
  
  cl <- makeCluster(parallel , type = "SOCK")
  registerDoParallel(cl)
  
  for(i in length(result):length(result)){
    features.set <- as.matrix(result[[i]][,1:(ncol(result[[i]])-1)])
    j <- NULL
    LOOCV.result <- foreach(j = 1:nrow(features.set), .packages = c("ROCR")) %dopar%{
      LOOCV(features.set[j,], x, y)
    }
    
    LOOCV.result <- do.call("rbind", LOOCV.result)
    colnames(LOOCV.result) <- c("loocv-acc", "loocv-auc")
    result[[i]] <- cbind(result[[i]], LOOCV.result)
    
    KFold.result <- foreach(j = 1:nrow(features.set), .packages = c("ROCR")) %dopar%{
      KFold(features.set[j,], x, y ,fold, times)
    }
    KFold.result <- do.call("rbind", KFold.result)
    colnames(KFold.result) <- c("mean-10-fold-acc", "sd-10-fold-acc", "mean-10-fold-auc", "sd-10-fold-auc")
    result[[i]] <- cbind(result[[i]], KFold.result)
  }
  stopCluster(cl)
  return(result)
  
}




