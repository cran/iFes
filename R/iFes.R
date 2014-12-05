RmDupVal <- function(features){
    features <- t(apply(features, 1, sort))
    features <- subset(features, duplicated(features) == F)
    return(features)
}

LRWithFixedVal <- function(para){
    x <- para$x
    y <- para$y
    n.comb <- para$n.comb
    ll.threshhold <- para$ll.threshhold
    fold <- para$fold
    device.id <- para$device.id
    fixed.features <- para$fixed.features
    
    result <- .Call("LRCUDAWithFixedVal", t(x), y, as.integer(n.comb), as.numeric(ll.threshhold), as.integer(fold), as.integer(device.id), t(fixed.features), nrow(fixed.features), ncol(fixed.features))

    result <- t(matrix(result, nrow = n.comb + ncol(fixed.features)+1 ))
    result <- as.data.frame(result)
    features.name <- paste0("feature", 1:(ncol(result)-1))
    names(result) <- c(features.name, "logloss")
    return(result)

}

LRWithFixedValMG <- function(x, y, n.comb = 1, ll.threshhold = 0 , fold = 10, device.id = 0, cl = NULL, fixed.features){
    if(!is.matrix(x)){
        stop("x should be matrix type !")
    }
  
    if(nrow(x) != length(y)){
        stop("x'rows is different from y'length !")
    }
    x <- cbind(rep(1, nrow(x)), x)
    if(is.null(cl)){
        cl <- makeCluster(length(device.id), type = "SOCK")
    }else{
        if(length(cl) != length(device.id)){
            stop("device count should be equal to cluster size! Please check you configures.")
        }
    }
    task.num <- nrow(fixed.features)
    print(paste("task.num", task.num))
  
    if (task.num < length(device.id)){
        device.id <- device.id[1:task.num]
        cl <- cl[1:task.num]
    }
    device.num <- length(device.id)

    clusterEvalQ(cl,library(iFes))
    para <- vector("list", device.num)
    task.piece <- floor(task.num / device.num)
   
    for(i in 1:device.num){
        if(i != device.num){
            para[[i]] <- list(x = x, y = y, n.comb = n.comb, ll.threshhold = ll.threshhold, fold = fold, device.id = device.id[i], fixed.features = as.matrix(fixed.features[((i-1)*task.piece + 1):(i*task.piece),]))
        }else{
            para[[i]] <- list(x = x, y = y, n.comb = n.comb, ll.threshhold = ll.threshhold, fold = fold, device.id = device.id[i], fixed.features = as.matrix(fixed.features[((i-1)*task.piece + 1):(nrow(fixed.features)),]))
        }
        
    }
    
    result <- clusterApply(cl, para, LRWithFixedVal)
    result <- do.call("rbind", result)
    stopCluster(cl)
    return(result)
}


logloss <- function(pred, act){
    ll <- -(act*log(pred) + (1-act)*log(1-pred))
    return(ll)
}

InitLogloss <- function(label){
  label.freq <- sort(table(label), decreasing = T)
  m.label <- as.numeric(names(label.freq)[1])
  pred <- rep(0.999, length(label))
  if(m.label == 0.0){
    pred <- rep(0.001, length(label))
  }
  ll <- logloss(pred, label)
  return(sum(ll))
}


iFes <- function(x, y, ll.diff = 0.00001, fold = 10, device.id = 0, cl = NULL){
    result.l <- list()
    rand.index <- sample(1:length(y), length(y))
    init.ll <- InitLogloss(y)
    pre.min.ll <- init.ll
    print(paste("init log loss is ", init.ll))
    result.one <- LRCUDA(x = x, y = y, n.comb = 1, ll.threshhold = init.ll, fold = fold, device.id = device.id, cl = cl)
    if(nrow(result.one) == 0){
         print("use one feature can't statisfy the error threshhold")
         return(result.l)
    }
    
    ll.min.one <- min(result.one$logloss)
    print(paste("logloss.min.one", ll.min.one))
    if(pre.min.ll - ll.min.one <= ll.diff){
        return(result.l)
    }else{
        #save(result.one, file = "result.one.feature.RData")
    }
    result.l[[1]] <- result.one
    pre.min.ll <- ll.min.one
   
 
    fixed.features <- result.one[,1]
    mode(fixed.features) <- "integer"
    rand.index <- sample(1:length(y), length(y))    
    result.two <- LRWithFixedValMG(x, y, n.comb = 1, ll.threshhold = ll.min.one, fold = fold, device.id = device.id , cl = cl, fixed.features = as.matrix(fixed.features))
    
    if(nrow(result.two) == 0){
        print("use two features can't satisfy the error theshhold")
        return(result.l)
    }
       
    ll.min.two <- min(result.two$logloss)
    if(pre.min.ll - ll.min.two <= ll.diff){
        return(result.l)
    }
    result.l[[2]] <- result.two
    ll.min.pre <- ll.min.two
    
    #save(result.two, file = "result.two.features.RData")
    print(paste("logloss.min.two", ll.min.two))
   
    

    fixed.features <- RmDupVal(result.two[,1:2])
    mode(fixed.features) <- "integer"
    ll.para <- ll.min.two
    
    for(i in 3:10){
        print(paste("i = ", i))
        print(paste("fixed features num", nrow(fixed.features)))
        rand.index <- sample(1:length(y), length(y))
        result <- LRWithFixedValMG(x, y, n.comb = 1, ll.threshhold = ll.para, fold = fold, device.id = device.id ,cl = cl, fixed.features = as.matrix(fixed.features))

        print(paste("row num is ", nrow(result)))
        
        if(nrow(result) == 0){
            print(paste("use", i+1 ,  "feature can't statisfy the error threshhold"))
            return(result.l)
        }
        
        ll.min <- min(result$logloss)
        print(paste("log loss min i = ", i, " ", ll.min))
        if(pre.min.ll - ll.min <= ll.diff){
            return(result.l)
        }
        else{
            #save(result, file = paste0("features", i, ".RData"))
        }
        result.l[[i]] <- result
        fixed.features <- RmDupVal(result[,1:i])
        mode(fixed.features) <- "integer"
        pre.min.ll <- ll.min
        ll.para <- ll.min
        
    }
    return(result.l) 
}


