LRCUDA <- function(x, y, n.comb = 2, ll.threshhold , fold = 10, device.id = 0, cl = NULL){

    

    if(!is.matrix(x)){
        stop("x should be matrix type !")
    }
  
    if(nrow(x) != length(y)){
        stop("x'rows is different from y'length !")
    }

    task.num <- choose(ncol(x), n.comb)
    x <- cbind(rep(1,length(y)), x) 
    device.num <- length(device.id)
 
    if(is.null(cl)){
        
        cl <- makeCluster(length(device.id), type = "SOCK")
        
    }else{
        if(length(cl) != length(device.id)){
            stop("device count should be equal to cluster size! Please check you configures.")
        }
    }
    

    clusterEvalQ(cl,library(iFes))
    para <- vector("list", device.num)
    task.piece <- floor(task.num / device.num)	

    for(i in 1:device.num){
        para[[i]] <- list(x = x, y = y, n.comb = n.comb, ll.threshhold = ll.threshhold, fold = fold, device.id = device.id[i], start = (i-1)*task.piece + 1, stop = i*task.piece)
    }
    if(para[[device.num]]$stop < task.num){
            para[[device.num]]$stop = task.num
    }
               
    result <- clusterApply(cl, para, LRMultipleGPU)
    return(combineResult(result))
    stopCluster(cl)
}


LRSingleGPU <- function(x, y, n.comb = 2, error.threshhold = 0 , fold = 10, device.id, start, stop){

    result <- .Call("LRCUDA", t(x), y, as.integer(n.comb), as.integer(error.threshhold), as.integer(fold), as.integer(device.id), as.integer(start), as.integer(stop))
    return(matrix(result, ncol = n.comb + 1))
}

LRMultipleGPU <- function(para){
     x <- para$x
     y <- para$y
     n.comb <- para$n.comb
     ll.threshhold <- para$ll.threshhold
     fold <- para$fold
     device.id <- para$device.id
     start <- para$start
     stop <- para$stop

     result <- .Call("LRCUDA", t(x), y, as.integer(n.comb), as.numeric(ll.threshhold), as.integer(fold), as.integer(device.id), as.integer(start), as.integer(stop))
     return(t(matrix(result, nrow = n.comb + 1)))
     
}





