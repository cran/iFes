getGPUCount <- function(cl = NULL){
    if(is.null(cl)){
        count <- vector("list", 1)
        count[[1]] <- .Call("getGPUCount")
        return(count)
    }else{
        clusterEvalQ(cl, library(iFes))
    	count <- clusterEvalQ(cl, .Call("getGPUCount"))
        return(count)
    }
}

generateIds <- function(count){
    return(0:(count-1))
}

getGPUIds <- function(cl = NULL){
    count <- getGPUCount(cl)
    Ids <- lapply(count, generateIds)
    return(Ids[[1]])
}

