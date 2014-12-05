

combineResult <- function(result){
    result <- do.call(rbind, result)
    result <- as.data.frame(result)
    features.num <- ncol(result) - 1
    header.names <- paste("feature", 1:features.num, sep=".")
    header.names <- c(header.names, "logloss")
    names(result) <- header.names
    return(result)
    
}
