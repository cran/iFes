cal.f.ratio <- function(feature, x, y){
  u.1 <- mean(x[y == 1, feature])
  v.1 <- var(x[y == 1, feature])
  u.0 <- mean(x[y == 0, feature])
  v.0 <- var(x[y == 0, feature])
  return(((u.1 - u.0)^2) / (v.1 + v.0))
}


f.ratio <- function(x, y, k){
  feature.f.ratio <- do.call("c", lapply(1:ncol(x), cal.f.ratio, x, y))
  gene.id <- 1:ncol(x)
  f.ratio.gene <- as.data.frame(cbind(gene.id, feature.f.ratio))
  f.ratio.gene <- f.ratio.gene[order(f.ratio.gene[,2], decreasing=T),]
  rank.id <- 1:ncol(x)
  f.ratio.gene <- cbind(f.ratio.gene, rank.id)
  names(f.ratio.gene) <- c("gene.id", "f.ratio", "rank.id")
  f.ratio.rank <- subset(f.ratio.gene, rank.id <= k)
  return(x[,f.ratio.rank$gene.id])
}
