\name{iFes}
\alias{iFes}
\title{
An incremental feature selection algorithm for gene expression profile. We implement it using logistic regression which is accelerated by NVIDIA GPU technology.
}
\description{
iFes expands the screening targets in each step of the forward stepwise algorithm, and proposes a novel feature selection algorithm iFes with both high accuracies and low feature numbers. Logistic regression classification algorithm is utilized to measure how good the currently chosen feature subset is. 
}
\usage{
iFes(x, y, ll.diff = 0.00001, fold = 10, device.id = 0, cl = NULL)
}
%- maybe also 'usage' for other objects documented here.


\arguments{
\item{x}{Gene expression profile matrix. The columns represent different prob sets. The rows represent different samples. The values in the matrix represent gene expression levels.}
\item{y}{The response vector, positive(1) or negative(0).}
\item{ll.diff}{log loss difference between two iterations when searching forward. When the difference is less than ll.difference, the search will stop.}
\item{fold}{The number of fold of cross validation used in the program.}
\item{device.id}{The GPU device ID. }
\item{cl}{The cluster of computers which you created. If it is NULL, the program will create clusters based on the number of devices automatically.}
}

\value{
\item{result}{The result of feature selection by iFes. Each row in result represent the one combination of selected features and its log loss.}
}

\author{
Qinghan Meng
}


%% ~Make other sections like Warning with \section{Warning }{....} ~


\examples{
data(colon)
x <- colon$x
y <- colon$y
x <- f.ratio(x, y, 1000)
gpu.ids <- getGPUIds()
#result <- iFes(x = x, y = y, device.id = gpu.ids)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
