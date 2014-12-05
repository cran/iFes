\name{getGPUIds}
\alias{getGPUIds}
\title{
Get GPU device IDs from localhost or a cluster.
}
\description{
Query GPU device number and ID from the localhost or a cluster. 
}
\usage{
getGPUIds(cl = NULL)
}
%- maybe also 'usage' for other objects documented here.


\arguments{
\item{cl}{The cluster of computers which you created. If it is NULL, the program will execute in localhost.}
}

\value{
\item{Ids}{A vector contains devices' IDs.}
}

\author{
Qinghan Meng
}


%% ~Make other sections like Warning with \section{Warning }{....} ~


\examples{
gpu.ids <- getGPUIds()
print(gpu.ids)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
