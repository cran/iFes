%% the rd file refers to Package plsgenomics.

\name{colon example dataset}
\alias{colon}

\title{Gene expression data from Alon et al. (1999)}
\usage{
data(colon)
}

\description{
Expression data of 2000 genes for 62 samples, which was generated from the microarray experiments of colon tissue samples of Alon et al. (1999).
}

\details{ This data set contains 62 samples with 2000 genes: 40 tumor tissues, coded 1 and 22 normal tissues, coded 0.  
}

\value{
A list with the following elements:
  \item{X}{a (2000 x 62) matrix giving the expression levels of 2000 genes for the 62 colon tissue samples. Each row corresponds to a gene, and each column to a patient/sample.}
  \item{y}{a numeric vector of length 62 giving the type of tissue sample (tumor or normal).} 

}

\source{The data are described in Alon et al.(1999) and can be freely
downloaded from 
\url{http://microarray.princeton.edu/oncology/affydata/index.html}.}

\references{
Alon, U. and Barkai, N. and Notterman, D.A. and Gish, K. and Ybarra, S. and Mack, D. and Levine, A.J. (1999).
Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays. Proc. Natl. Acad. Sci. USA,\bold{96}(12), 6745--6750.
}


\examples{



# load data set
data(colon)

# how many samples and how many genes ?
dim(colon$X)

# how many samples of class 0 and 1 respectively ?
sum(colon$y==0)
sum(colon$y==1)
}

\keyword{datasets}

