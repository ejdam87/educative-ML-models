
samples <- c(
  c(1, -1, -3),
  c(1, -1, -2),
  c(1, -1, -1),
  c(1, 1, 1),
  c(1, 1, 2),
  c(1, 1, 3)
  )

X <- matrix( samples, 6, 3,  byrow=TRUE )
X.t <- t( X )

n <- 6
k <- 3

Y <- c( 5, 7, 8, 12, 13, 15 )
XX <- X.t %*% X

XX.inv <- solve(XX)

B <- XX.inv %*% X.t %*% Y
Y.predicted <- X %*% B

S.e <- t((Y - Y.predicted)) %*% (Y - Y.predicted)
s.2 <- S.e / ( n - k )

## index of determination

ID <- function( Y, Y.predicted )
{
  Y.mean <- mean(Y)
  n <- length(Y)

  S.2y.pred <- 1/n * (Y.predicted - Y.mean)^2
  S.2y <- 1/n * (Y - Y.mean)^2
  
  return( S.2y.pred / s.2y )
  
}
