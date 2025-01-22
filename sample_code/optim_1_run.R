n <- 1000
x <- c(0.3, 0.1, 0.03, 0, 0)
p <- length(x)
A <- matrix(rnorm(n * p), n, p)
y <- A %*% x + rnorm(n)
lambda <- 10   #-- lambda is the tuning parameter, not step size
x0 <- rnorm(p)


# closed form solution to ridge
solve(t(A) %*% A + 2*lambda*diag(p)) %*% t(A) %*% y

# ## different methods for gradient descent
# out = gradient.descent.lsq_R(y, A, x0, lambda, 0.001)
# out = gradient.descent.lsq.v2_R(y, A, x0, lambda, 0.00001)
# 
# 
# ## gradient descent with BB
# out = gradient.descent.BB.lsq_R(y, A, x0, lambda)
# 
# 
# ## stochastic gradient descent
# 
# out.batch = stochastic.gradient.descent.lsq(y, A, x0,lambda, n)
# #plot(out.batch$loss)
# #out.batch$x
# 
# out.minibatch = stochastic.gradient.descent.lsq(y, A, x0,lambda, 100)
# #plot(out.minibatch$loss)
# #out.minibatch$x 
# 
# out.persample = stochastic.gradient.descent.lsq(y, A, x0,lambda, 1) 
# #plot(out.persample$loss)
# #out.persample$x  

# install.packages("rbenchmark")
library(rbenchmark)

benchmark(
  R_version = gradient.descent.lsq.v2_R(y, A, x0, lambda, gamma=2, tol=0.0001),
  CPP_version = gradient_descent_lsq_cpp(y, A, x0, lambda, gamma=2, tol=0.0001),
  replications = 100
)

r_result <- gradient.descent.lsq.v2_R(y, A, x0, lambda, gamma=2) # ?
cpp_result <- gradient_descent_lsq_cpp(y, A, x0, lambda, gamma=2)
