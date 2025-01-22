

### functions
loss.ridge <- function(y, A, x, lambda) {  # function definition // HL
  #' Ridge regression Loss Function for Linear Models
  #'
  #' @param y      A (n x 1) vector of response variables
  #' @param A      A (n x p) matrix of predictor variables
  #' @param x      A (p x 1) vector of effect size for each predictor variable
  #' @param lambda Regularization parameter in ridge regression
  #' @return Ridge Regression errors with penalty
  #' @export
  n <- length(y)
  p <- length(x)
  stopifnot(dim(A) == c(n,p))                 # sanity check
  res <- y - A %*% x                          # calculate residuals // HL
  return ((1/2)*sum(res*res) + sum(x*x)*lambda) # calculate objective function // HL
}


####
gradient.descent.lsq<-function(y, A, x0,lambda,gamma, method, tol=0.0001, max.iter=10000){
  #' gamma is step size
  #' method is 'lasso' or 'ridge'
  n <- length(y)
  p <- ncol(A)
  stopifnot(n == nrow(A))             # sanity check
  stopifnot(n >= p) 
  

  AA=t(A)%*%A
  Ay=t(A)%*%y
  grad = AA%*%x0-Ay    # gradient for the least squares
  if(method=='lasso'){
    loss=loss.lasso(y, A, x0, lambda)
    grad=grad+lambda*sign(x0)
  }
  if(method=='ridge'){
    loss=loss.ridge(y, A, x0, lambda)
    grad=grad+2*lambda*x0
  }
  
  prevloss=loss
  x=x0
  iter=1
  diff=Inf
  diff.rec=c()
  loss.rec=c()
  while(iter<max.iter & diff>tol) {
    x=x-gamma*grad # gradient descent
    grad = AA%*%x-Ay  # gradient for the least squares
    
    if(method=='lasso'){
      loss=loss.lasso(y, A, x, lambda)
      grad=grad+lambda*sign(x)
    }
    if(method=='ridge'){
      loss=loss.ridge(y, A, x, lambda)
      grad=grad+2*lambda*x
    }
    diff.rec[iter]=(prevloss-loss)/abs(prevloss) # should be positive for properly chosen step size
    diff=abs(diff.rec[iter])
    
    loss.rec[iter]=loss
    prevloss=loss
    iter=iter+1
  }
  print(paste0("converge after ",iter, " steps"))
  return(list(x=x,diff=diff.rec,loss=loss.rec))
}
  

gradient.descent.lsq.v2<-function(y, A, x0,lambda,gamma, method, tol=0.0001, max.iter=10000){
  #' gamma is initial step size
  #' method is 'lasso' or 'ridge'
  n <- length(y)
  p <- ncol(A)
  stopifnot(n == nrow(A))             # sanity check
  stopifnot(n >= p) 
  
  
  AA=t(A)%*%A
  Ay=t(A)%*%y
  grad = AA%*%x0-Ay    # gradient for the least squares
  if(method=='lasso'){
    loss=loss.lasso(y, A, x0, lambda)
    grad=grad+lambda*sign(x0)
  }
  if(method=='ridge'){
    loss=loss.ridge(y, A, x0, lambda)
    grad=grad+2*lambda*x0
  }
  
  prevloss=loss
  x=x0
  iter=1
  diff=Inf
  diff.rec=c()
  loss.rec=c()
  while(iter<max.iter) {
    x=x-gamma*grad # gradient descent
    grad = AA%*%x-Ay  # gradient for the least squares
    
    if(method=='lasso'){
      loss=loss.lasso(y, A, x, lambda)
      grad=grad+lambda*sign(x)
    }
    if(method=='ridge'){
      loss=loss.ridge(y, A, x, lambda)
      grad=grad+2*lambda*x
    }
    
    diff.rec[iter]=(prevloss-loss)/abs(prevloss) 
    
    ## adaptive step size
    if(prevloss>loss){
      if(diff.rec[iter]<tol){break}
      gamma=gamma*1.1
    }else{
      gamma=gamma/2
    }
    
    loss.rec[iter]=loss
    prevloss=loss
    iter=iter+1
  }
  print(paste0("converge after ",iter, " steps"))
  return(list(x=x,diff=diff.rec,loss=loss.rec))
}

######################################################

gradient.descent.BB.lsq<-function(y, A, x0,lambda,method, tol=0.0001, max.iter=10000){
  #' use BB method to determine adaptive step sizes
  #' method is 'lasso' or 'ridge'
  n <- length(y)
  p <- ncol(A)
  stopifnot(n == nrow(A))             # sanity check
  stopifnot(n >= p) 
  
  
  AA=t(A)%*%A
  Ay=t(A)%*%y
  grad = AA%*%x0-Ay    # gradient for the least squares
  if(method=='lasso'){
    loss=loss.lasso(y, A, x0, lambda)
    grad=grad+lambda*sign(x0)
  }
  if(method=='ridge'){
    loss=loss.ridge(y, A, x0, lambda)
    grad=grad+2*lambda*x0
  }
  x=x0-grad # gradient descent
  prevx=x0
  prevgrad=grad
  prevloss=loss
  iter=1
  diff=Inf
  diff.rec=c()
  while(iter<max.iter & diff>tol) {
    grad = AA%*%x-Ay  # gradient for the least squares
    if(method=='lasso'){
      grad=grad+lambda*sign(x)
    }
    if(method=='ridge'){
      grad=grad+2*lambda*x
    }
    gamma=crossprod(x-prevx,x-prevx)/crossprod(x-prevx,grad-prevgrad)
    # print(gamma)
    # gradient descent
    prevx=x    
    prevgrad=grad
    prevloss=loss
    x=prevx-c(gamma)*prevgrad
    if(method=='lasso'){
      loss=loss.lasso(y, A, x, lambda)
    }
    if(method=='ridge'){
      loss=loss.ridge(y, A, x, lambda)
    }
    
    diff.rec[iter]=(prevloss-loss)/abs(prevloss) # should be positive for properly chosen step size
    diff=abs(diff.rec[iter])
    
    diff
    
    prevloss=loss
    iter=iter+1
  }
  print(paste0("converge after ",iter, " steps"))
  return(list(x=x,diff=diff.rec))
}

####################################################

stochastic.gradient.descent.lsq<-function(y, A, x0,lambda, batch, initial.step.size=1, method, tol=1E-6, max.iter=10000){
  #' method is 'lasso' or 'ridge'
  #' batch is batch size, ranging from 1 to n (original gradient descent)
  n <- length(y)
  p <- ncol(A)
  stopifnot(n == nrow(A))             # sanity check
  stopifnot(n >= p) 
  
  keep=sample(n,batch)
  Asub=A[keep,,drop=FALSE]
  ysub=y[keep,drop=FALSE]
  AA=t(Asub)%*%Asub
  Ay=t(Asub)%*%ysub
  grad = (AA%*%x0-Ay)/batch    # gradient for the least squares
  if(method=='lasso'){
    loss=loss.lasso(y, A, x0, lambda)
    grad=grad+lambda*sign(x0)/n
  }
  if(method=='ridge'){
    loss=loss.ridge(y, A, x0, lambda)
    grad=grad+2*lambda*x0/n
  }
  
  prevloss=loss
  x=x0
  iter=1
  diff=Inf
  diff.rec=c()
  loss.rec=c()
  while(iter<max.iter & diff>tol) {
    x=x-(initial.step.size/iter)*grad # gradient descent with decreasing step size
    
    keep=sample(n,batch)
    Asub=A[keep,,drop=FALSE]
    ysub=y[keep,drop=FALSE]
    AA=t(Asub)%*%Asub
    Ay=t(Asub)%*%ysub
    grad = (AA%*%x-Ay)/batch   # gradient for the least squares
    
    if(method=='lasso'){
      loss=loss.lasso(y, A, x, lambda)
      grad=grad+lambda*sign(x)/n
    }
    if(method=='ridge'){
      loss=loss.ridge(y, A, x, lambda)
      grad=grad+2*lambda*x/n
    }
    diff.rec[iter]=(prevloss-loss)/abs(prevloss) 
    diff=abs(diff.rec[iter])
    
    loss.rec[iter]=loss
    prevloss=loss
    iter=iter+1
    # print(iter)
  }
  print(paste0("converge after ",iter, " steps"))
  return(list(x=x,diff=diff.rec,loss=loss.rec))
}



  




