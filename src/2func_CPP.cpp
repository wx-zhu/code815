// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
#include "losses.h"

// [[Rcpp::export]]
double loss_ridge_cpp(const arma::vec& y, const arma::mat& A, 
                      const arma::vec& x, double lambda) {
  
  int n = y.n_elem;  // length of y vector
  int p = x.n_elem;  // length of x vector
  
  // sanity check: stopifnot(dim(A) == c(n,p))     
  if (A.n_rows != n || A.n_cols != p) {
    Rcpp::stop("Dimension incorrect: A should be n x p");
  }
  
  arma::vec res = y - A * x;  // residual
  
  // objective function: (1/2) * sum(res^2) + lambda * sum(x^2)
  return (0.5 * dot(res, res) + lambda * dot(x, x));
}


// [[Rcpp::export]]
Rcpp::List gradient_descent_lsq_cpp(const arma::vec& y, const arma::mat& A, 
                                    const arma::vec& x0, double lambda, double gamma, 
                                    double tol = 0.0001, int max_iter = 10000, 
                                    bool printing = false) {
  // dimensions
  int n = y.n_elem;
  int p = A.n_cols;
  
  // sanity checks: stopifnot(n == nrow(A))             # sanity check
  //                stopifnot(n >= p) 
  if (A.n_rows != n) {
    Rcpp::stop("Dimension mismatch: nrow(A) should = length(y)");
  }
  if (n < p) {
    Rcpp::stop("n should be >= p");
  }
  
  //  A'A and A'y
  arma::mat AA = A.t() * A;
  arma::vec Ay = A.t() * y;
  
  // initialize variables
  arma::vec x = x0;
  arma::vec grad = AA * x0 - Ay;
  grad = grad + 2 * lambda * x0;
  
  //  initial loss using loss_ridge
  double loss = loss_ridge(y, A, x0, lambda);
  double prev_loss = loss;
  double diff = arma::datum::inf;
  arma::vec diff_rec(max_iter);
  arma::vec loss_rec(max_iter);
  int iter = 0;
  
  while (iter < max_iter && diff > tol) {
    x = x - gamma * grad;
    
    // new gradient
    grad = AA * x - Ay;
    grad = grad + 2 * lambda * x;
    
    // new loss using loss_ridge 
    loss = loss_ridge(y, A, x, lambda);
    
    // record
    diff_rec(iter) = (prev_loss - loss) / std::abs(prev_loss);
    diff = std::abs(diff_rec(iter));
    loss_rec(iter) = loss;
    
    prev_loss = loss;
    iter++;
  }
  
  // Trim vectors to actual number of iterations
  diff_rec = diff_rec.head(iter);
  loss_rec = loss_rec.head(iter);
  
  if (printing) {
    Rcpp::Rcout << "converge after " << iter << " steps" << std::endl;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("diff") = diff_rec,
    Rcpp::Named("loss") = loss_rec
  );
}

