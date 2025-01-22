#include "losses.h"

//[[Rcpp::export]]
double loss_ridge(const arma::vec& y, const arma::mat& A, const arma::vec& x, double lambda) {
  arma::vec res = y - A * x;
  
  // Ridge regression loss: (1/2) * sum(res^2) + lambda * sum(x^2)
  double loss = 0.5 * arma::accu(res % res) + lambda * arma::accu(x % x);
  return loss;
}