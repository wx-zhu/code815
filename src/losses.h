#include <RcppArmadillo.h>
using namespace std;

double loss_ridge(const arma::vec& y, const arma::mat& A, const arma::vec& x, double lambda);
