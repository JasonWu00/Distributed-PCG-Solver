#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <vector>

struct CG_Solver {
  CG_Solver(const int& n, const int& N);
  void solve(const std::vector<double>& b, std::vector<double>& x, double tol);
};

struct CG_Solver_csr {
  CG_Solver_csr(const int& n, const int& N);
  void solve(const std::vector<double>& b, std::vector<double>& x, double tol);
};

#endif