#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>
#include <map>
#include <vector>

#include <Eigen/Sparse>

/*
2: Layout Formulation: Figure out how to distribute the sparse matrix format across MPI processes.

3: Parallelize matrix-vector multiplication and norm across different MPI ranks.

4: Optimize the PCG framework:
Is there any opportunity to reuse intermediate values or reorder operations to make the solver faster?
Do the methods you used affect the residual error, and why?

5: Scaling: Evaluate strong and weak scaling of your distributed PCG solver by conducting tests
with varying numbers of ranks to understand the performance characteristics of your implementation.
Evaluate speedups across 1-64 ranks for one node and across different N x N matrices (N=2^20 - 2^26).

Strong scaling: fixed problem size, increase processors
Weak scaling: size scales with processors

Strong: ranks=64, N for 2^20, 2^22, 2^24, 2^26
Weak: rank, N: 2, 2^20; 4, 2^21; 8, 2^22; 16, 2^23; 32, 2^24; 64, 2^26;
*/

// Compressed sparse row implementation of a sparse matrix.
class Matrix_csr{
  public:
    std::vector<double> values, col_index, row_index;
    int nbrow, nbcol;

    int NbRow() const {return nbrow;}
    int NbCol() const {return nbcol;}

    // // This uses the same shortcuts found in the default Matrix constructor to quickly build a 1d Poisson.
    // Matrix_csr(const int& nr = 0, const int& nc = 0) {
    //   nbrow = nr;
    //   nbcol = nc;
    //   int elements_count = 0;
    //   row_index.push_back(elements_count);
    //   for (int i = 0; i < nc; ++i) {
    //     if (i - 1 >= 0) {
    //       values.push_back(-1);
    //       col_index.push_back(i-1);
    //       elements_count++;
    //     }
    //     values.push_back(2);
    //     col_index.push_back(i);
    //     elements_count++;
    //     if (i + 1 < nc) {
    //       values.push_back(-1);
    //       col_index.push_back(i+1);
    //       elements_count++;
    //     }
    //     row_index.push_back(elements_count);
    //   }
    // };

    // Special constructor that produces a slice of a full csr matrix from rows rstart to rend.
    // total denotes number of ranks active in this MPI run. rank is self explanatory.
    // Running this without a specified rank and total values generate a full csr matrix.
    Matrix_csr(const int& nr = 0, const int& nc = 0, int rank = 0, int total = 1) {
      //std::cout << "Rank " << rank << " making a partial CSR" << std::endl;
      //std::cout << "DEBUG: nr, nc, rank, total: " << nr << ", " << nc << ", " << rank << ", " << total << std::endl;
      nbrow = nr;
      nbcol = nc;
      // Calculate the slice this MPI rank should have, based on the provided rank value.
      int step = nr / total;
      int rstart = step * rank;
      int rend = step * (rank+1);
      int elements_count = 3 * (rstart-1) + 2; // row 0 has 2 non-0 values, all subsequent ones have 3
      if (rank == 0) elements_count = 0;
      //std::cout << "DEBUG: start and end rows are: " << rstart << ", " << rend-1 << std::endl;
      //std::cout << "DEBUG: nr, nc: " << nr << ", " << nc << std::endl;
      row_index.push_back(elements_count);
      for (int i = rstart; i < rend; ++i) {
        if (i - 1 >= 0) {
          values.push_back(-1);
          col_index.push_back(i-1);
          elements_count++;
        }
        values.push_back(2);
        col_index.push_back(i);
        elements_count++;
        if (i + 1 < nc) {
          values.push_back(-1);
          col_index.push_back(i+1);
          elements_count++;
        }
        row_index.push_back(elements_count);
      }
    };

    // // matrix-vector product with vector xi. This operator gets used at 3 places in the PCG loop.
    // std::vector<double> operator*(const std::vector<double>& xi) const {
    //   std::vector<double> b(row_index.size()-1, 0.); // init vector with all 0s
    //   // Using row_index, identify the range of each row in col_index and values.
    //   for (size_t row = 1; row < row_index.size(); row++) {
    //     // Subtract row_index[0] b/c partial CSR matrices have absolute row index values
    //     // while iterating to the right values require relative row index values
    //     int start = row_index[row-1] - row_index[0];
    //     int end = row_index[row] - row_index[0];
    //     for (int i = start; i < end; i++) { // iterate thru each nonzero element in a given row
    //       int column = col_index[i];
    //       int value = values[i]; 
    //       b[row-1] += value * xi[column]; // for each row: multiply every nonzero element by corresponding val in xi
    //     }
    //   }
    //   return b;
    // }

    // https://mathinsight.org/matrix_vector_multiplication
    // Self reminder: Each row in A multiplied by the entire vector xi forms one entry in the output vector b
    // Row 0 of A contributes to b[0], Row 1 contributes to b[1], so on
    std::vector<double> operator*(const std::vector<double>& xi) const {
      // Sandbox function where I build an initial skeleton of a distributed mult.
      int size; // total number of ranks at play
      MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
      //std::cout << "DEBUG: final and local results sizes: " << nbrow << ", " << nbrow/size << std::endl;
      std::vector<double> final_results(nbrow, 0);
      std::vector<double> local_results(nbrow/size, 0);
      // We know that each rank must have access to the full values of xi
      // And that each rank has a slice of A
      // Each rank must calculate a partial slice of b, then merge all values together to get a full b
      // I can either gather them to one b at rank==0 or allgather them to one b at every rank
      for (size_t row = 1; row < row_index.size(); row++) {
        int start = row_index[row-1]- row_index[0];
        int end = row_index[row]- row_index[0];
        for (int i = start; i < end; i++) { // iterate thru each nonzero element in a given row
          int column = col_index[i];
          int value = values[i]; 
          local_results[row-1] += value * xi[column]; // for each row: multiply every nonzero element by corresponding val in xi
        }
      }
      // Using allgather because each proc runs its own pcg job, thus each proc needs the full results to continue
      // Also I can't let the procs go out of sync with rank 0 having correct data and the rest having partial broken data
      // the whole pcg logic breaks when that happens and I can't be bothered to rebuild the logic to account for that
      // This introduces major slowdowns from communications bottlenecking but I don't have a better solution
      MPI_Allgather(&local_results[0], nbrow/size, MPI_DOUBLE, &final_results[0], nbrow/size, MPI_DOUBLE, MPI_COMM_WORLD);
      //MPI_Gather(&local_results[0], nbrow/size, MPI_DOUBLE, &final_results[0], nbrow/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      return final_results;

    }

    // Utility function to print the csr matrix. Used for debug purposes.
    void print() {
      //std::cout << "Compressed Sparse Row matrix representation" << std::endl;
      std::cout << "values: [ ";
      for (int value : values) {
        std::cout << value << ",";
      }
      std::cout << "]" << std::endl;
      std::cout << "col_index: [ ";
      for (int col : col_index) {
        std::cout << col << ",";
      }
      std::cout << "]" << std::endl;
      std::cout << "row_index: [ ";
      for (int row : row_index) {
        std::cout << row << ",";
      }
      std::cout << "]" << std::endl;
    }

    void print_mpi() {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
      std::cout << "MPI rank #" << rank << " printing slice of full matrix" << std::endl;
      print();
    }
};

// original coordinates-based implementation, left here for reference.
class Matrix{
  public:
    typedef std::pair<int, int> N2;
  
    std::map<N2, double> data;
    int nbrow;
    int nbcol;

    // Constructor. Produces matrix with dims nr x nc.
    Matrix(const int& nr = 0, const int& nc = 0, int rank = 0): nbrow(nr), nbcol(nc) {
      // rank included in here for debug purposes and does nothing else
      //std::cout << "Rank " << rank << " making a partial old matrix" << std::endl;
      //std::cout << "DEBUG: nr, nc: " << nr << ", " << nc << std::endl;
      for (int i = 0; i < nc; ++i) {
        data[std::make_pair(i, i)] = 2.0;
        if (i - 1 >= 0) data[std::make_pair(i, i - 1)] = -1.0;
        if (i + 1 < nc) data[std::make_pair(i, i + 1)] = -1.0;
      }
    }; 
  
    int NbRow() const {return nbrow;}
    int NbCol() const {return nbcol;}
  
    // matrix-vector product with vector xi
    std::vector<double> operator*(const std::vector<double>& xi) const {
      std::vector<double> b(NbRow(), 0.);
      for(auto it = data.begin(); it != data.end(); ++it){
        int j = (it->first).first; // x coord
        int k = (it->first).second; // y coord
        double Mjk = it->second; // value at coords x,y
        b[j] += Mjk * xi[k];
      }
  
      return b;
    }

    void print() {
      std::cout << "Printing coordinates based matrix" << std::endl;
      for(auto it = data.begin(); it != data.end(); ++it){
        int j = (it->first).first;
        int k = (it->first).second; 
        double Mjk = it->second;
        std::cout << "[" << j << ", " << k << ", " << Mjk << "]" << std::endl;
      }
    }
};
  
// scalar product (u, v)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  double sp = 0.;
  for(int j = 0; j < u.size(); j++)
    sp += u[j] * v[j];
  return sp; 
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  std::vector<double> w = u;
  for(int j = 0; j < u.size(); j++)
    w[j] += v[j];
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j = 0; j < w.size(); j++) 
    w[j] = a * u[j];
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  for(int j = 0; j < u.size(); j++)
    u[j] += v[j];
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i = 0; i < u.size(); i++) 
    b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (int i = 0; i < u.size(); i++) 
    x[i] = xe[i];
  return x;
}

Matrix A;
Matrix_csr Acsr;
std::vector<Eigen::Triplet<double>> coefficients;

// Old CG_Solver constructor kept here for reference.
CG_Solver::CG_Solver(const int& n, const int& N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
  A = Matrix(n, N, rank);
}

/* N is the size of the matrix, and n is the number of rows assigned per rank.
 * It is your responsibility to generate the input matrix, assuming the ranks are 
 * partitioned rowwise.
 * The input matrix is L, where L represents a discretized 1D Possion's equation.
 * That is to say L has 2s on its diagonal and -1s on it super/sub-diagonals.
 * See the constructor of the Matrix structure as an example.
 * The constructor of CG_Solver will not be included in the timing result.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
CG_Solver_csr::CG_Solver_csr(const int& n, const int& N) {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
  Acsr = Matrix_csr(N, N, rank, size);

  // Since each process has only a partial slice of the overall csr matrix
  // Each process creates a partial coefficients vector
  // I need every process to have a full coefficients vector
  // I can't allgather eigen triplets using default MPI datatypes
  // and I don't feel like making a custom datatype.
  // I am just going to use a locally made Matrix_csr to build a full coefficients ahead of time.
  // Especially since coefficients never gets changed after population.
  Matrix_csr temp_full_csr = Matrix_csr(Acsr.NbRow(), Acsr.NbCol(), 0, 1);
  int total_rows = temp_full_csr.row_index.size();
  for (int i = 1; i < total_rows; i++) {
    int start = temp_full_csr.row_index[i-1] - temp_full_csr.row_index[0];
    int end = temp_full_csr.row_index[i] - temp_full_csr.row_index[0];
    for (int j = start; j < end; j++) {//iterate through each row, insert values and x-y coords to coefficients
      int col = temp_full_csr.col_index[j];
      int value = temp_full_csr.values[j];
      // offset: how many rows exist before this csr slice; used to compute true row indices for a given point
      //int offset = (Acsr.NbRow()/size) * rank;
      // coeffs_local.push_back(i-1+offset);
      // coeffs_local.push_back(col);
      // coeffs_local.push_back(value);
      coefficients.push_back(Eigen::Triplet<double>(i-1, col, value));
    }
  }
}

/* The preconditioned conjugate gradient method solving Ax = b with tolerance tol.
 * This is the function being evalauted for performance.
 * Note that the starter code only works for 1 rank and it is not efficient.
 * The vector b is all 1s, and the PCG method starts with an initial guess x of all 0s.
 */
void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.NbRow();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for(auto it = A.data.begin(); it != A.data.end(); ++it){
    // For each data point in A: grab the x, y coordinate and value
    int j = (it->first).first;
    int k = (it->first).second;
    // Insert each x, y coord and corresponding value to coefficients
    coefficients.push_back(Eigen::Triplet<double>(j, k, it -> second)); 
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n, n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  const double epsilon = tol * std::sqrt((b, b));
  x.assign(b.size(), 0.);
  std::vector<double> r = b, z = prec(P, b), p = z;
  double alpha = 0., beta = 0.;
  double res = std::sqrt((r, r));

  int num_it = 0;
  
  while(res >= epsilon) {
    //std::vector<double> Ap = A * p;
    //double pAp = (p, Ap);
    alpha = (r, z) / (p, A * p); // alpha is a double
    x += (+alpha) * p; 
    r += (-alpha) * (A * p);
    z = prec(P, r);
    beta = (r, z) / (alpha * (p, A * p)); 
    p = z + beta * p;    
    res = std::sqrt((r, r));
    
    num_it++;
    if (rank == 0 && !(num_it % 1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << res << "\n";
    }
  }
}

// debug print func
void print_coeffs(std::vector<double> coef) {
  for (int i = 0; i < coef.size(); i+=3) {
    std::cout << coef[i] << ", " << coef[i+1] << ", " << coef[i+2] << ", " << std::endl;
  }
  //std::cout << "END OF PARALLEL COMPILED EIGEN VEC" << std::endl;
}

/* The preconditioned conjugate gradient method solving Ax = b with tolerance tol.
 * This is the function being evalauted for performance.
 * Note that the starter code only works for 1 rank and it is not efficient.
 * The vector b is all 1s, and the PCG method starts with an initial guess x of all 0s.
 */
void CG_Solver_csr::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  const int MASTER = 0;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  
  // // get the local diagonal block of A
  // std::vector<Eigen::Triplet<double>> coefficients;

  // // Since each process has only a partial slice of the overall csr matrix
  // // Each process creates a partial coefficients vector
  // // I need every process to have a full coefficients vector
  // // I can't allgather eigen triplets using default MPI datatypes
  // // and I don't feel like making a custom datatype.
  // // I am just going to build 
  // std::vector<double> coeffs_local;
  // Matrix_csr temp_full_csr = Matrix_csr(Acsr.NbRow(), Acsr.NbCol(), 0, 1);
  // int total_rows = temp_full_csr.row_index.size();
  // for (int i = 1; i < total_rows; i++) {
  //   int start = temp_full_csr.row_index[i-1] - temp_full_csr.row_index[0];
  //   int end = temp_full_csr.row_index[i] - temp_full_csr.row_index[0];
  //   for (int j = start; j < end; j++) {//iterate through each row, insert values and x-y coords to coefficients
  //     int col = temp_full_csr.col_index[j];
  //     int value = temp_full_csr.values[j];
  //     // offset: how many rows exist before this csr slice; used to compute true row indices for a given point
  //     //int offset = (Acsr.NbRow()/size) * rank;
  //     // coeffs_local.push_back(i-1+offset);
  //     // coeffs_local.push_back(col);
  //     // coeffs_local.push_back(value);
  //     coefficients.push_back(Eigen::Triplet<double>(i-1, col, value));
  //   }
  // }
  
  
  int n = Acsr.NbRow();
  
  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n, n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  const double epsilon = tol * std::sqrt((b, b));
  x.assign(b.size(), 0.);
  std::vector<double> r = b, z = prec(P, b), p = z;
  double alpha = 0., beta = 0.;
  double res = std::sqrt((r, r));

  int num_it = 0;
  
  // Thoughts:
  // A * p shows up 3 times; we can precalculate the value then reuse it
  // (p, A * p) shows up 2 times; once again, precalculate
  // This works b/c A is not changed and p is updated after A*p and (p, A*p) gets used
  // Better to do matmul once than 3 times
  // Note that A is called Acsr here hence different var names.
  // I do not expect this to affect error because Acsr * p is the same when precomputed or done on the spot.
  int master_is_done = 0;
  while(res >= epsilon) {
    // if (master_is_done) {
    //   break;
    // }
    if (rank == MASTER) {
      std::cout << "Rank 0 doing an iteration" << std::endl;
    }
    std::vector<double> Acsrp = Acsr * p;
    double pAcsrp = (p, Acsrp);
    std::cout << "Rank 0 init Acsrp" << std::endl;
    alpha = (r, z) / pAcsrp;
    x += (+alpha) * p; 
    r += (-alpha) * Acsrp;
    std::cout << "Rank 0 updating residuals" << std::endl;
    z = prec(P, r);
    beta = (r, z) / (alpha * pAcsrp); 
    p = z + beta * p;    
    res = std::sqrt((r, r));
    
    num_it++;
    //if (rank == 0 && !(num_it % 1)) {
      std::cout << "Rank: " << rank << "\n";
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << res << "\n";
    //}
  }
  master_is_done = 1;
  if (rank == MASTER) {
    std::cout << "Rank 0 done" << std::endl;
    MPI_Bcast(&master_is_done, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  }
}

/*
// A snippet of code for a previous attempt at building a full eigen coefficients vector.
// The following approach has each process build a local list of (j, k, value) data points
// then allgather so all procs have all data points, then insert into coefficients

std::vector<double> coeffs_local;
  int total_rows = Acsr.row_index.size();
  for (int i = 1; i < total_rows; i++) {
    int start = Acsr.row_index[i-1] - Acsr.row_index[0];
    int end = Acsr.row_index[i] - Acsr.row_index[0];
    for (int j = start; j < end; j++) {//iterate through each row, insert values and x-y coords to coefficients
      int col = Acsr.col_index[j];
      int value = Acsr.values[j];
      // offset: how many rows exist before this csr slice; used to compute true row indices for a given point
      int offset = (Acsr.NbRow()/size) * rank;
      coeffs_local.push_back(i-1+offset);
      coeffs_local.push_back(col);
      coeffs_local.push_back(value);
    }
  }
  if (rank == MASTER || rank == size-1) {
    // first and last rows only have 2 nonzeros instead of 3; allgather requires same amt of memory from all procs
    // this ensures the condition by filling in with junk data; I will dispose of the junk data later
    //std::cout << "Rank 0 or size-1 adding junk to ensure vector size matches" << std::endl;
    coeffs_local.push_back(0);
    coeffs_local.push_back(0);
    coeffs_local.push_back(0);
  }
  //std::cout << "Rank " << rank << " Checking coeffs for proper vals" << std::endl;
  //print_coeffs(coeffs_local);
  std::vector<double> coefficients_all(coeffs_local.size()*size, 0);
  //std::cout << "Rank " << rank << " alltogether: DEBUG: sizes of local and overall coeffs: " << coeffs_local.size() << ", " << coefficients_all.size() << std::endl;
  MPI_Allgather(&coeffs_local[0], coeffs_local.size(), MPI_DOUBLE, &coefficients_all[0], coeffs_local.size(), MPI_DOUBLE, MPI_COMM_WORLD);
  // now with the full (j, k, value) sets, populate coefficients for real this time
  for (int i = 0; i < coefficients_all.size(); i += 3) {
    if (coefficients_all[i+2] != 0) { // value == 0 means junk data that should not be included
      coefficients.push_back(Eigen::Triplet<double>(coefficients_all[i],coefficients_all[i+1],coefficients_all[i+2])); 
    }
  }
*/