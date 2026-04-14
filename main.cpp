#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "common.h"

// ./pcg -N <size of the matrix>

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  //std::cout << "DEBUG: Size and rank count for this PCG run: " << size << ", " << rank << std::endl;

  if (find_arg_idx(argc, argv, "-h") >= 0) {
      std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
      return 0;
  }

  int N = find_int_arg(argc, argv, "-N", 1 << 20); // global size

  // The old CG_Solver code is not parallel. I did not change that since my work focuses on a new CG_Solver_csr.
  // I made some patchwork changes to make the old CG_Solver work even with many ranks.
  // I Left the code here mostly for reference.

  assert(N % size == 0);
  int n = N / size; // number of local rows

  // generate L + I
  CG_Solver cg(N, N);

  // initial guess
  std::vector<double> x(N, 0);

  // right-hand side
  std::vector<double> b(N, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();
  //std::cout << "Rank " << rank << " doing CG solve (old)" << std::endl;

  cg.solve(b, x, 1e-8);

  MPI_Barrier(MPI_COMM_WORLD);

  // Do not modify this line, use for grading
  if (rank == 0) {
    std::cout << "Time for CG (old) of size " << N << " with " 
              << size << " rank(s): " << MPI_Wtime() - time 
              << " seconds." << std::endl;
  }
  
  std::vector<double> global_x;
  
  if (rank == 0)
    global_x.resize(N);
  //std::cout << "Beginning MPI gather of local x to global x" << std::endl;
  global_x = x;
  //MPI_Gather(x.data(), n, MPI_DOUBLE, global_x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Rank 0 evaluating results of old serial PCG" << std::endl;
    double r_square = 0;
    for (int i = 0; i < N; ++i) {
      double r = global_x[i] * 2;
      if (i > 0)  r -= global_x[i - 1];
      if (i + 1 < N)  r -= global_x[i + 1];
      r_square += (r - 1) * (r - 1);
    }
    std::cout << "|Ax - b| / |b| = " << std::sqrt(r_square) / std::sqrt(N) << std::endl;
  }

  // Do the same thing above, but now with the distributed csr pcg
  // Notice that X and b is initialized fully for all ranks. See comments for the matvec mult operator overload
  // in distributed_pcg.cpp for Matrix_csr for an explanation why.

  // generate L + I
  CG_Solver_csr cgcsr(n, N);

  // initial guess
  std::vector<double> xcsr(N, 0);

  // right-hand side
  std::vector<double> bcsr(N, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time_csr = MPI_Wtime();
  //std::cout << "Rank " << rank << " doing CG solve (new, with CSR matrix and MPI parallelization)" << std::endl;

  cgcsr.solve(bcsr, xcsr, 1e-8);

  MPI_Barrier(MPI_COMM_WORLD);

  // Do not modify this line, use for grading
  if (rank == 0) {
    std::cout << "Time for CG (csr+MPI) of size " << N << " with " 
              << size << " rank(s): " << MPI_Wtime() - time_csr
              << " seconds." << std::endl;
  }
  
  std::vector<double> global_x_csr;
  
  if (rank == 0)
    global_x_csr.resize(N);
  //std::cout << "Beginning MPI gather of local x to global x" << std::endl;
  // This current setup has every proc do its own full PCG solving. Since they all use the same numbers + inputs
  // the output should be all the same. If this is production code I should make it so that each proc
  // does some of the computing and produces partial x results to be gathered.
  // However I don't want to poke around with PCG math I don't fully understand and break things in ways
  // that I cannot fix. Therefore I am leaving things as-is.
  global_x_csr = xcsr;
  //MPI_Gather(x.data(), n, MPI_DOUBLE, global_x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Rank 0 evaluating results of parallelized PCG" << std::endl;
    double r_square = 0;
    for (int i = 0; i < N; ++i) {
      double r = global_x_csr[i] * 2;
      if (i > 0)  r -= global_x_csr[i - 1];
      if (i + 1 < N)  r -= global_x_csr[i + 1];
      r_square += (r - 1) * (r - 1);
    }
    std::cout << "|Ax - b| / |b| = " << std::sqrt(r_square) / std::sqrt(N) << std::endl;
  }

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}