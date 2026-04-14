// This is a sandbox file for testing out some CSR matrix functionality.
// The main function tests out an overloaded Matrix_csr constructor that creates slices of a full CSR matrix.
// Use the following commands to compile and run:
// mpic++ -g -Wall -o matrix_csr_tests matrix_csr_tests.cpp
// mpiexec -n NUM_OF_RANKS ./matrix_csr_tests
// replace NUM_OF_RANKS with a number. I recommend making NUM_OF_RANKS a factor of the matrix size on line 156.

//#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>
#include <map>
#include <vector>
#include <tuple>
//#include <Eigen/Sparse>


void printvec(std::vector<double> vec) {
  std::cout << "[";
  for (double d : vec) {
    std::cout << d << ", ";
  }
  std::cout << "]" << std::endl;
}

/*
2: Layout Formulation: Figure out how to distribute the sparse matrix format across MPI processes.

3: Parallelize matrix-vector multiplication and norm across different MPI ranks.
*/

// original coordinates-based implementation, left here for reference.
class Matrix{
  public:
    typedef std::pair<int, int> N2;
  
    std::map<N2, double> data;
    int nbrow;
    int nbcol;

    // Constructor. Produces matrix with dims nr x nc.
    Matrix(const int& nr = 0, const int& nc = 0): nbrow(nr), nbcol(nc) {
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


int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  const int MASTER = 0;
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  std::cout << "MPI rank " << rank << " reporting" << std::endl;

  std::vector<double> b_mult = {2,4,6,8,12,14,16,18};
  // if (rank == MASTER) {
  //   Matrix_csr small_matrix = Matrix_csr(9,9,0,1);
  //   Matrix matrix_old = Matrix(9,9);
  //   std::vector<double> Ab = small_matrix*b_mult;
  //   std::cout << "ACTUAL ANSWER " << "[";
  //   for (double d : Ab) {
  //     std::cout << d << ", ";
  //   }
  //   std::cout << "]" << std::endl;

  //   Ab = matrix_old*b_mult;
  //   std::cout << "ACTUAL ANSWER " <<  "[";
  //   for (double d : Ab) {
  //     std::cout << d << ", ";
  //   }
  //   std::cout << "]" << std::endl;
  // }
  // if (rank == MASTER) {
  //   std::cout << "TESTING DISTRIBUTED MATVEC" << std::endl << std::endl;
  // }

  Matrix_csr Acsr = Matrix_csr(8,8,rank,size);

  if (rank == 0) {
    std::vector<double> test = Acsr * b_mult;
    for (double d : test) {
      std::cout << d << ", ";
    }
    std::cout << std::endl;
  }
  // std::vector<double> parallel_mult_results = partial_matrix.placeholder_distr_mult(b_mult);
  // std::cout << "PARALLEL MPI MATVEC RESULTS FROM RANK " << rank <<  "[";
  // for (double d : parallel_mult_results) {
  //   std::cout << d << ", ";
  // }
  // std::cout << "]" << std::endl;
  // Acsr.print();
  // std::vector<double> coeffs_local;
  // int total_rows = Acsr.row_index.size();
  // //int total_rows = Acsr.row_index.size();
  // for (int i = 1; i < total_rows; i++) {
  //   int start = Acsr.row_index[i-1] - Acsr.row_index[0];
  //   int end = Acsr.row_index[i] - Acsr.row_index[0];
  //   for (int j = start; j < end; j++) {//iterate through each row, insert values and x-y coords to coefficients
  //     int col = Acsr.col_index[j];
  //     int value = Acsr.values[j];
  //     // offset: how many rows exist before this csr slice; used to compute true row indices for a given point
  //     int offset = (Acsr.NbRow()/size) * rank;
  //     coeffs_local.push_back(i-1+offset);
  //     coeffs_local.push_back(col);
  //     coeffs_local.push_back(value);
  //   }
  // }
  // if (rank == MASTER || rank == size-1) {
  //   // first and last rows only have 2 nonzeros instead of 3; allgather requires same amt of memory from all procs
  //   // this ensures the condition by filling in with junk data; I will dispose of the junk data later
  //   coeffs_local.push_back(0);
  //   coeffs_local.push_back(0);
  //   coeffs_local.push_back(0);
  // }
  // std::vector<double> coefficients_all(coeffs_local.size()*size, 0);
  // MPI_Allgather(&coeffs_local[0], coeffs_local.size(), MPI_DOUBLE, &coefficients_all[0], coeffs_local.size(), MPI_DOUBLE, MPI_COMM_WORLD);
  // if (rank == MASTER) {
  //   for (int i = 0; i < coefficients_all.size(); i+=3) {
  //     std::cout << coefficients_all[i] << ", " << coefficients_all[i+1] << ", " << coefficients_all[i+2] << ", " << std::endl;
  //   }
  //   std::cout << "END OF PARALLEL COMPILED EIGEN VEC" << std::endl;
  // }

  // if (rank == MASTER) {
  //   Matrix A = Matrix(16,16);
  //   std::vector<std::tuple<double, double, double>> coefficients2;
  //   for(auto it = A.data.begin(); it != A.data.end(); ++it){
  //     // For each data point in A: grab the x, y coordinate and value
  //     int j = (it->first).first;
  //     int k = (it->first).second;
  //     // Insert each x, y coord and corresponding value to coefficients
  //     coefficients2.push_back(std::tuple<double, double, double>(j, k, it -> second)); 
  //   }
  //   for (std::tuple<double, double, double> trip : coefficients2) {
  //     std::cout << std::get<0>(trip) << ", " << std::get<1>(trip) << ", " << std::get<2>(trip) << std::endl;
  //   }
  //   std::cout << "END OF SERIAL COMPILED EIGEN VEC" << std::endl;
  // }
  // std::vector<double> x = {1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18};
  // std::vector<double> b = Acsr * x;
  // std::cout << "[";

  // for (double d : b) {
  //   std::cout << d << ", ";
  // }

  // std::cout << "]" << std::endl;

  MPI_Finalize(); // Finalize the MPI environment
}