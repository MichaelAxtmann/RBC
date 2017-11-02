/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include <random>
#include <vector>
#include "Sort/SQuick.hpp"

#define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Create random input elements
    PRINT_ROOT("Create random input elements");
    std::mt19937 generator;
    int data_seed = 3469931 + rank;
    generator.seed(data_seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<double> data;
    for (int i = 0; i < 10; ++i)
        data.push_back(dist(generator));
    int global_elements = 10 * size;
    
    // Sort data descending
    PRINT_ROOT("Start sorting algorithm with MPI_Comm");
    SQuick::sort(comm, data, global_elements, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");
    
    PRINT_ROOT("Start sorting algorithm with RBC::Comm");
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(comm, &rcomm);
    SQuick::sort(rcomm, data, global_elements, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    PRINT_ROOT("Start sorting algorithm with RBC::Comm " <<
               "but use MPI communicators and collectives");
    RBC::Comm rcomm1;
    RBC::Create_Comm_from_MPI(comm, &rcomm1, true, true);
    SQuick::sort(rcomm1, data, global_elements, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    // Finalize the MPI environment
    MPI_Finalize();
}
