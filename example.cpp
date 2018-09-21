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

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // // Allgather
    // {
    //     std::vector<size_t> send(2, rank);
    //     std::vector<size_t> recv(2 * size, 0);

    //     PRINT_ROOT("Start Allgather test");
    //     RBC::Comm rcomm;
    //     RBC::Create_Comm_from_MPI(comm, &rcomm);

    //     RBC::_internal::optimized::AllgatherPipeline(send.data(), send.size(), MPI_UNSIGNED_LONG,
    //             recv.data(), send.size(), MPI_UNSIGNED_LONG,
    //             rcomm);

    //     std::cout << "rank " << rank << ": " << recv << std::endl;
    // }

    // // Allreduce
    // {
    //     std::vector<size_t> send{1,2, 3, 4};
    //     std::vector<size_t> recv(4, 0);

    //     PRINT_ROOT("Start Allreduce test");
    //     RBC::Comm rcomm;
    //     RBC::Create_Comm_from_MPI(comm, &rcomm);

    //     // RBC::_internal::optimized::AllreduceScatterAllgather(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);
    //     RBC::_internal::optimized::AllreduceHypercube(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);

    //     std::cout << "rank " << rank << ": " << recv << std::endl;
    // }

    // // Scan
    // {
    //     std::vector<size_t> send{0, 1, 2, 3};
    //     std::vector<size_t> recv(4, 0);

    //     PRINT_ROOT("Start Allreduce test");
    //     RBC::Comm rcomm;
    //     RBC::Create_Comm_from_MPI(comm, &rcomm);

    //     // RBC::_internal::optimized::AllreduceScatterAllgather(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);
    //     RBC::_internal::optimized::Scan(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);

    //     std::cout << "rank " << rank << ": " << recv << std::endl;
    // }    

    // // Exscan
    // {
    //     std::vector<size_t> send{0, 1, 2, 3};
    //     std::vector<size_t> recv(4, 0);

    //     PRINT_ROOT("Start Allreduce test");
    //     RBC::Comm rcomm;
    //     RBC::Create_Comm_from_MPI(comm, &rcomm);

    //     // RBC::_internal::optimized::AllreduceScatterAllgather(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);
    //     RBC::_internal::optimized::Exscan(send.data(), recv.data(), send.size(), MPI_UNSIGNED_LONG, MPI_SUM, rcomm);

    //     std::cout << "rank " << rank << ": " << recv << std::endl;
    // }       

    // Bcast
    {
        int root = 3;
        std::vector<size_t> send{4, 3, 2, 1};
        if (rank != root) {
            send = std::vector<size_t>(send.size(), 0);
        }
        
        PRINT_ROOT("Start broadcast test");
        RBC::Comm rcomm;
        RBC::Create_Comm_from_MPI(comm, &rcomm);

        RBC::_internal::optimized::BcastScatterAllgather(send.data(), send.size(), MPI_UNSIGNED_LONG,
                root, rcomm);

        std::cout << "rank " << rank << ": " << send << std::endl;
    }        
    
    // // Create random input elements
    // PRINT_ROOT("Create random input elements");
    // std::mt19937 generator;
    // int data_seed = 3469931 + rank;
    // generator.seed(data_seed);
    // std::uniform_real_distribution<double> dist(-100.0, 100.0);
    // std::vector<double> data;
    // for (int i = 0; i < 10; ++i)
    //     data.push_back(dist(generator));
    // int global_elements = 10 * size;
    
    // // Sort data descending
    // PRINT_ROOT("Start sorting algorithm with MPI_Comm");
    // SQuick::sort(comm, data, global_elements, std::greater<double>());
    // PRINT_ROOT("Elements have been sorted");
    
    // PRINT_ROOT("Start sorting algorithm with RBC::Comm");
    // RBC::Comm rcomm;
    // RBC::Create_Comm_from_MPI(comm, &rcomm);
    // SQuick::sort(rcomm, data, global_elements, std::greater<double>());
    // PRINT_ROOT("Elements have been sorted");

    // PRINT_ROOT("Start sorting algorithm with RBC::Comm " <<
    //            "but use MPI communicators and collectives");
    // RBC::Comm rcomm1;
    // RBC::Create_Comm_from_MPI(comm, &rcomm1, true, true);
    // SQuick::sort(rcomm1, data, global_elements, std::greater<double>());
    // PRINT_ROOT("Elements have been sorted");

    // Finalize the MPI environment
    MPI_Finalize();
}
