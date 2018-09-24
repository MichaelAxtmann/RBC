/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "Sort/SQuick.hpp"
#include "tlx/math.hpp"
#include "tlx/algorithm.hpp"

#include <mpi.h>
#include "sstream" /* stringstream */
#include <random>
#include <vector>
#include <stdlib.h> /* srand, rand */
#include <cstdlib>

#define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "]";
  }
  return out;
}

void GenerateData(std::vector<long>& send, std::vector<long>& recv) {
    for (size_t idx = 0; idx != send.size(); ++idx) {
        send[idx] = rand();
    }    
    for (size_t idx = 0; idx != recv.size(); ++idx) {
        recv[idx] = 0;
    }    
}

#define PrintDistributed(string)                                                                           \
    {std::stringstream buffer;                                          \
    buffer << rank << ": " << string << "\n";                              \
    MPI_File_write_ordered( fh, buffer.str().c_str(), buffer.str().size(), MPI_CHAR, MPI_STATUS_IGNORE );}


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 2) {
        std::cout << "Error: We expect a single argument. "
                  << "Pass 0 for MPI and pass 1 for RBC." << std::endl;
    }
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(comm, &rcomm);

    bool mpi = atoi(argv[1]);

    MPI_File fh;
    int err = 0;
    if (mpi) {
        err = MPI_File_open( comm, "out_mpi.log", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh );
    } else {
        err = MPI_File_open( comm, "out_rbc.log", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh );
    }        
    if (err)
    {
        MPI_Abort(MPI_COMM_WORLD, 911);
    }

    srand(rank + 1);

    MPI_Datatype type = MPI_LONG;
    for (int num_els = 0; num_els != 50; ++num_els) {
        std::vector<long> send(num_els);
        std::vector<long> recv(num_els * size);

        if (mpi) {
            GenerateData(send, recv);
            MPI_Allgather(send.data(), num_els, type, recv.data(), num_els, type, comm);
            PrintDistributed("AllgatherDissemination: " << std::endl << send << std::endl << recv);

            {
                std::vector<long> send(rand() % (num_els + 1));
                int send_cnt = send.size();
                std::vector<int> sizes(size);
                RBC::Allgather(&send_cnt, 1, MPI_INT, sizes.data(), 1, MPI_INT, rcomm);
                std::vector<int> sizes_exscan(size + 1, 0);
                tlx::exclusive_scan(sizes.begin(), sizes.end(), sizes_exscan.begin(), 0);
                std::vector<long> recv(sizes_exscan.back());
                if (recv.size()) MPI_Allgatherv(send.data(), send.size(),
                        type, recv.data(), sizes.data(), sizes_exscan.data(),
                        type, comm);
                PrintDistributed("AllgathervDissemination: " << std::endl << send << std::endl << recv);
            }
            
            if (tlx::is_power_of_two(size)) {
                GenerateData(send, recv);
                MPI_Allgather(send.data(), num_els, type, recv.data(), num_els, type, comm);
                PrintDistributed("AllgatherHypercube: " << std::endl << send << std::endl << recv);
            }

            GenerateData(send, recv);
            MPI_Allgather(send.data(), num_els, type, recv.data(), num_els, type, comm);
            PrintDistributed("AllgatherPipeline: " << std::endl << send << std::endl << recv);
        
        } else {

            GenerateData(send, recv);
            RBC::_internal::optimized::AllgatherDissemination(send.data(), num_els, type, recv.data(), num_els, type, rcomm);
            PrintDistributed("AllgatherDissemination: " << std::endl << send << std::endl << recv);

            {
                std::vector<long> send(rand() % (num_els + 1));
                int send_cnt = send.size();
                int recv_cnt = 0;
                RBC::Allreduce(&send_cnt, &recv_cnt, 1, MPI_INT, MPI_SUM, rcomm);
                std::vector<long> recv(recv_cnt);
                RBC::_internal::optimized::AllgathervDissemination(send.data(), send.size(),
                        type, recv.data(), recv.size(), type, rcomm);
                PrintDistributed("AllgathervDissemination: " << std::endl << send << std::endl << recv);
            }
            
            if (tlx::is_power_of_two(size)) {
                GenerateData(send, recv);
                RBC::_internal::optimized::AllgatherHypercube(send.data(), num_els, type, recv.data(), num_els, type, rcomm);
                PrintDistributed("AllgatherHypercube: " << std::endl << send << std::endl << recv);
            }
                
            GenerateData(send, recv);
            RBC::_internal::optimized::AllgatherPipeline(send.data(), num_els, type, recv.data(), num_els, type, rcomm);
            PrintDistributed("AllgatherPipeline: " << std::endl << send << std::endl << recv);

        }

        if (mpi) {
            GenerateData(send, recv);
            MPI_Allreduce(send.data(), recv.data(), num_els, type, MPI_SUM, comm);
            PrintDistributed("AllreduceScatterAllgather: " << std::endl << send << std::endl << recv);

            GenerateData(send, recv);
            MPI_Allreduce(send.data(), recv.data(), num_els, type, MPI_SUM, comm);
            PrintDistributed("AllreduceHypercube: " << std::endl << send << std::endl << recv);
        
        } else {

            GenerateData(send, recv);
            RBC::_internal::optimized::AllreduceScatterAllgather(send.data(), recv.data(), num_els, type, MPI_SUM, rcomm);
            PrintDistributed("AllreduceScatterAllgather: " << std::endl << send << std::endl << recv);

            GenerateData(send, recv);
            RBC::_internal::optimized::AllreduceHypercube(send.data(), recv.data(), num_els, type, MPI_SUM, rcomm);
            PrintDistributed("AllreduceHypercube: " << std::endl << send << std::endl << recv);

        }

        if (mpi) {
            GenerateData(send, recv);
            MPI_Scan(send.data(), recv.data(), num_els, type, MPI_SUM, comm);
            PrintDistributed("Scan: " << std::endl << send << std::endl << recv);
        } else {

            GenerateData(send, recv);
            RBC::_internal::optimized::Scan(send.data(), recv.data(), num_els, type, MPI_SUM, rcomm);
            PrintDistributed("Scan: " << std::endl << send << std::endl << recv);
        }

        if (mpi) {
            GenerateData(send, recv);
            MPI_Exscan(send.data(), recv.data(), num_els, type, MPI_SUM, comm);
            PrintDistributed("Exscan: " << std::endl << send << std::endl << recv);
        } else {

            GenerateData(send, recv);
            RBC::_internal::optimized::Exscan(send.data(), recv.data(), num_els, type, MPI_SUM, rcomm);
            PrintDistributed("Exscan: " << std::endl << send << std::endl << recv);
        }

        for (int root = 0; root != size; ++root) {
            if (mpi) {
                GenerateData(send, recv);
                MPI_Bcast(send.data(), num_els, type, root, comm);
                PrintDistributed("BcastBinomial: " << std::endl << send << std::endl << recv);

                GenerateData(send, recv);
                MPI_Bcast(send.data(), num_els, type, root, comm);
                PrintDistributed("BcastScatterAllgather: " << std::endl << send << std::endl << recv);
        
            } else {

                GenerateData(send, recv);
                RBC::_internal::optimized::BcastBinomial(send.data(), num_els, type, root, rcomm);
                PrintDistributed("BcastBinomial: " << std::endl << send << std::endl << recv);

                GenerateData(send, recv);
                RBC::_internal::optimized::BcastScatterAllgather(send.data(), num_els, type, root, rcomm);
                PrintDistributed("BcastScatterAllgather: " << std::endl << send << std::endl << recv);

            }
        }
    }

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

    // // Bcast
    // {
    //     int root = 3;
    //     std::vector<size_t> send{4, 3, 2, 1};
    //     if (rank != root) {
    //         send = std::vector<size_t>(send.size(), 0);
    //     }
        
    //     PRINT_ROOT("Start broadcast test");
    //     RBC::Comm rcomm;
    //     RBC::Create_Comm_from_MPI(comm, &rcomm);

    //     RBC::_internal::optimized::BcastScatterAllgather(send.data(), send.size(), MPI_UNSIGNED_LONG,
    //             root, rcomm);

    // }        
    
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
