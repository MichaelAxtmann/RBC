/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SQS_DATAEXCHANGE_HPP
#define SQS_DATAEXCHANGE_HPP

#include "../../RangeBasedComm/RBC.hpp"
#include "QSInterval.hpp"
#include "Constants.hpp"
#include <mpi.h>
#include <cassert>
#include <cstring>

#define W(X) #X << "=" << X << ", "

template<typename T>
class DataExchange_SQS {
    
public:
    
    /*
     * Exchange the data with other PEs
     */
    static void exchangeData(QSInterval_SQS<T> &ival) {
        long long recv_small = 0, recv_large = 0, recv_count_small = 0, recv_count_large = 0;
        std::vector<std::unique_ptr<RBC::Request>> requests;
        //copy current data (that will be send) into buffer
        copyDataToBuffer(ival);

        //calculate how much data need to be send and received, then start non-blocking sends
        getRecvCount(ival, recv_small, recv_large,
                recv_count_small, recv_count_large);
                
        long long recv_elements = recv_small + recv_large;
        assert(ival.getLocalElements() == recv_elements);
        if (!ival.evenly_distributed)
            ival.data->resize(recv_elements);
        
        sendData(ival, requests, recv_small, recv_large,
                recv_count_small, recv_count_large);
        
        T *data_ptr = ival.data->data();
        //receive data
        while (recv_count_small < recv_small || recv_count_large < recv_large) {
            if (recv_count_small < recv_small) {
                receiveData(ival.comm, requests,
                            data_ptr + ival.local_start + recv_count_small,
                            recv_count_small, recv_small, Constants::EXCHANGE_SMALL, ival.mpi_type);
            }
            if (recv_count_large < recv_large) {
                receiveData(ival.comm, requests,
                            data_ptr + ival.local_start + recv_small + recv_count_large,
                            recv_count_large, recv_large, Constants::EXCHANGE_LARGE, ival.mpi_type);
            }
            TestallVector(requests);
        }
        WaitallVector(requests);
    }
    
    /*
     * Exchange the data with other PEs on both intervals simultaneously
     */
    static void exchangeDataSchizo(QSInterval_SQS<T> &ival_left, QSInterval_SQS<T> &ival_right) {
        long long recv_small_l = 0, recv_large_l = 0, recv_count_small_l = 0, recv_count_large_l = 0;
        long long recv_small_r = 0, recv_large_r = 0, recv_count_small_r = 0, recv_count_large_r = 0;
        std::vector<std::unique_ptr<RBC::Request>> requests;
        
        //copy current data (that will be send) into buffer        
        copyDataToBuffer(ival_left);
        copyDataToBuffer(ival_right);
        
        //calculate how much data need to be send and received, then start non-blocking sends
        getRecvCount(ival_left, recv_small_l, recv_large_l,
                recv_count_small_l, recv_count_large_l);
        getRecvCount(ival_right, recv_small_r, recv_large_r,
                recv_count_small_r, recv_count_large_r);
        sendData(ival_left, requests, recv_small_l, recv_large_l,
                recv_count_small_l, recv_count_large_l);
        sendData(ival_right, requests, recv_small_r, recv_large_r,
                recv_count_small_r, recv_count_large_r);

        T *data_ptr_left = ival_left.data->data();
        T *data_ptr_right = ival_right.data->data();
        //receive data
        while ((recv_count_small_l < recv_small_l) || (recv_count_large_l < recv_large_l)
                || (recv_count_small_r < recv_small_r) || (recv_count_large_r < recv_large_r)) {
            if (recv_count_small_l < recv_small_l) {
                receiveData(ival_left.comm, requests,
                            data_ptr_left + ival_left.local_start + recv_count_small_l,
                            recv_count_small_l, recv_small_l, Constants::EXCHANGE_SMALL, ival_left.mpi_type);
            }
            if (recv_count_large_l < recv_large_l) {
                receiveData(ival_left.comm, requests,
                            data_ptr_left + ival_left.local_start + recv_small_l + recv_count_large_l,
                            recv_count_large_l, recv_large_l, Constants::EXCHANGE_LARGE, ival_left.mpi_type);
            }
            if (recv_count_small_r < recv_small_r) {
                receiveData(ival_right.comm, requests,
                            data_ptr_right + ival_right.local_start + recv_count_small_r,
                            recv_count_small_r, recv_small_r, Constants::EXCHANGE_SMALL, ival_right.mpi_type);
            }
            if (recv_count_large_r < recv_large_r) {
                receiveData(ival_right.comm, requests,
                            data_ptr_right + ival_right.local_start + recv_small_r + recv_count_large_r,
                            recv_count_large_r, recv_large_r, Constants::EXCHANGE_LARGE, ival_right.mpi_type);
            }

            //test all send and receive operations
            TestallVector(requests);
        }
        WaitallVector(requests);
    }  

private:
    
        
    /*
     * Calculate how much small and large data need to be received
     */
    static void getRecvCount(QSInterval_SQS<T> &ival, long long &recv_small, long long &recv_large,
            long long &recv_count_small, long long &recv_count_large) {
        int small_end_PE = ival.getRankFromIndex(ival.missing_first_PE + ival.global_small_elements - 1);
        int large_start_PE = ival.getRankFromIndex(ival.missing_first_PE + ival.global_small_elements);
        int local_elements = ival.getLocalElements();
        
        if (large_start_PE > ival.rank) {
            recv_small = local_elements;
            recv_large = 0;
        } else if (small_end_PE < ival.rank) {
            recv_small = 0;
            recv_large = local_elements;
        } else {
            recv_small = ival.getOffsetFromIndex(ival.missing_first_PE + ival.global_small_elements)
                    - ival.local_start;
            recv_large = local_elements - recv_small;
        }        
    }
    
    /*
     * Calculate how much data need to be send then start non-blocking sends
     */
    static void sendData(QSInterval_SQS<T> &ival,
            std::vector<std::unique_ptr<RBC::Request>> &requests, long long &recv_small, 
            long long &recv_large, long long &recv_count_small, long long &recv_count_large) {
        long long small_start = ival.local_start;
        long long large_start = ival.local_start + ival.local_small_elements;
        long long large_end = ival.local_end;
        long long global_idx_small = ival.missing_first_PE + ival.presum_small;
        long long global_idx_large = ival.missing_first_PE + ival.global_small_elements
            + ival.presum_large;
        T *buffer_small = ival.data->data() + ival.local_start;
        T *buffer_large = ival.data->data() + ival.local_start + recv_small;
        
        // send small elements
        recv_count_small = sendDataRecursive(ival, small_start, 
                large_start, global_idx_small, buffer_small,
                requests, Constants::EXCHANGE_SMALL);
        
        // send large elements
        recv_count_large = sendDataRecursive(ival, large_start, 
                large_end, global_idx_large, buffer_large,
                requests, Constants::EXCHANGE_LARGE);
    }

    /*
     * Returns the number of elements that have been copied locally into the recv_buffer
     */
    static int sendDataRecursive(QSInterval_SQS<T> &ival, long long local_start_idx,
            long long local_end_idx, long long global_start_idx,
            T *recv_buffer, std::vector<std::unique_ptr<RBC::Request>> &requests,
            int tag) {
        // return if no elements need to be send
        if (local_start_idx >= local_end_idx)
            return 0;
        
        int target_rank = ival.getRankFromIndex(global_start_idx);
        long long send_max = ival.getIndexFromRank(target_rank + 1) - global_start_idx;
        long long local_elements = local_end_idx - local_start_idx;
        long long send_count = std::min(send_max, local_elements);
        
        long long copied_local = 0;
        if (target_rank == ival.rank) {
            copied_local += send_count;
            std::memcpy(recv_buffer, ival.buffer + local_start_idx, send_count * sizeof(T));            
        } else {
            requests.push_back(std::unique_ptr<RBC::Request>(new RBC::Request));
            RBC::Isend(ival.buffer + local_start_idx, send_count, ival.mpi_type,
                    target_rank, tag, ival.comm, requests.back().get());
        }
        
        if (local_elements > send_count) {
            // send remaining data
            copied_local += sendDataRecursive(ival, local_start_idx + send_count,
                    local_end_idx, ival.getIndexFromRank(target_rank + 1),
                    recv_buffer + copied_local, requests, tag);
        }
        return copied_local;        
    }
    
     /*
     * Starts a non-blocking receive if data can be received
     */
    static void receiveData(RBC::Comm const &comm, std::vector<std::unique_ptr<RBC::Request>> &requests,
            void *recvbuf, long long &recv_count, long long recv_total, int tag,
            MPI_Datatype mpi_type) {
        if (recv_count < recv_total) {
            int ready;
            MPI_Status status;
            RBC::Iprobe(MPI_ANY_SOURCE, tag, comm, &ready, &status);
            if (ready) {
                int count;
                MPI_Get_count(&status, mpi_type, &count);
//                std::cout << W(recv_total) << W(recv_count) << W(count) << std::endl;
                assert(recv_count + count <= recv_total);
                int source = RBC::get_Rank_from_Status(comm, status);
                requests.push_back(std::unique_ptr<RBC::Request>(new RBC::Request));
                RBC::Irecv(recvbuf, count, mpi_type, source,
                        tag, comm, requests.back().get());
                recv_count += count;
            }
        }
    }


    /*
     * Copy data into the send buffer such that all small (and large) elements
     * are stored consecutively
     */
    static void copyDataToBuffer(QSInterval_SQS<T> &ival) {      
        T *data_ptr = ival.data->data();
        long long copy = ival.bound1 - ival.local_start;
        std::memcpy(ival.buffer + ival.local_start, data_ptr + ival.local_start, copy * sizeof(T));
        
        long long small_right = ival.bound2 - ival.split;
        copy = ival.split - ival.bound1;
        std::memcpy(ival.buffer + ival.bound1 + small_right, data_ptr + ival.bound1, copy * sizeof(T));
        
        copy = ival.bound2 - ival.split;
        std::memcpy(ival.buffer + ival.bound1, data_ptr + ival.split, copy * sizeof(T));
        
        copy = ival.local_end - ival.bound2;
        std::memcpy(ival.buffer + ival.bound2, data_ptr + ival.bound2, copy * sizeof(T));
    }

    /*
     * Call the test function for all requests of the vector
     */
    static void TestallVector(std::vector<std::unique_ptr<RBC::Request>> &requests) {
        for (size_t i = 0; i < requests.size(); i++) {
            int tmp_flag;
            RBC::Test(requests[i].get(), &tmp_flag, MPI_STATUS_IGNORE);
        }
    }

    /*
     * Wait until all operations of the requests of the vector are completed 
     */
    static void WaitallVector(std::vector<std::unique_ptr<RBC::Request>> &requests) {
        int flag = 0;
        while(flag == 0) {
            flag = 1;
            for (size_t i = 0; i < requests.size(); i++) {
                int tmp_flag;
                RBC::Test(requests[i].get(), &tmp_flag, MPI_STATUS_IGNORE);
                if (tmp_flag == 0)
                    flag = 0;
            }
        }
    }
    
};

#endif /* DATAEXCHANGE_HPP */

