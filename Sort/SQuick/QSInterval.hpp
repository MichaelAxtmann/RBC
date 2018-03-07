/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/


#ifndef SQS_QSINTERVAL_HPP
#define SQS_QSINTERVAL_HPP

#include <mpi.h>
#include <iostream>
#include <vector>

#include "../../RangeBasedComm/RBC.hpp"

/*
 * This struct represents a interval of data used in a Quicksort call
 */
template<typename T>
struct QSInterval_SQS {
    QSInterval_SQS() {}
    
    QSInterval_SQS(std::vector<T> *data,  T *buffer, long long split_size, long long extra_elements,
            long long local_start, long long local_end, 
            RBC::Comm comm, long long offset_first_PE, long long offset_last_PE,
            MPI_Datatype mpi_type, int seed, long long min_samples, bool add_pivot,
            bool blocking_priority, bool evenly_distributed = true) :
            data(data), buffer(buffer), split_size(split_size), extra_elements(extra_elements),
            local_start(local_start), local_end(local_end),
            missing_first_PE(offset_first_PE), missing_last_PE(offset_last_PE),
            min_samples(min_samples),
            seed(seed), comm(comm), mpi_type(mpi_type), 
            evenly_distributed(evenly_distributed), add_pivot(add_pivot),
            blocking_priority(blocking_priority) {
        this->seed = seed * 48271 % 2147483647;

        if (MPI_COMM_NULL != comm.GetMpiComm()) {
            RBC::Comm_size(comm, &number_of_PEs);
            RBC::Comm_rank(comm, &rank);
        } else {
            number_of_PEs = -1;
            rank = -1;
        }
        
        start_PE = 0;
        end_PE = number_of_PEs - 1;
        local_elements = local_end - local_start;
    }
    
    int getRankFromIndex(long long global_index) const {
        long long idx = extra_elements * (split_size + 1);
        if (global_index < idx)
            return global_index / (split_size + 1);
        else {
            if (split_size == 0)
                return extra_elements;
            else {
                long long idx_dif = global_index - idx;
                int r = idx_dif / split_size;
                return extra_elements + r;
            }
        }
    }
    
    int getOffsetFromIndex(long long global_index) const {
        long long idx = extra_elements * (split_size + 1);
        if (global_index < idx)
            return global_index % (split_size + 1);
        else {
            long long idx_dif = global_index - idx;
            return idx_dif % split_size;
        }
    }
    
    long long getIndexFromRank(int rank) const {
        if (rank <= extra_elements)
            return rank * (split_size + 1);
        else
            return extra_elements * (split_size + 1) + (rank - extra_elements) * split_size;        
    }
    
    long long getSplitSize() const {
        return getSplitSize(rank);
    }
    
    long long getSplitSize(int rank) const {
        if (rank < extra_elements)
            return split_size + 1;
        else
            return split_size;
    }
    
    long long getLocalElements() const {
        return getLocalElements(rank);
    }
    
    long long getLocalElements(int rank) const {
        int elements = getSplitSize(rank);
        if (rank == 0)
            elements -= missing_first_PE;
        if (rank == number_of_PEs - 1)
            elements -= missing_last_PE;
        return elements;
    }

    std::vector<T> *data;
    T *buffer;
    long long split_size, extra_elements, local_start, local_end, 
            missing_first_PE, missing_last_PE,
            local_elements, presum_small, presum_large, 
            local_small_elements, local_large_elements, 
            global_small_elements, global_large_elements, global_elements, 
            bound1, split, bound2, min_samples;
    int seed, number_of_PEs, rank, start_PE, end_PE;
    RBC::Comm comm;
    MPI_Datatype mpi_type;
    bool evenly_distributed, add_pivot, blocking_priority;
};

#endif /* QSINTERVAL_HPP */

