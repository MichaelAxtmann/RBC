/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SQS_SEQUENTIALSORT_HPP
#define SQS_SEQUENTIALSORT_HPP

#include "../../RangeBasedComm/RBC.hpp"
#include "QSInterval.hpp"
#include "Constants.hpp"
#include <mpi.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#define W(X) #X << "=" << X << ", "

template<typename T>
class SequentialSort_SQS {
    
public:
        
    /* 
     * Sort all local intervals 
     */
    template<class Compare>
    static void sortLocalIntervals(Compare comp, std::vector<QSInterval_SQS<T>> &local_intervals) {
        for (size_t i = 0; i < local_intervals.size(); i++) {
            T *data_ptr = local_intervals[i].data->data();
            std::sort(data_ptr + local_intervals[i].local_start, 
                      data_ptr + local_intervals[i].local_end, comp);
        }
    }
    
    /* 
     * Sort the saved intervals with exactly two PEs
     */
    template<class Compare>
    static long long sortTwoPEIntervals(Compare comp, std::vector<QSInterval_SQS<T>> &two_PE_intervals) {
        if (two_PE_intervals.size() == 2)
            return sortOnTwoPEsSchizo(two_PE_intervals[0], two_PE_intervals[1], comp);
        else if (two_PE_intervals.size() == 1)
            return sortOnTwoPEs(two_PE_intervals[0], comp);
        else
            assert(two_PE_intervals.size() == 0);
        return 0;
    }
    
private:
    /*
     * Sort an interval with only two PEs sequentially to terminate recursion
     */
    template<class Compare>
    static long long sortOnTwoPEs(QSInterval_SQS<T> &ival, Compare comp) {
        RBC::Request requests[2];      
        T *data_ptr = ival.data->data();
        //gather all elements on both PEs
        int partner = (ival.rank + 1) % 2;
        RBC::Isend(data_ptr + ival.local_start, ival.local_elements, ival.mpi_type,
                partner, Constants::TWO_PE, ival.comm, &requests[0]);
        
        int recv_elements = 0, flag = 0;
        MPI_Status status;
        while (recv_elements == 0) {
            RBC::Iprobe(partner, Constants::TWO_PE, ival.comm, &flag, &status);
            if (flag) {
                MPI_Get_count(&status, ival.mpi_type, &recv_elements);
            }                
            int x;
            RBC::Test(&requests[0], &x, MPI_STATUS_IGNORE);
        }
        
        long long total_elements = ival.local_elements + recv_elements;
        T* tmp_buffer = new T[total_elements];
        RBC::Irecv(tmp_buffer, recv_elements, ival.mpi_type, partner, Constants::TWO_PE, ival.comm,
                &requests[1]);
        
        std::memcpy(tmp_buffer + recv_elements, data_ptr + ival.local_start,
                ival.local_elements * sizeof(T));        
        RBC::Waitall(2, &requests[0], MPI_STATUSES_IGNORE);     
        
        partitionAndSort(ival, comp, tmp_buffer, recv_elements);
        
        delete[] tmp_buffer;
        return total_elements;
    }
    
    /*
     * Partition the buffer and sort one partition 
     */
    template<class Compare>
    static void partitionAndSort(QSInterval_SQS<T> &ival, Compare comp,
            T* buffer, int recv_elements) {
        T* nth_element;
        if (ival.rank == 0)
            nth_element = buffer + ival.local_elements;
        else
            nth_element = buffer + recv_elements;

        long long total_elements = ival.local_elements + recv_elements;
        std::nth_element(buffer, nth_element, buffer + total_elements, comp);

        if (ival.rank == 0) {
            std::sort(buffer, nth_element + 1, comp);
        } else {
            std::sort(nth_element, buffer + total_elements, comp);
        }

        T* copy_ptr;
        if (ival.rank == 0)
            copy_ptr = buffer;
        else
            copy_ptr = buffer + recv_elements;
        T *data_ptr = ival.data->data();
        std::memcpy(data_ptr + ival.local_start, copy_ptr,
                ival.local_elements * sizeof (T));

    }

    /*
     * Sort two intervals with two PEs simultaneously
     */
    template<class Compare>
    static long long sortOnTwoPEsSchizo(QSInterval_SQS<T> &ival_1,
                            QSInterval_SQS<T> &ival_2,
                            Compare comp) {       
        RBC::Request requests[4];
        T *data_ptr_1 = ival_1.data->data();
        T *data_ptr_2 = ival_2.data->data();
        
        int partner_1 = (ival_1.rank + 1) % 2;
        int partner_2 = (ival_2.rank + 1) % 2;
        RBC::Isend(data_ptr_1 + ival_1.local_start, ival_1.local_elements, ival_1.mpi_type,
                partner_1, Constants::TWO_PE, ival_1.comm, &requests[0]);
        RBC::Isend(data_ptr_2 + ival_2.local_start, ival_2.local_elements, ival_2.mpi_type,
                partner_2, Constants::TWO_PE, ival_2.comm, &requests[2]);

        int recv_elements_1 = 0, flag_1 = 0, recv_elements_2 = 0, flag_2 = 0;
        MPI_Status status_1, status_2;
        while (recv_elements_1 == 0 || recv_elements_2 == 0) {
            RBC::Iprobe(partner_1, Constants::TWO_PE, ival_1.comm, &flag_1, &status_1);
            if (flag_1)
                MPI_Get_count(&status_1, ival_1.mpi_type, &recv_elements_1);
            RBC::Iprobe(partner_2, Constants::TWO_PE, ival_2.comm, &flag_2, &status_2);
            if (flag_2)
                MPI_Get_count(&status_2, ival_2.mpi_type, &recv_elements_2);
            int x1, x2;
            RBC::Test(&requests[0], &x1, MPI_STATUS_IGNORE);
            RBC::Test(&requests[2], &x2, MPI_STATUS_IGNORE);
        }
                
        long long total_elements_1 = ival_1.local_elements + recv_elements_1;
        long long total_elements_2 = ival_2.local_elements + recv_elements_2;
        T* buffer_1 = new T[total_elements_1];
        T* buffer_2 = new T[total_elements_2];
        
        RBC::Irecv(buffer_1, recv_elements_1, ival_1.mpi_type, partner_1, Constants::TWO_PE, ival_1.comm,
                &requests[1]);
        RBC::Irecv(buffer_2, recv_elements_2, ival_2.mpi_type, partner_2, Constants::TWO_PE, ival_2.comm,
                &requests[3]);
        
        std::memcpy(buffer_1 + recv_elements_1, data_ptr_1 + ival_1.local_start,
                ival_1.local_elements * sizeof(T));        
        std::memcpy(buffer_2 + recv_elements_2, data_ptr_2 + ival_2.local_start,
                ival_2.local_elements * sizeof(T)); 
        
        RBC::Waitall(4, &requests[0], MPI_STATUSES_IGNORE);
        
        //partiotion data and sort one partition
        partitionAndSort(ival_1, comp, buffer_1, recv_elements_1);
        partitionAndSort(ival_2, comp, buffer_2, recv_elements_2);
        
        delete[] buffer_1;
        delete[] buffer_2;
        
        return total_elements_1 + total_elements_2;
    }

};
    
#endif /* SEQUENTIALSORT_HPP */

