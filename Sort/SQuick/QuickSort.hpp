/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef QUICKSORT_HPP
#define QUICKSORT_HPP

#include <mpi.h>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>

#include "../SortingDatatype.hpp"
#include "../TbSplitter.hpp"
#include "QSInterval.hpp"
#include "PivotSelection.hpp"
#include "../Constants.hpp"
#include "DataExchange.hpp"
#include "SequentialSort.hpp"
#include "../../RangeBasedComm/RBC.hpp"

/*
 * This class represents the Quicksort algorithm
 */
template<typename T>
class QuickSort {
public:

    /**
     * Constructor
     * @param seed Seed for RNG, has to be the same for all PEs
     * @param min_samples Minimal number of samples for the pivot selection
     * @param barriers Use barriers to measure the running time of the algorithm phases 
     * @param split_MPI_comm If true, split communicators using MPI_Comm_create/MPI_Comm_split
     *      else use the RBC split operations
     * @param use_MPI_collectives If true, use the collective operations provided by MPI whenever possible,
     *      else always use the collective operations of RBC
     * @param add_pivot If true, use k1+k2+k3 as the number of samples,
     *      else use max(k1,k2,k3) 
     */
    QuickSort(int seed, long long min_samples = 64, bool barriers = false, bool split_MPI_comm = false,
            bool use_MPI_collectives = false, bool add_pivot = false)
    : seed(seed), barriers(barriers),
    split_MPI_comm(split_MPI_comm), use_MPI_collectives(use_MPI_collectives), add_pivot(add_pivot),
    min_samples(min_samples){
        mpi_type = SortingDatatype<T>::getMPIDatatype();
    }

    ~QuickSort() {
    }
       
    /**
     * Sorts the input data
     * @param mpi_comm MPI commuicator (all ranks have to call the function)
     * @param data_vec Vector that contains the input data
     * @param global_elements The total number of elements on all PEs, set to -1 if unknown
     */
    void sort(MPI_Comm mpi_comm, std::vector<T> &data_vec, long long global_elements = -1) {
        sort(mpi_comm, data_vec, global_elements, std::less<T>());
    }
    
    /**
     * Sorts the input data with a custom compare operator
     * @param mpi_comm MPI commuicator (all ranks have to call the function)
     * @param data_vec Vector that contains the input data
     * @param global_elements The total number of elements on all PEs, set to -1 if unknown
     * @param comp The compare operator
     */
    template<class Compare>
    void sort(MPI_Comm mpi_comm,  std::vector<T> &data, long long global_elements, Compare comp) {
        RBC::Comm comm;
        RBC::Create_Comm_from_MPI(mpi_comm, &comm, use_MPI_collectives, split_MPI_comm);
        MPI_Barrier(mpi_comm);
        sort_range(comm, data, global_elements, comp);
    }
    
    /**
     * Sort data on an RBC communicator
     * @param mpi_comm MPI commuicator (all ranks have to call the function)
     * @param data_vec Vector that contains the input data
     * @param global_elements The total number of elements on all PEs, set to -1 if unknown
     * @param comp The compare operator
     */
    template<class Compare>
    void sort_range(RBC::Comm comm, std::vector<T> &data, 
            long long global_elements, Compare comp) {
        assert(comm.GetMpiComm() != MPI_COMM_NULL);
        
        double total_start = getTime();
        parent_comm = comm.GetMpiComm();
        this->data = &data;
        
        int size, rank;
        RBC::Comm_size(comm, &size);
        RBC::Comm_rank(comm, &rank);
	generator.seed(seed);
        sample_generator.seed(seed + rank);
        
        QSInterval_SQS<T> ival;
        assert(global_elements >= -1);
        if (global_elements == -1) {
            buffer = nullptr;
            // split_size and extra_elements will be assigned in calculateExchange
            // local_elements and global_end will be changed after dataExchange
            ival = QSInterval_SQS<T>(&data, buffer, -1, -1, 0, 
                    data.size(), comm, 0, 0, mpi_type, seed, min_samples, add_pivot, true, false);            
        } else {
            long long split_size = global_elements / size;
            long long extra_elements = global_elements % size;
            buffer = new T[data.size()];
            ival = QSInterval_SQS<T>(&data, buffer, split_size, extra_elements, 0, 
                    data.size(), comm, 0, 0, mpi_type, seed, min_samples, add_pivot, true);
        }

        /* Recursive */
        quickSort(ival, comp);

        delete[] buffer;
        
        /* Base Cases */
        double start, end;
        if (barriers)
            RBC::Barrier(comm);
        start = getTime();
        sortTwoPEIntervals(comp);
        end = getTime();
        t_sort_two = end - start;
                
        start = getTime();
        sortLocalIntervals(comp);
        end = getTime();
        t_sort_local = end - start;

        double total_end = getTime();
        t_runtime = (total_end - total_start);
    }
    
    /** 
     * @return The maximal depth of recursion  
     */
    int getDepth() {
        return depth;
    }
    
    /**
     * Get timers and their names
     * @param timer_names Vector containing the names of the timers
     * @param max_timers Vector containing the maximal timer value across all PEs
     * @param comm RBC communicator
     */
    void getTimers(std::vector<std::string> &timer_names,
	    std::vector<double> &max_timers, RBC::Comm comm) {
        std::vector<double> timers;
        int size, rank;
        RBC::Comm_size(comm, &size);     
        RBC::Comm_rank(comm, &rank);
        if (barriers) {
            timers.push_back(t_pivot);
            timer_names.push_back("pivot");
            timers.push_back(t_partition);
            timer_names.push_back("partition");
            timers.push_back(t_calculate);
            timer_names.push_back("calculate");
            timers.push_back(t_exchange);
            timer_names.push_back("exchange");
//            timers.push_back(t_sort_two);
//            timer_names.push_back("sort_two");
//            timers.push_back(t_sort_local);
//            timer_names.push_back("sort_local");
            timers.push_back(t_sort_local + t_sort_two);
            timer_names.push_back("base_cases");
            double sum = 0.0;
            for (size_t i = 0; i < timers.size(); i++)
                sum += timers[i];
            timers.push_back(sum);
            timer_names.push_back("sum");
        }
//        timers.push_back(bc1_elements);
//        timer_names.push_back("BaseCase1_elements");
//        timers.push_back(bc2_elements);
//        timer_names.push_back("BaseCase2_elements");
//        timers.push_back(t_runtime);
//        timer_names.push_back("runtime");
        timers.push_back(depth);
        timer_names.push_back("depth");
        timers.push_back(t_create_comms);
        timer_names.push_back("create_comms");

        for (size_t i = 0; i < timers.size(); i++) {
            double time = 0.0;
            RBC::Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            max_timers.push_back(time);
//            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
//            min_timers.push_back(time);
//            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
//            avg_timers.push_back(time / size);
        }
        
//        double runtimes[size];
//        MPI_Allgather(&t_runtime, 1, MPI_DOUBLE, runtimes, 1, MPI_DOUBLE, comm);
//        double t_max = 0.0;
//        int rank_max = 0;
//        for (int i = 0; i < size; i++) {
//            if (runtimes[i] > t_max) {
//                t_max = runtimes[i];
//                rank_max = i;
//            }
//        }        
//        if (rank_max == 0) {
//            max_timers.push_back(bc1_elements);
//            timer_names.push_back("slow_BaseCase1_elements");
//            max_timers.push_back(bc2_elements);
//            timer_names.push_back("slow_BaseCase2_elements");
//            max_timers.push_back(depth);
//            timer_names.push_back("slow_depth");
//        } else {
//            if (rank == rank_max) {
//                double slowest_timers[3] = {bc1_elements,
//                    bc2_elements, static_cast<double>(depth)};
//                MPI_Send(slowest_timers, 5, MPI_DOUBLE, 0, 0, comm);
//            }
//            if (rank == 0) {
//                double slowest_timers[5];
//                MPI_Recv(slowest_timers, 5, MPI_DOUBLE, rank_max, 0, comm,
//                        MPI_STATUS_IGNORE);   
//                max_timers.push_back(slowest_timers[0]);
//                timer_names.push_back("slow_BaseCase1_elements");
//                max_timers.push_back(slowest_timers[1]);
//                timer_names.push_back("slow_BaseCase2_elements");
//                max_timers.push_back(slowest_timers[2]);
//                timer_names.push_back("slow_depth");
//            }
//        }
    }

private:
    
    /*
     * Execute the Quicksort algorithm on the given QSInterval
     */
    template<class Compare>
    void quickSort(QSInterval_SQS<T> &ival, Compare comp) {
        depth++;
        
        //Check if recursion should be ended 
        if (isBaseCase(ival))
            return;        
        
        /* Pivot Selection */
        T pivot;
        long long split_idx;
        double t_start, t_end;
        t_start = startTime(ival.comm);
        bool zero_global_elements;
        getPivot(ival, pivot, split_idx, comp, zero_global_elements);
        t_end = getTime();
        t_pivot += (t_end - t_start);
        
        if (zero_global_elements)
            return;
        
        /* Partitioning */
        t_start = startTime(ival.comm);
        long long bound1, bound2;        
        partitionData(ival, pivot, split_idx, &bound1, &bound2, comp);
        t_end = getTime();
        t_partition += (t_end - t_start);
        
        /* Calculate how data has to be exchanged */
        t_start = startTime(ival.comm);
        calculateExchangeData(ival, bound1, split_idx, bound2);
        t_end = getTime();
        t_calculate += (t_end - t_start);        
	
        /* Exchange data */
        t_start = startTime(ival.comm);
        exchangeData(ival);
        t_end = getTime();
        t_exchange += (t_end - t_start);
        t_vec_exchange.push_back(t_end - t_start);

        /* Create QSIntervals for the next recursion level */
        long long mid, offset;
        int left_size;
        bool shizophrenic;
        calculateSplit(ival, left_size, offset, shizophrenic, mid);
        
        RBC::Comm comm_left, comm_right;
        if (use_MPI_collectives)
            t_start = startTime_barrier(ival.comm);
        else
            t_start = startTime(ival.comm);
        createNewCommunicators(ival, left_size, shizophrenic, &comm_left, &comm_right);
        t_end = getTime();
        t_create_comms += (t_end - t_start);
        
        QSInterval_SQS<T> ival_left, ival_right;
        createIntervals(ival, offset, left_size, shizophrenic,
                mid, comm_left, comm_right, ival_left, ival_right);
        
        bool sort_left = false, sort_right = false;
        if (ival.rank <= ival_left.end_PE)
            sort_left = true;
        if (ival.rank >= ival_left.end_PE) {
            if (ival.rank > ival_left.end_PE || shizophrenic)
                sort_right = true;
        }

        /* Call recursively */
        if (sort_left && sort_right) {
            shizophrenicQuickSort(ival_left, ival_right, comp);
        } else if (sort_left) {
            quickSort(ival_left, comp);
        } else if (sort_right) {
            quickSort(ival_right, comp);
        }
    }
    
    /*
     * Execute the Quicksort algorithm as shizophrenic PE
     */
    template<class Compare>
    void shizophrenicQuickSort(QSInterval_SQS<T> &ival_left, QSInterval_SQS<T> &ival_right,
            Compare comp) {
        depth++;
        
        //Check if recursion should be ended
        if (isBaseCase(ival_left)) {
            quickSort(ival_right, comp);
            return;
        }
        if (isBaseCase(ival_right)) {
            quickSort(ival_left, comp);
            return;
        }
        
        /* Pivot Selection */
        T pivot_left, pivot_right;
        long long split_idx_left, split_idx_right;
        double t_start, t_end;        
        t_start = startTimeShizo(ival_left.comm, ival_right.comm);
        getPivotShizo(ival_left, ival_right, pivot_left, pivot_right,
                       split_idx_left, split_idx_right, comp);
        t_end = getTime();
        t_pivot += (t_end - t_start);
        
        /* Partitioning */        
        t_start = startTimeShizo(ival_left.comm, ival_right.comm);        
        long long bound1_left, bound2_left, bound1_right, bound2_right;
        partitionData(ival_left, pivot_left, split_idx_left, &bound1_left, 
                      &bound2_left, comp);          
        partitionData(ival_right, pivot_right, split_idx_right, &bound1_right, 
                      &bound2_right, comp);  
        t_end = getTime();
        t_partition += (t_end - t_start);
                
        /* Calculate how data has to be exchanged */
        t_start = startTimeShizo(ival_left.comm, ival_right.comm);
        calculateExchangeDataShizo(ival_left, ival_right, bound1_left, split_idx_left,
                bound2_left, bound1_right, split_idx_right, bound2_right);
        t_end = getTime();
        t_calculate += (t_end - t_start);
        
        /* Exchange Data */
        t_start = startTimeShizo(ival_left.comm, ival_right.comm);
        exchangeDataShizo(ival_left, ival_right);
        t_end = getTime();
        t_exchange += (t_end - t_start);
        t_vec_exchange.push_back(t_end - t_start);
        
        /* Create QSIntervals for the next recursion level */
        long long mid_left, mid_right, offset_left, offset_right;
        int left_size_left, left_size_right;
        bool shizophrenic_left, shizophrenic_right;
        calculateSplit(ival_left, left_size_left, offset_left, shizophrenic_left, mid_left);
        calculateSplit(ival_right, left_size_right, offset_right, shizophrenic_right, mid_right);
        RBC::Comm left1, right1, left2, right2;
        if (use_MPI_collectives)
            t_start = startTimeShizo_barrier(ival_left.comm, ival_right.comm);
        else
            t_start = startTimeShizo(ival_left.comm, ival_right.comm);
        createNewCommunicatorsShizo(ival_left, ival_right, left_size_left, shizophrenic_left,
                left_size_right, shizophrenic_right, &left1, &right1, &left2, &right2);
        t_end = getTime();
        t_create_comms += (t_end - t_start);

        QSInterval_SQS<T> ival_left_left, ival_right_left,
                ival_left_right, ival_right_right;
        createIntervals(ival_left, offset_left, left_size_left,
                shizophrenic_left, mid_left, left1, right1, 
                ival_left_left, ival_right_left);
        createIntervals(ival_right, offset_right, left_size_right,
                shizophrenic_right, mid_right, left2, right2, 
                ival_left_right, ival_right_right);
        
        bool sort_left = false, sort_right = false;
        QSInterval_SQS<T> *left_i, *right_i;
        //Calculate new left interval and if it need to be sorted
        if (ival_right_left.number_of_PEs == 1) {
            addLocalInterval(ival_right_left);
            left_i = &ival_left_left;
            if (ival_left_left.number_of_PEs == ival_left.number_of_PEs)
                sort_left = true;
        } else {
            left_i = &ival_right_left;
            sort_left = true;
        }
        //Calculate new right interval and if it need to be sorted   
        if (ival_left_right.number_of_PEs == 1) {
            addLocalInterval(ival_left_right);
            right_i = &ival_right_right;
            if (ival_right_right.number_of_PEs == ival_right.number_of_PEs)
                sort_right = true;
        } else {
            right_i = &ival_left_right;
            sort_right = true;
        }

        /* Call recursively */
        if (sort_left && sort_right) {
            shizophrenicQuickSort(*left_i, *right_i, comp);
        } else if (sort_left) {
            quickSort(*left_i, comp);
        } else if (sort_right) {
            quickSort(*right_i, comp);
        }
    }    
    
    /**
     * Check for base cases
     * @return true if base case, false if no base case
     */
    bool isBaseCase(QSInterval_SQS<T> &ival) {
        if (ival.rank == -1)
            return true;   
        if (ival.number_of_PEs == 2) {
            addTwoPEInterval(ival);
            return true;
        }
        if (ival.number_of_PEs == 1) {
            addLocalInterval(ival);
            return true;
        }
        return false;
    }
    
    /*
     * Returns the current time
     */
    double getTime() {
        return MPI_Wtime();
    }
    
    double startTime(RBC::Comm &comm) {
        if (!barriers)
            return getTime();
        RBC::Request req;
        RBC::Ibarrier(comm, &req);
        RBC::Wait(&req, MPI_STATUS_IGNORE);        
        return getTime();
    }
    
    double startTime_barrier(RBC::Comm &comm) {
        RBC::Request req;
        RBC::Ibarrier(comm, &req);
        RBC::Wait(&req, MPI_STATUS_IGNORE);        
        return getTime();
    }

    double startTimeShizo(RBC::Comm &left_comm, RBC::Comm &right_comm) {
        if (!barriers)
            return getTime();        
        RBC::Request req[2];
        RBC::Ibarrier(left_comm, &req[0]);
        RBC::Ibarrier(right_comm, &req[1]);
        RBC::Waitall(2, req, MPI_STATUS_IGNORE);        
        return getTime();
    }
    
    double startTimeShizo_barrier(RBC::Comm &left_comm, RBC::Comm &right_comm) {       
        RBC::Request req[2];
        RBC::Ibarrier(left_comm, &req[0]);
        RBC::Ibarrier(right_comm, &req[1]);
        RBC::Waitall(2, req, MPI_STATUS_IGNORE);        
        return getTime();
    }
    
    /*
     * Select an element from the interval as the pivot
     */
    template<class Compare>
    void getPivot(QSInterval_SQS<T> const &ival, T &pivot, long long &split_idx,
            Compare comp, bool &zero_global_elements) {
        return PivotSelection_SQS<T>::getPivot(ival, pivot, split_idx, comp, 
                generator, sample_generator, zero_global_elements);
    }
    
    /*
     * Select an element as the pivot from both intervals
     */
    template<class Compare>
    void getPivotShizo(QSInterval_SQS<T> const &ival_left,
                        QSInterval_SQS<T> const &ival_right, T &pivot_left,
                        T &pivot_right, long long &split_idx_left, long long &split_idx_right,
                        Compare comp) {
        PivotSelection_SQS<T>::getPivotShizo(ival_left, ival_right, pivot_left, pivot_right,
                split_idx_left, split_idx_right, comp, generator, sample_generator);
    }
    
    /*
     * Partitions the data separatly for the elements with index smaller less_idx 
     * and the elements with index larger less_idx
     * Returns the indexes of the first element of the right partitions
     * @param index1 First element of the first partition with large elements
     * @param index2 First element of the second partition with large elements
     */
    template<class Compare>
    void partitionData(QSInterval_SQS<T> const &ival, T pivot, long long less_idx,
                       long long *index1, long long *index2, Compare comp) {
        long long start1 = ival.local_start, end1 = less_idx,
                start2 = less_idx, end2 = ival.local_end;
        *index1 = partitionSequence(data->data(), pivot, start1, end1, true, comp);
        *index2 = partitionSequence(data->data(), pivot, start2, end2, false, comp);
    }

    /**
     * Partition the data with index [start, end)
     * @param less_equal If true, compare to the pivot with <=, else compare with >
     * @return Index of the first large element
     */
    template<class Compare>
    long long partitionSequence(T *data_ptr, T pivot, long long start, long long end,
            bool less_equal, Compare comp) {
        T* bound;
	if (less_equal) {
            bound = std::partition(data_ptr + start, data_ptr + end,
                                   [pivot, comp](T x){return !comp(pivot, x)/*x <= pivot*/;});
	} else {
            bound = std::partition(data_ptr + start, data_ptr + end,
                                   [pivot, comp](T x){return comp(x, pivot);});
	}
	return bound - data_ptr;
    }

    /*
     * Prefix sum of small/large elements and broadcast of global small/large elements
     */
    void calculateExchangeData(QSInterval_SQS<T> &ival, long long bound1,
            long long split, long long bound2) {
        elementsCalculation(ival, bound1, split, bound2);
        long long in[2] = {ival.local_small_elements, ival.local_large_elements};
        long long presum[2], global[2];
        RBC::Request request;
        RBC::IscanAndBcast(&in[0], &presum[0], &global[0], 2, MPI_LONG_LONG,
                MPI_SUM, ival.comm, &request, Constants::CALC_EXCH);
        RBC::Wait(&request, MPI_STATUS_IGNORE);

        assignPresum(ival, presum, global);
        
        if (!ival.evenly_distributed) {
            ival.split_size = ival.global_elements / ival.number_of_PEs;
            ival.extra_elements = ival.global_elements % ival.number_of_PEs;
            long long buf_size = std::max(ival.local_elements, ival.getLocalElements());
            buffer = new T[buf_size];
            ival.buffer = buffer;
        }
    }

    void elementsCalculation(QSInterval_SQS<T> &ival, long long bound1,
            long long split, long long bound2) {
        ival.bound1 = bound1;
        ival.bound2 = bound2;
        ival.split = split;
        ival.local_small_elements = (bound1 - ival.local_start) + (bound2 - split);
        ival.local_large_elements = ival.local_elements - ival.local_small_elements;

    }

    void assignPresum(QSInterval_SQS<T> &ival,
            long long presum[2], long long global[2]) {
        ival.presum_small = presum[0] - ival.local_small_elements;
        ival.presum_large = presum[1] - ival.local_large_elements;
        ival.global_small_elements = global[0];
        ival.global_large_elements = global[1];
        ival.global_elements = ival.global_small_elements + ival.global_large_elements;
    }
    
    
    void calculateExchangeDataShizo(QSInterval_SQS<T> &ival_left,
            QSInterval_SQS<T> &ival_right, long long  bound1_left, long long split_left, 
            long long bound2_left, long long bound1_right, long long split_right, long long bound2_right) { 
        elementsCalculation(ival_left, bound1_left, split_left, bound2_left);
        elementsCalculation(ival_right, bound1_right, split_right, bound2_right);
        
        long long in_left[2] = {ival_left.local_small_elements, ival_left.local_large_elements};
        long long in_right[2] = {ival_right.local_small_elements, ival_right.local_large_elements};
        long long presum_left[2], presum_right[2], global_left[2], global_right[2];
        RBC::Request requests[2];        
        RBC::IscanAndBcast(&in_left[0], &presum_left[0], &global_left[0], 2, MPI_LONG_LONG, 
                MPI_SUM, ival_left.comm, &requests[1], Constants::CALC_EXCH);
        RBC::IscanAndBcast(&in_right[0], &presum_right[0], &global_right[0], 2, MPI_LONG_LONG, 
                MPI_SUM, ival_right.comm, &requests[0], Constants::CALC_EXCH);
        RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);
        
        assignPresum(ival_left, presum_left, global_left);
        assignPresum(ival_right, presum_right, global_right);
    }

    /*
     * Exchange the data with other PEs
     */
    void exchangeData(QSInterval_SQS<T> &ival) {
        DataExchange_SQS<T>::exchangeData(ival);
                
        if (!ival.evenly_distributed) {
            ival.local_elements = ival.getLocalElements();
            ival.local_end = ival.local_start + ival.local_elements;
        }
    }
    
    /*
     * Exchange the data with other PEs on both intervals simultaneously
     */
    void exchangeDataShizo(QSInterval_SQS<T> &left, QSInterval_SQS<T> &right) {
        DataExchange_SQS<T>::exchangeDataShizo(left, right);
    }  

    /*
     * Calculate how the PEs should be split into two groups
     */
    void calculateSplit(QSInterval_SQS<T> &ival, int &left_size, long long &offset,
            bool &shizophrenic, long long &mid) {
        assert(ival.global_small_elements != 0);
        long long last_small_element = ival.missing_first_PE + ival.global_small_elements - 1;
        
        left_size = ival.getRankFromIndex(last_small_element) + 1;
        offset = ival.getOffsetFromIndex(last_small_element);
                       
        if (offset + 1 == ival.getSplitSize(left_size - 1))
            shizophrenic = false;
        else
            shizophrenic = true;
        
        if (ival.rank < left_size - 1)
            mid = ival.local_end;
        else if (ival.rank > left_size - 1)
            mid = ival.local_start;
        else {
            mid = offset + 1;
        }        
    }
    
    /*
     * Splits the communicator into two new, left and right
     */
    void createNewCommunicators(QSInterval_SQS<T> &ival, long long left_size,
            bool shizophrenic, RBC::Comm *left, RBC::Comm *right) {
        int size;
        RBC::Comm_size(ival.comm, &size);
        int left_end = left_size - 1;
        int right_start = left_size;
        if (shizophrenic)
            right_start--;
        int right_end = std::min(static_cast<long long>(size - 1), ival.global_elements - 1);
        right_end = std::max(right_start, right_end);
        RBC::Split_Comm(ival.comm, 0, left_end, right_start, right_end,
                left, right);
        RBC::Comm_free(ival.comm);
    }
    
    void createNewCommunicatorsShizo(QSInterval_SQS<T> &ival_left,
            QSInterval_SQS<T> &ival_right, long long left_size_left,
            bool shizophrenic_left, long long left_size_right, bool shizophrenic_right,
            RBC::Comm *left_1, RBC::Comm *right_1,
            RBC::Comm *left_2, RBC::Comm *right_2) {
        if (ival_left.blocking_priority) {
            createNewCommunicators(ival_left, left_size_left, shizophrenic_left, left_1, right_1);
            createNewCommunicators(ival_right, left_size_right, shizophrenic_right, left_2, right_2);
        } else {
            createNewCommunicators(ival_right, left_size_right, shizophrenic_right, left_2, right_2);
            createNewCommunicators(ival_left, left_size_left, shizophrenic_left, left_1, right_1);
        }
    }
   
    /*
     * Create QSIntervals for the next recursion level
     */
    void createIntervals(QSInterval_SQS<T> &ival, long long offset, int left_size,
            bool shizophrenic,
            long long mid, RBC::Comm &comm_left, RBC::Comm &comm_right, 
            QSInterval_SQS<T> &ival_left,
            QSInterval_SQS<T> &ival_right) {
        long long missing_last_left, missing_first_right;
        if (shizophrenic) {
            missing_last_left = ival.getSplitSize(left_size - 1) - (offset + 1);
            missing_first_right = offset + 1; 
        } else {
            missing_last_left = 0;
            missing_first_right = 0;
        }
        
        long long start = ival.local_start;
        long long end = ival.local_end;
        long long extra_elements_left, extra_elements_right, 
                split_size_left, split_size_right;
        if (left_size <= ival.extra_elements) {
            extra_elements_left = 0;
            split_size_left = ival.split_size + 1;
            extra_elements_right = ival.extra_elements - left_size;
            if (shizophrenic)
                extra_elements_right++;
        } else {
            extra_elements_left = ival.extra_elements;
            split_size_left = ival.split_size;
            extra_elements_right = 0;
        }
        split_size_right = ival.split_size;
                
        ival_left = QSInterval_SQS<T>(ival.data, ival.buffer, split_size_left, extra_elements_left,
                start, mid, comm_left, ival.missing_first_PE, missing_last_left,
                mpi_type, ival.seed, ival.min_samples, ival.add_pivot, true);        
        ival_right = QSInterval_SQS<T>(ival.data, ival.buffer, split_size_right, extra_elements_right,
                mid, end, comm_right, missing_first_right, ival.missing_last_PE,
                mpi_type, ival.seed + 1, ival.min_samples, ival.add_pivot, false);
        
    }
    
    /* 
     * Add an interval with two PEs (base case)
     */
    void addTwoPEInterval(QSInterval_SQS<T> const &ival) {
        two_PE_intervals.push_back(ival);
    }
    
    /* 
     * Add an interval that can be sorted locally (base case)
     */
    void addLocalInterval(QSInterval_SQS<T> &ival) {
        local_intervals.push_back(ival);
    }

    /* 
     * Sort the saved intervals with exactly two PEs
     */
    template<class Compare>
    void sortTwoPEIntervals(Compare comp) { 
        bc2_elements = SequentialSort_SQS<T>::sortTwoPEIntervals(comp, two_PE_intervals);
        for (size_t i = 0; i < two_PE_intervals.size(); i++)
            RBC::Comm_free(two_PE_intervals[i].comm);
    }
    /* 
     * Sort all local intervals 
     */
    template<class Compare>
    void sortLocalIntervals(Compare comp) {
        SequentialSort_SQS<T>::sortLocalIntervals(comp, local_intervals);
        bc1_elements = 0.0;
        for (size_t i = 0; i < local_intervals.size(); i++) {
            bc1_elements += local_intervals[i].local_elements;
            RBC::Comm_free(local_intervals[i].comm);
        }
    }

    int depth = 0, seed;
    double t_pivot = 0.0, t_calculate = 0.0, t_exchange = 0.0, t_partition = 0.0,
            t_sort_two = 0.0, t_sort_local = 0.0, t_create_comms = 0.0, t_runtime,
            bc1_elements, bc2_elements;
    std::vector<double> t_vec_exchange, exchange_times{0.0, 0.0, 0.0, 0.0};
    T *buffer;
    std::vector<T> *data;
    MPI_Datatype mpi_type;
    MPI_Comm parent_comm;
    std::mt19937_64 generator, sample_generator;
    std::vector<QSInterval_SQS<T>> local_intervals, two_PE_intervals;
    bool barriers, split_MPI_comm, use_MPI_collectives, add_pivot;
    long long min_samples;
};

#endif // QUICKSORT_HPP
