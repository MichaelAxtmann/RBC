/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef QUICKSORT_HPP
#define QUICKSORT_HPP

#include <mpi.h>
#include "SortingDatatype.hpp"
#include "TbSplitter.hpp"
#include "QSInterval.hpp"
#include "PivotSelection.hpp"
#include "Constants.hpp"
#include "DataExchange.hpp"
#include "SequentialSort.hpp"
#include "../../RangeBasedComm/RBC.hpp"
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>

//Macros
#define V(X) std::cout << #X << "=" << X << endl
#define W(X) #X << "=" << X << ", "

/*
 * This class represents the Quicksort algorithm
 */
template<typename T>
class QuickSort {
public:

    /**
     * Create a sort algorithm configuration
     * @param seed A seed for a random number generator. 
     *             Has to be the same on all ranks that execute the sorting algorithm
     * @param min_samples (optional) The minimal number of samples for the pivot selection
     * @param add_pivot (optional) Smooth transition for number of samples in pivot selection
     */
    QuickSort(int seed, long long min_samples = 64, bool add_pivot = false)
    : seed(seed)
    , add_pivot(add_pivot)
    , min_samples(min_samples) {
        mpi_type = SortingDatatype<T>::getMPIDatatype();
    }

    ~QuickSort() {}
   
    /**
     * Sort the data
     * @param mpi_comm MPI communicator on which the sorting is executed
     * @param data_vec The vector containing the data that have to be sorted
     * @param global_elements The number of elements on all ranks in the communicator.
     *                        Set to -1 if it is unknown
     */
    void sort(MPI_Comm mpi_comm, std::vector<T> &data, long long global_elements = -1) {
        sort(mpi_comm, data, global_elements, std::less<T>());
    }
    
    /**
     * Sort the data with a custom compare operator
     * @param mpi_comm MPI communicator on which the sorting is executed
     * @param data The vector containing the data that have to be sorted
     * @param global_elements The number of elements on all ranks in the communicator.
     *                        Set to -1 if it is unknown
     * @param comp A compare operator 
     */
    template<class Compare>
    void sort(MPI_Comm mpi_comm, std::vector<T> &data, long long global_elements, Compare comp) {
        RBC::Comm comm;
        RBC::Create_Comm_from_MPI(mpi_comm, &comm, true, true);
        MPI_Barrier(mpi_comm);
        sort_rbc(comm, data, global_elements, comp);
    }
    
    /**
     * Sort on a RBC communicator
     * @param comm RBC communicator on which the sorting is executed
     * @param data The vector containing the data that have to be sorted
     * @param global_elements The number of elements on all ranks in the communicator.
     *                        Set to -1 if it is unknown
     * @param comp A compare operator 
     */
    template<class Compare>
    void sort_rbc(RBC::Comm comm, std::vector<T> &data, 
            long long global_elements, Compare comp) {
        assert(comm.GetMpiComm() != MPI_COMM_NULL);
        
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

        quickSort(ival, comp);

        delete[] buffer;
        
        sortTwoPEIntervals(comp);
        sortLocalIntervals(comp);
    }
    
    /*
     * Returns the depth of the recursion that was reached 
     */
    int getDepth() {
        return depth;
    }

private:
    
    /*
     * Execute the Quicksort algorithm on the given QSInterval
     */
    template<class Compare>
    void quickSort(QSInterval_SQS<T> &ival, Compare comp) {
        //Check if recursion should be ended 
        if (isBaseCase(ival))
            return;
        
        depth++;
        
        T pivot;
        long long split_idx;
        bool zero_global_elements;
        getPivot(ival, pivot, split_idx, comp, zero_global_elements);
        
        if (zero_global_elements)
            return;
        
        long long bound1, bound2;        
        partitionData(ival, pivot, split_idx, &bound1, &bound2, comp);
        
        calculateExchangeData(ival, bound1, split_idx, bound2);

        exchangeData(ival);

        long long mid, offset;
        int left_size;
        bool schizophrenic;
        calculateSplit(ival, left_size, offset, schizophrenic, mid);
        
        RBC::Comm comm_left, comm_right;
        createNewCommunicators(ival, left_size, schizophrenic, &comm_left, &comm_right);
        
        QSInterval_SQS<T> ival_left, ival_right;
        createIntervals(ival, offset, left_size, schizophrenic,
                mid, comm_left, comm_right, ival_left, ival_right);
        
        bool sort_left = false, sort_right = false;
        if (ival.rank <= ival_left.end_PE)
            sort_left = true;
        if (ival.rank >= ival_left.end_PE) {
            if (ival.rank > ival_left.end_PE || schizophrenic)
                sort_right = true;
        }
//        std::cout << W(depth) << W(comm_start) << W(ival.rank) << "Recursive" << std::endl;
        
        if (sort_left && sort_right) {
            schizophrenicQuickSort(ival_left, ival_right, comp);
        } else if (sort_left) {
            quickSort(ival_left, comp);
        } else if (sort_right) {
            quickSort(ival_right, comp);
        }
    }
    
    /*
     * Execute the Quicksort algorithm as schizophrenic PE
     */
    template<class Compare>
    void schizophrenicQuickSort(QSInterval_SQS<T> &ival_left, QSInterval_SQS<T> &ival_right,
            Compare comp) {
        //Check if recursion should be ended
        if (isBaseCase(ival_left)) {
            quickSort(ival_right, comp);
            return;
        }
        if (isBaseCase(ival_right)) {
            quickSort(ival_left, comp);
            return;
        }
        
        depth++;

        T pivot_left, pivot_right;
        long long split_idx_left, split_idx_right;
        
        getPivotSchizo(ival_left, ival_right, pivot_left, pivot_right,
                       split_idx_left, split_idx_right, comp);
        
        long long bound1_left, bound2_left, bound1_right, bound2_right;
        partitionData(ival_left, pivot_left, split_idx_left, &bound1_left, 
                      &bound2_left, comp);          
        partitionData(ival_right, pivot_right, split_idx_right, &bound1_right, 
                      &bound2_right, comp);  

        calculateExchangeDataSchizo(ival_left, ival_right, bound1_left, split_idx_left,
                bound2_left, bound1_right, split_idx_right, bound2_right);

        exchangeDataSchizo(ival_left, ival_right);
        
        long long mid_left, mid_right, offset_left, offset_right;
        int left_size_left, left_size_right;
        bool schizophrenic_left, schizophrenic_right;
        calculateSplit(ival_left, left_size_left, offset_left, schizophrenic_left, mid_left);
        calculateSplit(ival_right, left_size_right, offset_right, schizophrenic_right, mid_right);

        RBC::Comm left1, right1, left2, right2;

        createNewCommunicatorsSchizo(ival_left, ival_right, left_size_left, schizophrenic_left,
                left_size_right, schizophrenic_right, &left1, &right1, &left2, &right2);

        QSInterval_SQS<T> ival_left_left, ival_right_left,
                ival_left_right, ival_right_right;
        createIntervals(ival_left, offset_left, left_size_left,
                schizophrenic_left, mid_left, left1, right1, 
                ival_left_left, ival_right_left);
        createIntervals(ival_right, offset_right, left_size_right,
                schizophrenic_right, mid_right, left2, right2, 
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

        //Recursive call to quicksort/schizophrenicQuicksort
        if (sort_left && sort_right) {
            schizophrenicQuickSort(*left_i, *right_i, comp);
        } else if (sort_left) {
            quickSort(*left_i, comp);
        } else if (sort_right) {
            quickSort(*right_i, comp);
        }
    }
    
    
    /*
     * Check for base cases
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
     * Selects a random element from the interval as the pivot
     */
    template<class Compare>
    void getPivot(QSInterval_SQS<T> const &ival, T &pivot, long long &split_idx,
            Compare comp, bool &zero_global_elements) {
        return PivotSelection_SQS<T>::getPivot(ival, pivot, split_idx, comp, 
                generator, sample_generator, zero_global_elements);
    }
    
    /*
     * Select a random element as the pivot from both intervals
     */
    template<class Compare>
    void getPivotSchizo(QSInterval_SQS<T> const &ival_left,
                        QSInterval_SQS<T> const &ival_right, T &pivot_left,
                        T &pivot_right, long long &split_idx_left, long long &split_idx_right,
                        Compare comp) {
        PivotSelection_SQS<T>::getPivotSchizo(ival_left, ival_right, pivot_left, pivot_right,
                split_idx_left, split_idx_right, comp, generator, sample_generator);
    }
    
    /*
     * Partitions the data separatly for the elements with index smaller less_idx 
     * and the elements with index larger less_idx
     * Returns the indexes of the first element of the right partitions
     */
    template<class Compare>
    void partitionData(QSInterval_SQS<T> const &ival, T pivot, long long less_idx,
                       long long *index1, long long *index2, Compare comp) {
        long long start1 = ival.local_start, end1 = less_idx,
                start2 = less_idx, end2 = ival.local_end;
        *index1 = partitionData_(data->data(), pivot, start1, end1, true, comp);
        *index2 = partitionData_(data->data(), pivot, start2, end2, false, comp);
    }

    template<class Compare>
    long long partitionData_(T *data_ptr, T pivot, long long start, long long end,
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
    
    
    void calculateExchangeDataSchizo(QSInterval_SQS<T> &ival_left,
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
    void exchangeDataSchizo(QSInterval_SQS<T> &left, QSInterval_SQS<T> &right) {
        DataExchange_SQS<T>::exchangeDataSchizo(left, right);
    }  

    void calculateSplit(QSInterval_SQS<T> &ival, int &left_size, long long &offset,
            bool &schizophrenic, long long &mid) {
        assert(ival.global_small_elements != 0);
        long long last_small_element = ival.missing_first_PE + ival.global_small_elements - 1;
        
        left_size = ival.getRankFromIndex(last_small_element) + 1;
        offset = ival.getOffsetFromIndex(last_small_element);
                       
        if (offset + 1 == ival.getSplitSize(left_size - 1))
            schizophrenic = false;
        else
            schizophrenic = true;
        
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
            bool schizophrenic, RBC::Comm *left, RBC::Comm *right) {
        int size;
        RBC::Comm_size(ival.comm, &size);
        int left_end = left_size - 1;
        int right_start = left_size;
        if (schizophrenic)
            right_start--;
        int right_end = size - 1; //std::min((long long) size - 1, ival.global_elements - 1);
        right_end = std::max(right_start, right_end);
        RBC::Split_Comm(ival.comm, 0, left_end, right_start, right_end,
                left, right);
        RBC::Comm_free(ival.comm);
    }
    
    void createNewCommunicatorsSchizo(QSInterval_SQS<T> &ival_left,
            QSInterval_SQS<T> &ival_right, long long left_size_left,
            bool schizophrenic_left, long long left_size_right, bool schizophrenic_right,
            RBC::Comm *left_left, RBC::Comm *right_left,
            RBC::Comm *left_right, RBC::Comm *right_right) {
        if (ival_left.blocking_priority) {
            createNewCommunicators(ival_left, left_size_left, schizophrenic_left, left_left, right_left);
            createNewCommunicators(ival_right, left_size_right, schizophrenic_right, left_right, right_right);
        } else {
            createNewCommunicators(ival_right, left_size_right, schizophrenic_right, left_right, right_right);
            createNewCommunicators(ival_left, left_size_left, schizophrenic_left, left_left, right_left);
        }
    }
   
    void createIntervals(QSInterval_SQS<T> &ival, long long offset, int left_size,
            bool schizophrenic,
            long long mid, RBC::Comm &comm_left, RBC::Comm &comm_right, 
            QSInterval_SQS<T> &ival_left,
            QSInterval_SQS<T> &ival_right) {
        long long missing_last_left, missing_first_right;
        if (schizophrenic) {
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
            if (schizophrenic)
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
     * Add an interval with two PEs 
     */
    void addTwoPEInterval(QSInterval_SQS<T> const &ival) {
        two_PE_intervals.push_back(ival);
    }
    
    /* 
     * Add an interval that can be sorted locally
     */
    void addLocalInterval(QSInterval_SQS<T> &ival) {
        local_intervals.push_back(ival);
    }

    /* 
     * Sort the saved intervals with exactly two PEs
     */
    template<class Compare>
    void sortTwoPEIntervals(Compare comp) { 
        SequentialSort_SQS<T>::sortTwoPEIntervals(comp, two_PE_intervals);
        for (size_t i = 0; i < two_PE_intervals.size(); i++)
            RBC::Comm_free(two_PE_intervals[i].comm);
    }
    /* 
     * Sort all local intervals 
     */
    template<class Compare>
    void sortLocalIntervals(Compare comp) {
        SequentialSort_SQS<T>::sortLocalIntervals(comp, local_intervals);
        for (size_t i = 0; i < local_intervals.size(); i++) {
            RBC::Comm_free(local_intervals[i].comm);
        }
    }

    int depth = 0, seed;
    T *buffer;
    std::vector<T> *data;
    MPI_Datatype mpi_type;
    MPI_Comm parent_comm;
    std::mt19937_64 generator, sample_generator;
    std::vector<QSInterval_SQS<T>> local_intervals, two_PE_intervals;
    bool add_pivot;
    long long min_samples;
};

#endif // QUICKSORT_HPP
