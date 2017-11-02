/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SQS_PIVOTSELECTION_HPP
#define SQS_PIVOTSELECTION_HPP

#include "QuickSort.hpp"
#include <vector>
#include "Constants.hpp"
#include "QSInterval.hpp"
#include "../../RangeBasedComm/RBC.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>

#define W(X) #X << "=" << X << ", "

template<typename T>
class PivotSelection_SQS {

public:
    
    template<class Compare>
    static void getPivot(QSInterval_SQS<T> const &ival, T &pivot,
            long long &split_idx, Compare comp, std::mt19937_64 &generator,
            std::mt19937_64 &sample_generator, bool &zero_global_elements) {        
        long long global_samples, local_samples;
        
        if (ival.evenly_distributed)
            getLocalSamples_calculate(ival, global_samples, local_samples, comp, 
                    generator);
        else
            getLocalSamples_communicate(ival, global_samples, local_samples, comp,
                    generator);
        
        if (global_samples == -1) {
            zero_global_elements = true;
            return;
        } else {
            zero_global_elements = false;
        }
        
//        std::cout << W(ival.rank) << W(local_samples) << W(global_samples) << std::endl;
        
        std::vector<TbSplitter < T>> samples;
        pickLocalSamples(ival, local_samples, samples, comp, sample_generator);        
        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        
        //Merge function used in the gather
        std::function<void (void*, void*, void*) > merge = [tb_splitter_comp](
                void* start, void* mid, void* end) {
            std::inplace_merge(static_cast<TbSplitter<T>*> (start),
                    static_cast<TbSplitter<T>*> (mid),
                    static_cast<TbSplitter<T>*> (end),
                    tb_splitter_comp);
        };

        //Gather the samples to rank 0
        TbSplitter<T>* all_samples;
        if (ival.rank == 0)
            all_samples = new TbSplitter<T>[global_samples];

        MPI_Datatype splitter_type = TbSplitter<T>::MpiType(ival.mpi_type);
        
        RBC::Request req_gather;
        RBC::Igatherm(&samples[0], samples.size(), splitter_type, all_samples, global_samples,
                0, merge, ival.comm, &req_gather, Constants::PIVOT_GATHER);
        RBC::Wait(&req_gather, MPI_STATUS_IGNORE);

        TbSplitter<T> splitter;
        if (ival.rank == 0)
            splitter = all_samples[global_samples / 2];

        //Broadcast the pivot from rank 0
        RBC::Request req_bcast;
        RBC::Ibcast(&splitter, 1, splitter_type, 0, ival.comm, &req_bcast,
                Constants::PIVOT_BCAST);
        RBC::Wait(&req_bcast, MPI_STATUS_IGNORE);

        pivot = splitter.Splitter();
        selectSplitter(ival, splitter, split_idx);

        //        std::cout << W(ival.rank) << W(pivot) << std::endl;
        if (ival.rank == 0)
            delete[] all_samples;
    }
    
    /*
     * Select a random element as the pivot from both intervals
     */
    template<class Compare>
    static void getPivotSchizo(QSInterval_SQS<T> const &ival_left,
            QSInterval_SQS<T> const &ival_right, T &pivot_left,
            T &pivot_right, long long &split_idx_left, long long &split_idx_right,
            Compare comp, std::mt19937_64 &generator, std::mt19937_64 &sample_generator) {

        long long global_samples_left, global_samples_right,
                local_samples_left, local_samples_right;

        //Randomly pick samples from local data
        getLocalSamples_calculate(ival_left, global_samples_left,
                local_samples_left, comp, generator);
        getLocalSamples_calculate(ival_right, global_samples_right,
                local_samples_right, comp, generator);

        std::vector<TbSplitter < T>> samples_left, samples_right;
        pickLocalSamples(ival_left, local_samples_left, samples_left, comp,
                sample_generator);
        pickLocalSamples(ival_right, local_samples_right, samples_right,
                comp, sample_generator);
        
        
        //Merge function used in the gather
        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        std::function<void (void*, void*, void*) > merge = [tb_splitter_comp](
                void* start, void* mid, void* end) {
            std::inplace_merge(static_cast<TbSplitter<T>*> (start),
                    static_cast<TbSplitter<T>*> (mid),
                    static_cast<TbSplitter<T>*> (end), tb_splitter_comp);
        };

        TbSplitter<T> splitter_left, splitter_right;
        MPI_Datatype splitter_type = TbSplitter<T>::MpiType(ival_left.mpi_type);
        
        //Gather the samples
        TbSplitter<T>* all_samples = new TbSplitter<T>[global_samples_right];
        RBC::Request req_gather[2];
        RBC::Igatherm(&samples_left[0], samples_left.size(), splitter_type, nullptr,
                global_samples_left, 0, merge, ival_left.comm, &req_gather[0], Constants::PIVOT_GATHER);
        RBC::Igatherm(&samples_right[0], samples_right.size(), splitter_type, all_samples,
                global_samples_right, 0, merge, ival_right.comm, &req_gather[1], Constants::PIVOT_GATHER);
        RBC::Waitall(2, req_gather, MPI_STATUSES_IGNORE);

        //Determine pivot on right interval 
        splitter_right = all_samples[global_samples_right / 2];

        //Broadcast the pivots
        RBC::Request req_bcast[2];
        RBC::Ibcast(&splitter_left, 1, splitter_type, 0, ival_left.comm, &req_bcast[0],
                Constants::PIVOT_BCAST);
        RBC::Ibcast(&splitter_right, 1, splitter_type, 0, ival_right.comm, &req_bcast[1],
                Constants::PIVOT_BCAST);
        RBC::Waitall(2, req_bcast, MPI_STATUSES_IGNORE);

        pivot_left = splitter_left.Splitter();
        pivot_right = splitter_right.Splitter();

        selectSplitter(ival_left, splitter_left, split_idx_left);
        selectSplitter(ival_right, splitter_right, split_idx_right);

        delete[] all_samples;
    }

private:

    /*
     * Determine how much samples need to be send and pick them randomly
     */
    template<class Compare>
    static void getLocalSamples_calculate(QSInterval_SQS<T> const &ival, 
            long long &total_samples, long long &local_samples,
            Compare comp, std::mt19937_64 &generator) {
        total_samples = getSampleCount(ival.number_of_PEs,
                ival.getIndexFromRank(ival.number_of_PEs), ival.min_samples,
		ival.add_pivot);        
        int max_height = ceil(log2(ival.number_of_PEs));
        int own_height = 0;
        for (int i = 0; ((ival.rank >> i) % 2 == 0) && (i < max_height); i++)
            own_height++;

        int first_PE = 0;
        int last_PE = ival.number_of_PEs - 1;
        local_samples = total_samples;
        generator.seed(ival.seed);
        
        for (int height = max_height; height > 0; height--) {
            if (first_PE + pow(2, height - 1) > last_PE) {
                //right subtree is empty
            } else {
                int left_size = pow(2, height - 1);
                int right_size = last_PE - first_PE + 1 - left_size;
                assert(left_size > 0);
                assert(right_size > 0);
                assert(left_size + right_size == last_PE - first_PE + 1);
                long long left_elements = ival.getIndexFromRank(first_PE + left_size)
                    - ival.getIndexFromRank(first_PE);
                long long right_elements = ival.getIndexFromRank(last_PE + 1)
                    - ival.getIndexFromRank(first_PE + left_size);
                if (first_PE == 0)
                    left_elements -= ival.missing_first_PE;
                if (last_PE == ival.number_of_PEs - 1)
                    right_elements -= ival.missing_last_PE;
                
                assert(left_elements > 0);
                assert(right_elements >= 0);
                double percentage_left = static_cast<double> (left_elements)
                        / static_cast<double> (left_elements + right_elements);
                assert(percentage_left > 0);
                
                std::binomial_distribution<long long> binom_distr(local_samples, percentage_left);
                long long samples_left = binom_distr(generator);
                long long samples_right = local_samples - samples_left;

                int mid_PE = first_PE + pow(2, height - 1);
                if (ival.rank < mid_PE) {
                    //left side
                    last_PE = mid_PE - 1;
                    local_samples = samples_left;
                } else {
                    //right side
                    first_PE = mid_PE;
                    local_samples = samples_right;
                }
            }
        }
        
        return;
    }
    
    /*
     * Determine how much samples need to be send and pick them randomly
     */
    template<class Compare>
    static void getLocalSamples_communicate(QSInterval_SQS<T> const &ival, 
            long long &total_samples, long long &local_samples,
            Compare comp, std::mt19937_64 &generator) {
        DistrToLocSampleCount(ival.local_elements, total_samples,
                local_samples, generator, ival.comm, ival.min_samples, ival.add_pivot);
    }
    
    /*
     * Returns the number of global and local samples
     */
    static void DistrToLocSampleCount(long long const local_elements,
            long long &global_samples, long long &local_samples,
            std::mt19937_64 &async_gen, RBC::Comm const comm,
            long long min_samples, bool add_pivot) {
        int comm_size, rank;
        RBC::Comm_size(comm, &comm_size);
        RBC::Comm_rank(comm, &rank);

        // Calculate height in tree
        int logp = ceil(log2(comm_size));
        int height = 0;
        while ((rank >> height) % 2 == 0 && height < logp)
            height++;

        MPI_Status status;
        const int tag = Constants::DISTR_SAMPLE_COUNT;
        long long tree_elements = local_elements;
        std::vector<long> load_l, load_r;

        // Gather element count
        for (int k = 0; k < height; k++) {
            int src_rank = rank + (1 << k);
            if (src_rank < comm_size) {
                long long right_subtree;
                RBC::Recv(&right_subtree, 1, MPI_LONG_LONG, src_rank,
                        tag, comm, &status);

                load_r.push_back(right_subtree);
                load_l.push_back(tree_elements);
                tree_elements += right_subtree;
            }
        }
        assert(tree_elements >= 0);
        if (rank > 0) {
            int target_id = rank - (1 << height);
            RBC::Send(&tree_elements, 1, MPI_LONG_LONG, target_id, tag, comm);
        }
        
        // Distribute samples
        long long tree_sample_cnt;
        if (rank == 0) {
            if (tree_elements == 0)
                tree_sample_cnt = -1;
            else
                tree_sample_cnt = getSampleCount(comm_size, tree_elements,
                        min_samples, add_pivot);
            global_samples = tree_sample_cnt;
        } else {
            int src_id = rank - (1 << height);
            long long recvbuf[2];
            RBC::Recv(recvbuf, 2, MPI_LONG_LONG, src_id, tag, comm, &status);
            tree_sample_cnt = recvbuf[0];
            global_samples = recvbuf[1];
        }

        for (int kr = height; kr > 0; kr--) {
            int k = kr - 1;
            int target_rank = rank + (1 << k);
            if (target_rank < comm_size) {
                long long right_subtree_sample_cnt;

                if (tree_sample_cnt < 0) {
                    // There are no global elements at all.
                    right_subtree_sample_cnt = -1;
                } else if (tree_sample_cnt == 0) {
                    right_subtree_sample_cnt = 0;
                } else if (load_r.back() == 0) {
                    right_subtree_sample_cnt = 0;
                } else if (load_l.back() == 0) {
                    right_subtree_sample_cnt = tree_sample_cnt;
                    tree_sample_cnt -= right_subtree_sample_cnt;
                } else {
                    double right_p = load_r.back() / (static_cast<double> (load_l.back())
                            + static_cast<double> (load_r.back()));
                    std::binomial_distribution<long long> distr(tree_sample_cnt, right_p);
                    right_subtree_sample_cnt = distr(async_gen);
                    tree_sample_cnt -= right_subtree_sample_cnt;
                }

                long long sendbuf[2] = {right_subtree_sample_cnt, global_samples};
                RBC::Send(sendbuf, 2, MPI_LONG_LONG, target_rank, tag, comm);

                load_l.pop_back();
                load_r.pop_back();
            }
        }
        local_samples = tree_sample_cnt;
    }
    
    /*
     * Get the number of samples
     */
    static long long getSampleCount(int comm_size, long long global_elements,
            long long min_samples, bool add_pivot) {
        if (global_elements == 0)
            return -1;
        
        long long k_1 = 16, k_2 = 50, k_3 = min_samples; // tuning parameters
        long long count_1 = k_1 * log2(comm_size);
        long long count_2 = (global_elements / comm_size) / k_2;
        long long sample_count = std::max(count_1, std::max(count_2, k_3));
        if (add_pivot)
	    sample_count = std::max(count_1 + count_2, k_3);
        if (sample_count % 2 == 0)
            sample_count++;        
        
        return sample_count;
    }
    
    /*
     * Pick samples randomly from local data
     */
    template<class Compare>
    static void pickLocalSamples(QSInterval_SQS<T> const &ival, long long sample_count,
            std::vector<TbSplitter<T>> &sample_vec, Compare comp,
            std::mt19937_64 &generator) {
        T *data = ival.data->data();
        std::uniform_int_distribution<long long> distr(ival.local_start, ival.local_end - 1);
        for (long long i = 0; i < sample_count; i++) {
            long long index = distr(generator);
            long long global_index;
            if (ival.evenly_distributed)
                global_index = ival.getIndexFromRank(ival.rank) + index;
            else
                global_index = ival.rank;
            sample_vec.push_back(TbSplitter<T>(data[index], global_index));
        }

        auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                const TbSplitter<T>& second) {
            return first.compare(second, comp);
        };
        
        std::sort(sample_vec.begin(), sample_vec.end(), tb_splitter_comp);
    }

    /*
     * Calculate the local splitter index
     */
    static void selectSplitter(QSInterval_SQS<T> const &ival,
            TbSplitter<T> &splitter, long long &split_idx) {
        if (!ival.evenly_distributed) {
            if (ival.rank <= splitter.GID())
                split_idx = ival.local_end;
            else
                split_idx = ival.local_start;
            return;
        }
            
        long long splitter_PE;
        splitter_PE = ival.getRankFromIndex(splitter.GID());               
        if (ival.rank < splitter_PE)
            split_idx = ival.local_end;
        else if (ival.rank > splitter_PE)
            split_idx = ival.local_start;
        else {
            split_idx = ival.getOffsetFromIndex(splitter.GID()) + 1;
        }
    }
};

#endif /* PIVOTSELECTION_HPP */

