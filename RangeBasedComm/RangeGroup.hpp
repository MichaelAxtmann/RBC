/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#pragma once

#include <iterator>
#include <vector>
#include <iostream>
#include <ostream>
#include <algorithm>
#include <numeric>
#include <assert.h>

class RangeGroup {
public:
    RangeGroup() : f_(0), l_(-1), s_(0), size_(0) {}
    
    RangeGroup(int mpi_first, int mpi_last, int stride)
        : f_(mpi_first)
          // Cut tail, if last ranks not included.
        , l_(mpi_first + ((mpi_last - mpi_first) / stride) * stride)
        , s_(stride)
        , size_(((mpi_last - mpi_first) / stride) + 1) {}
    
    RangeGroup Split(int range_first, int range_last, int stride) const {
        const auto mpi_first = RangeRankToMpiRank(range_first);
        const auto mpi_last = RangeRankToMpiRank(range_last);
        const auto my_stride = stride * s_;
        return RangeGroup(mpi_first, mpi_last, my_stride);
    }
    
    /*
     * Returns the number of ranks in the group.
     */
    int getSize() {
        return size_;
    }

    /**
     * Transforms a range rank to a mpi rank.
     */
    int RangeRankToMpiRank(int range_rank) const {
        return f_ + range_rank * s_;
    }

    /**
     * Transforms a mpi rank to a range rank.
     */
    int MpiRankToRangeRank(int mpi_rank) const {
        return (mpi_rank - f_) / s_;
    }

    /**
     * Returns true if the mpi rank 'rank' is included in this range group.
     * Elsewise, this method returns false.
     */
    bool IsMpiRankIncluded(int mpi_rank) const {
        bool is_included = false;
        if (s_ > 0
            && f_ <= mpi_rank
            && l_ >= mpi_rank
            && abs((mpi_rank - f_) % s_) == 0) {
            is_included = true;
        } else if (s_ < 0
                   && f_ >= mpi_rank
                   && l_ <= mpi_rank
                   && abs((mpi_rank - f_) % s_) == 0) {
            is_included = true;
        }
        return is_included;
    }

    friend std::ostream& operator<<(std::ostream& os, const RangeGroup& rc)  
    {  
        os << rc.f_ << '/' << rc.l_ << '/' << rc.s_;
        return os;  
    }  

private:
    
    // First
    int f_;
    // Large
    int l_;
    // Stride
    int s_;
    // Number of ranks
    int size_;
};
