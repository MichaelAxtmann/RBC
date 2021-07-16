/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#pragma once

#include <iostream>

class RangeGroup {
 public:
  RangeGroup();

  RangeGroup(int mpi_first, int mpi_last, int stride, int mpi_rank);

  void reset(int mpi_first, int mpi_last, int stride, int mpi_rank);

  RangeGroup Split(int range_first, int range_last, int stride) const;

  int getSize() const;

  int getRank() const;

  int getMpiFirst() const;

  int getMpiLast() const;

  int getStride() const;

/**
 * Transforms a range rank to a mpi rank.
 */
  int RangeRankToMpiRank(int range_rank) const;

/**
 * Transforms a mpi rank to a range rank.
 */
  int MpiRankToRangeRank(int mpi_rank) const;

/**
 * Returns true if the mpi rank 'rank' is included in this range group.
 * Elsewise, this method returns false.
 */
  bool IsMpiRankIncluded(int mpi_rank) const;

  friend std ::ostream& operator<< (std::ostream& os, const RangeGroup& rc) {
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
// RBC rank. This rank is valid, if this PE is part of the
// group. Otherwise, 'rank_' is not defined.
  int rank_;
};
