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

#include <cstdlib>

#include "RangeGroup.hpp"

#include <EvilHeaders.hpp>

RangeGroup::RangeGroup() :
  f_(0),
  l_(-1),
  s_(0),
  size_(0),
  rank_(MPI_ANY_SOURCE) { }

RangeGroup::RangeGroup(int mpi_first, int mpi_last, int stride, int mpi_rank) :
  f_(mpi_first)
  // Cut tail, if last ranks not included.
  ,
  l_(mpi_first + ((mpi_last - mpi_first) / stride) * stride),
  s_(stride),
  size_(((mpi_last - mpi_first) / stride) + 1),
  rank_(MpiRankToRangeRank(mpi_rank))
{ }

void RangeGroup::reset(int mpi_first, int mpi_last, int stride, int mpi_rank) {
  f_ = mpi_first;
  l_ = mpi_first + ((mpi_last - mpi_first) / stride) * stride;
  s_ = stride;
  size_ = ((mpi_last - mpi_first) / stride) + 1;
  rank_ = MpiRankToRangeRank(mpi_rank);
}

RangeGroup RangeGroup::Split(int range_first, int range_last, int stride) const {
  const auto mpi_first = RangeRankToMpiRank(range_first);
  const auto mpi_last = RangeRankToMpiRank(range_last);
  const auto my_stride = stride * s_;
  return RangeGroup(mpi_first, mpi_last, my_stride,
                    RangeRankToMpiRank(rank_));
}

int RangeGroup::getSize() const {
  return size_;
}

int RangeGroup::getRank() const {
  return rank_;
}

int RangeGroup::getMpiFirst() const {
  return f_;
}

int RangeGroup::getMpiLast() const {
  return l_;
}

int RangeGroup::getStride() const {
  return s_;
}

int RangeGroup::RangeRankToMpiRank(int range_rank) const {
  if (range_rank == MPI_ANY_SOURCE) {
    return range_rank;
  }
  return f_ + range_rank * s_;
}

int RangeGroup::MpiRankToRangeRank(int mpi_rank) const {
  if (mpi_rank == MPI_ANY_SOURCE) {
    return mpi_rank;
  }
  return (mpi_rank - f_) / s_;
}

bool RangeGroup::IsMpiRankIncluded(int mpi_rank) const {
  bool is_included = false;
  if (s_ > 0 &&
      f_ <= mpi_rank &&
      l_ >= mpi_rank &&
      abs((mpi_rank - f_) % s_) == 0) {
    is_included = true;
  } else if (s_ < 0 &&
             f_ >= mpi_rank &&
             l_ <= mpi_rank &&
             abs((mpi_rank - f_) % s_) == 0) {
    is_included = true;
  }
  return is_included;
}
