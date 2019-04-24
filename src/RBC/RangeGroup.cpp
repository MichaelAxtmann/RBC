/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#include <cstdlib>
#include <iostream>

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
