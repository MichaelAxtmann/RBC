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
