/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <cassert>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "RangeGroup.hpp"
#include "RBC.hpp"

// Initially, we just delete the MPI_Comm object -- no freeing.
RBC::Comm::Comm() :
  m_comm(new MPICommWrapper { })
{ }

RBC::Comm::Comm(const std::shared_ptr<MPICommWrapper>& comm,
                const RangeGroup& group) :
  m_comm(comm),
  m_group(group) { }

RBC::Comm::Comm(const std::shared_ptr<MPICommWrapper>& comm,
                int mpi_first, int mpi_last, int stride, int mpi_rank) :
  m_comm(comm),
  m_group(mpi_first, mpi_last, stride, mpi_rank) { }

RBC::Comm::Comm(std::shared_ptr<MPICommWrapper>&& comm,
                const RangeGroup& group) :
  m_comm(comm),
  m_group(group) { }

RBC::Comm::Comm(std::shared_ptr<MPICommWrapper>&& comm,
                int mpi_first, int mpi_last, int stride, int mpi_rank) :
  m_comm(comm),
  m_group(mpi_first, mpi_last, stride, mpi_rank) { }


void RBC::Comm::reset(const std::shared_ptr<MPICommWrapper>& comm,
                      const RangeGroup& group) {
  m_comm = comm;
  m_group = group;
}

void RBC::Comm::reset(const std::shared_ptr<MPICommWrapper>& comm,
                      int mpi_first, int mpi_last, int stride, int mpi_rank) {
  m_comm = comm;
  m_group.reset(mpi_first, mpi_last, stride, mpi_rank);
}

void RBC::Comm::reset(std::shared_ptr<MPICommWrapper>&& comm,
                      const RangeGroup& group) {
  m_comm = comm;
  m_group = group;
}
void RBC::Comm::reset(std::shared_ptr<MPICommWrapper>&& comm,
                      int mpi_first, int mpi_last, int stride, int mpi_rank) {
  m_comm = comm;
  m_group.reset(mpi_first, mpi_last, stride, mpi_rank);
}


RBC::Comm::~Comm() { }

MPI_Comm RBC::Comm::get() const {
  return m_comm->get();
}
int RBC::Comm::getSize() const {
  return m_group.getSize();
}

int RBC::Comm::getRank() const {
  return m_group.getRank();
}


RBC::Comm RBC::Create_Comm_from_MPI(MPI_Comm mpi_comm,
                                    bool use_mpi_collectives, bool split_mpi_comm, bool use_comm_create) {
  RBC::Comm rcomm;
  RBC::Comm::Create_Comm_from_MPI(mpi_comm, &rcomm,
                                  use_mpi_collectives, split_mpi_comm, use_comm_create);
  return rcomm;
}

int RBC::Create_Comm_from_MPI(MPI_Comm mpi_comm, RBC::Comm* rcomm,
                              bool use_mpi_collectives, bool split_mpi_comm, bool use_comm_create) {
  RBC::Comm::Create_Comm_from_MPI(mpi_comm, rcomm,
                                  use_mpi_collectives, split_mpi_comm, use_comm_create);
  return 0;
}

int RBC::Comm::Create_Comm_from_MPI(MPI_Comm mpi_comm, RBC::Comm* rcomm,
                                    bool use_mpi_collectives, bool split_mpi_comm, bool use_comm_create) {
  const bool destroy = false;
  const bool is_mpi_comm = true;
  int rank = -1;
  MPI_Comm_rank(mpi_comm, &rank);
  int size = -1;
  MPI_Comm_size(mpi_comm, &size);
  rcomm->reset(std::make_shared<MPICommWrapper>(mpi_comm, destroy,
                                                use_mpi_collectives, split_mpi_comm, use_comm_create,
                                                is_mpi_comm), 0, size - 1, 1, rank);
  return 0;
}

int RBC::Comm_create(RBC::Comm const& comm, RBC::Comm* new_comm,
                     int first, int last, int stride) {
  return RBC::Comm::Comm_create(comm, new_comm, first, last, stride);
}

int RBC::Comm::Comm_create(RBC::Comm const& comm, RBC::Comm* new_comm,
                           int first, int last, int stride) {
  if (comm.splitMPIComm()) {
    int rank;
    RBC::Comm_rank(comm, &rank);
    MPI_Group group, new_group;
    MPI_Comm_group(comm.get(), &group);
    int ranges[3] = { first, last, stride };
    MPI_Group_range_incl(group, 1, &ranges, &new_group);
    MPI_Comm mpi_comm{ MPI_COMM_NULL };
    MPI_Comm_create(comm.get(), new_group, &mpi_comm);

    if (mpi_comm != MPI_COMM_NULL) {
      int size = -1;
      MPI_Comm_size(mpi_comm, &size);

      const bool destroy = true;
      const bool use_mpi_collectives = comm.useMPICollectives();
      const bool split_mpi_comm = comm.splitMPIComm();
      const bool use_comm_create = comm.useCommCreate();
      const bool is_mpi_comm = true;
      int new_rank = -1;
      MPI_Comm_rank(mpi_comm, &new_rank);
      const int new_first = 0;
      const int new_last = size - 1;
      const int new_stride = 1;

      new_comm->reset(std::make_shared<MPICommWrapper>(mpi_comm, destroy,
                                                       use_mpi_collectives,
                                                       split_mpi_comm, use_comm_create,
                                                       is_mpi_comm),
                      new_first, new_last,
                      new_stride, new_rank);
    } else {
      new_comm->m_comm.reset(new MPICommWrapper{ });
    }
  } else {
    int rank;
    RBC::Comm_rank(comm, &rank);
    int mpi_rank = comm.RangeRankToMpiRank(rank);
    const RangeGroup range_group = comm.m_group.Split(first, last, stride);
    if (range_group.IsMpiRankIncluded(mpi_rank)) {
      new_comm->reset(comm.m_comm, range_group);
    } else {
      new_comm->m_comm.reset(new MPICommWrapper{ });
    }
  }
  return 0;
}

int RBC::Comm_create_group(RBC::Comm const& comm, RBC::Comm* new_comm,
                           int first, int last, int stride) {
  return RBC::Comm::Comm_create_group(comm, new_comm, first, last, stride);
}

int RBC::Comm::Comm_create_group(RBC::Comm const& comm, RBC::Comm* new_comm,
                                 int first, int last, int stride) {
  if (comm.splitMPIComm()) {
    MPI_Group group, new_group;
    MPI_Comm_group(comm.get(), &group);
    int ranges[3] = { first, last, stride };
    MPI_Group_range_incl(group, 1, &ranges, &new_group);
    MPI_Comm new_mpi_comm{ MPI_COMM_NULL };
#ifdef NO_NONBLOCKING_COLL_MPI_SUPPORT
    MPI_Comm_create(comm.m_mpi_comm, new_group, &new_mpi_comm);
#else
    MPI_Comm_create_group(comm.get(), new_group, 0, &new_mpi_comm);
#endif
    if (new_mpi_comm != MPI_COMM_NULL) {
      const bool destroy = true;
      const bool is_mpi_comm = true;
      auto comm_wrapper = std::make_shared<MPICommWrapper>(new_mpi_comm,
                                                           destroy, comm.useMPICollectives(),
                                                           comm.splitMPIComm(),
                                                           comm.useCommCreate(), is_mpi_comm);
      const int mpi_first = 0;
      int mpi_last = -1;
      MPI_Comm_size(new_mpi_comm, &mpi_last);
      --mpi_last;
      const int new_stride = 1;
      int mpi_rank = -1;
      MPI_Comm_rank(new_mpi_comm, &mpi_rank);
      new_comm->reset(std::move(comm_wrapper), mpi_first, mpi_last, new_stride, mpi_rank);
    } else {
      new_comm->m_comm.reset(new MPICommWrapper{ });
    }
  } else {
    RBC::Comm_create(comm, new_comm, first, last, stride);
  }
  return 0;
}

int RBC::Split_Comm(Comm const& comm, int left_start, int left_end, int right_start,
                    int right_end, Comm* left_comm, Comm* right_comm) {
  return RBC::Comm::Split_Comm(comm, left_start, left_end, right_start,
                               right_end, left_comm, right_comm);
}

int RBC::Comm::Split_Comm(Comm const& comm, int left_start, int left_end, int right_start,
                          int right_end, Comm* left_comm, Comm* right_comm) {
  if (!comm.splitMPIComm()) {
    int rank;
    RBC::Comm_rank(comm, &rank);
    if (rank >= left_start && rank <= left_end)
      RBC::Comm_create(comm, left_comm, left_start, left_end);
    if (rank >= right_start && rank <= right_end)
      RBC::Comm_create(comm, right_comm, right_start, right_end);
  } else {
    // split MPI communicator
    assert(comm.m_comm->isMPIComm());
    MPI_Comm mpi_comm = comm.get();
    MPI_Comm mpi_left{ MPI_COMM_NULL };
    MPI_Comm mpi_right{ MPI_COMM_NULL };
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    // create MPI communicators
    if (left_end < right_start || right_end < left_start) {
      // disjoint communicators
      if (comm.useCommCreate()) {
        MPI_Group group, new_group = MPI_GROUP_EMPTY;
        MPI_Comm_group(comm.get(), &group);
        int ranges[2][3] = {
          { left_start, left_end, 1 },
          { right_start, right_end, 1 }
        };
        if (rank >= left_start && rank <= left_end)
          MPI_Group_range_incl(group, 1, &ranges[0], &new_group);
        else if (rank >= right_start && rank <= right_end)
          MPI_Group_range_incl(group, 1, &ranges[1], &new_group);

        MPI_Comm new_mpi_comm = MPI_COMM_NULL;
#ifdef NO_NONBLOCKING_COLL_MPI_SUPPORT
        MPI_Comm_create(comm.get(), new_group, new_mpi_comm);
#else
        MPI_Comm_create_group(comm.get(), new_group, 0, &new_mpi_comm);
#endif
        if (rank >= left_start && rank <= left_end) {
          assert(!(rank >= right_start && rank <= right_end));
          mpi_left = new_mpi_comm;
        } else if (rank >= right_start && rank <= right_end) {
          mpi_right = new_mpi_comm;
        }
      } else {
        int color;
        if (rank >= left_start && rank <= left_end) {
          assert(!(rank >= right_start && rank <= right_end));
          color = 1;
        } else if (rank >= right_start && rank <= right_end) {
          color = 2;
        } else {
          color = MPI_UNDEFINED;
        }

        MPI_Comm new_comm{ MPI_COMM_NULL };
        MPI_Comm_split(comm.get(), color, rank, &new_comm);

        if (color == 1) {
          mpi_left = new_comm;
        } else if (color == 2) {
          mpi_right = new_comm;
        }
      }
    } else {
      // overlapping communicators
      if (comm.useCommCreate()) {
        MPI_Group group, new_group_left = MPI_GROUP_EMPTY, new_group_right = MPI_GROUP_EMPTY;
        MPI_Comm_group(comm.get(), &group);
        int ranges[2][3] = {
          { left_start, left_end, 1 },
          { right_start, right_end, 1 }
        };
        if (rank >= left_start && rank <= left_end)
          MPI_Group_range_incl(group, 1, &ranges[0], &new_group_left);
        if (rank >= right_start && rank <= right_end)
          MPI_Group_range_incl(group, 1, &ranges[1], &new_group_right);

#ifdef NO_NONBLOCKING_COLL_MPI_SUPPORT

        MPI_Comm_create(comm.mpi_comm, new_group_left, &mpi_left);
        MPI_Comm_create(comm.mpi_comm, new_group_right, &mpi_right);

        if (!(rank >= left_start && rank <= left_end)) {
          assert(mpi_left == MPI_COMM_NULL);
        }
        if (!(rank >= right_start && rank <= right_end)) {
          assert(mpi_right == MPI_COMM_NULL);
        }
#else
        if (rank >= left_start && rank <= left_end) {
          MPI_Comm_create_group(comm.get(), new_group_left, 0, &mpi_left);
        }
        if (rank >= right_start && rank <= right_end) {
          MPI_Comm_create_group(comm.get(), new_group_right, 0, &mpi_right);
        }
#endif
      } else {
        int color1, color2;
        if (rank >= left_start && rank <= left_end) {
          color1 = 1;
        } else {
          color1 = MPI_UNDEFINED;
        }

        if (rank >= right_start && rank <= right_end) {
          color2 = 2;
        } else {
          color2 = MPI_UNDEFINED;
        }

        MPI_Comm_split(comm.get(), color1, rank, &mpi_left);
        MPI_Comm_split(comm.get(), color2, rank, &mpi_right);

        if (color1 == MPI_UNDEFINED) {
          assert(mpi_left == MPI_COMM_NULL);
        }
        if (color2 == MPI_UNDEFINED) {
          assert(mpi_right == MPI_COMM_NULL);
        }
      }
    }

    if (rank >= left_start && rank <= left_end) {
      assert(mpi_left != MPI_COMM_NULL);
      const bool destroy = true;
      const bool is_mpi_comm = true;
      auto comm_wrapper = std::make_shared<MPICommWrapper>(mpi_left,
                                                           destroy, comm.useMPICollectives(),
                                                           comm.splitMPIComm(),
                                                           comm.useCommCreate(), is_mpi_comm);
      const int mpi_first = 0;
      int mpi_last = -1;
      MPI_Comm_size(mpi_left, &mpi_last);
      --mpi_last;
      const int new_stride = 1;
      int mpi_rank = -1;
      MPI_Comm_rank(mpi_left, &mpi_rank);
      left_comm->reset(std::move(comm_wrapper), mpi_first, mpi_last, new_stride, mpi_rank);
    } else {
      assert(mpi_left == MPI_COMM_NULL);
      left_comm->m_comm.reset(new MPICommWrapper{ });
    }
    if (rank >= right_start && rank <= right_end) {
      assert(mpi_right != MPI_COMM_NULL);
      const bool destroy = true;
      const bool is_mpi_comm = true;
      auto comm_wrapper = std::make_shared<MPICommWrapper>(mpi_right,
                                                           destroy, comm.useMPICollectives(),
                                                           comm.splitMPIComm(),
                                                           comm.useCommCreate(), is_mpi_comm);
      const int mpi_first = 0;
      int mpi_last = -1;
      MPI_Comm_size(mpi_right, &mpi_last);
      --mpi_last;
      const int new_stride = 1;
      int mpi_rank = -1;
      MPI_Comm_rank(mpi_right, &mpi_rank);
      right_comm->reset(std::move(comm_wrapper), mpi_first, mpi_last, new_stride, mpi_rank);
    } else {
      assert(mpi_right == MPI_COMM_NULL);
      right_comm->m_comm.reset(new MPICommWrapper{ });
    }
  }
  return 0;
}

int RBC::Comm_free(Comm& comm) {
  std::ignore = comm;
  // We don't need to free communicators, as we use shared pointers
  // with costumized deleters.
  return 0;
}

int RBC::Comm::MpiRankToRangeRank(int mpi_rank) const {
  return m_group.MpiRankToRangeRank(mpi_rank);
}

int RBC::Comm::RangeRankToMpiRank(int range_rank) const {
  return m_group.RangeRankToMpiRank(range_rank);
}

bool RBC::Comm::useMPICollectives() const {
  return m_comm->useMPICollectives() && m_comm->splitMPIComm() && m_comm->isMPIComm();
}

bool RBC::Comm::useCommCreate() const {
  return m_comm->useCommCreate();
}


bool RBC::Comm::splitMPIComm() const {
  if (m_comm->splitMPIComm()) {
    assert(m_comm->isMPIComm());
  }
  return m_comm->splitMPIComm();
}

bool RBC::Comm::includesMpiRank(int rank) const {
  return m_group.IsMpiRankIncluded(rank);
}

bool RBC::Comm::isEmpty() const {
  return get() == MPI_COMM_NULL;
}

RBC::Request::Request() :
  req_ptr(nullptr) { }

void RBC::Request::set(const std::shared_ptr<_internal::RequestSuperclass>& req) {
  req_ptr = req;
}

namespace RBC {
std::ostream& operator<< (std::ostream& os, const RBC::Comm& comm) {
  os
    << "("
    << comm.get() << ", " << comm.m_group
    << ")";
  return os;
}
}  // namespace RBC

RBC::Request& RBC::Request::operator= (const Request& req) {
  this->set(req.req_ptr);
  return *this;
}

int RBC::Request::test(int* flag, MPI_Status* status) {
  *flag = 0;
  if (req_ptr) {
    return req_ptr->test(flag, status);
  } else {
    *flag = 1;
    return 0;
  }
}

int RBC::Comm_rank(RBC::Comm const& comm, int* rank) {
  *rank = comm.getRank();
  return 0;
}

int RBC::Comm_size(RBC::Comm const& comm, int* size) {
  *size = comm.getSize();
  return 0;
}

int RBC::Iprobe(int source, int tag, RBC::Comm const& comm, int* flag, MPI_Status* status) {
  if (source != MPI_ANY_SOURCE)
    source = comm.RangeRankToMpiRank(source);
  MPI_Status tmp_status;
  int return_value = MPI_Iprobe(source, tag, comm.get(), flag, &tmp_status);
  if (*flag) {
    if (!comm.includesMpiRank(tmp_status.MPI_SOURCE))
      *flag = 0;
    else if (status != MPI_STATUS_IGNORE)
      *status = tmp_status;
  }
  return return_value;
}

int RBC::Probe(int source, int tag, RBC::Comm const& comm, MPI_Status* status) {
  if (source != MPI_ANY_SOURCE) {
    source = comm.RangeRankToMpiRank(source);
    return MPI_Probe(source, tag, comm.get(), status);
  }

  int flag = 0;
  while (!flag)
    RBC::Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, status);
  return 0;
}

int RBC::Test(RBC::Request* request, int* flag, MPI_Status* status) {
  return request->test(flag, status);
}

int RBC::Testall(int count, RBC::Request* array_of_requests, int* flag,
                 MPI_Status array_of_statuses[]) {
  *flag = 1;
  for (int i = 0; i < count; i++) {
    int temp_flag;
    if (array_of_statuses == MPI_STATUSES_IGNORE) {
      Test(&array_of_requests[i], &temp_flag, MPI_STATUS_IGNORE);
    } else {
      Test(&array_of_requests[i], &temp_flag, &array_of_statuses[i]);
    }
    if (temp_flag == 0) {
      *flag = 0;
    }
  }
  return 0;
}

int RBC::Wait(RBC::Request* request, MPI_Status* status) {
  int flag = 0, return_value;
  while (flag == 0) {
    return_value = Test(request, &flag, status);
  }
  return return_value;
}

int RBC::Waitall(int count, RBC::Request array_of_requests[],
                 MPI_Status array_of_statuses[]) {
  // We are not allowed to call Testall until success
  // as Testall calls Test on requests which are already successful.
  int pending_requests = count;
  std::vector<bool> succs(count, false);
  while (pending_requests > 0) {
    for (int i = 0; i < count; i++) {
      if (succs[i]) continue;
      int succ;
      if (array_of_statuses == MPI_STATUSES_IGNORE) {
        Test(&array_of_requests[i], &succ, MPI_STATUS_IGNORE);
      } else {
        Test(&array_of_requests[i], &succ, &array_of_statuses[i]);
      }
      if (succ == 1) {
        --pending_requests;
        succs[i] = true;
      }
    }
  }
  return 0;
}

int RBC::get_Rank_from_Status(RBC::Comm const& comm, MPI_Status status) {
  return comm.MpiRankToRangeRank(status.MPI_SOURCE);
}
