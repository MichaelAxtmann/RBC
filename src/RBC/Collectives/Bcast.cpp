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

#include <cmath>

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include <RBC.hpp>
#include <tlx/math.hpp>

#include "../PointToPoint/Recv.hpp"
#include "../PointToPoint/Send.hpp"
#include "../PointToPoint/Sendrecv.hpp"

namespace RBC {
int Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
          Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Bcast(buffer, count, datatype, root, comm.get());
  }
  int tag = Tag_Const::BCAST;
  MPI_Status status;
  int rank = 0;
  int size = 0;
  int own_height = 0;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  int target_mask = (rank - root + size) % size;
  int height = tlx::integer_log2_ceil(size);
  for (int i = 0; ((target_mask >> i) % 2 == 0) && (i < height); i++)
    own_height++;

  if (rank != root) {
    int temp_rank = rank - root;
    if (temp_rank < 0)
      temp_rank += size;
    int mask = 0x1;
    while ((temp_rank ^ mask) > temp_rank) {
      mask = mask << 1;
    }
    int temp_src = temp_rank ^ mask;
    int src = (temp_src + root) % size;
    Recv(buffer, count, datatype, src, tag, comm, &status);
  }

  while (height > 0) {
    if (own_height >= height) {
      int temp_rank = rank - root;
      if (temp_rank < 0)
        temp_rank += size;
      int temp_dest = temp_rank + std::pow(2, height - 1);
      if (temp_dest < size) {
        int dest = (temp_dest + root) % size;
        Send(buffer, count, datatype, dest, tag, comm);
      }
    }
    height--;
  }
  return 0;
}

namespace _internal {
namespace optimized {
/* If lpo2_diff == 0, then we rotate the ranks such that
 * the process with rank 'root' becomes rank 0. Otherwise,
 * rank 'root' becomes rank 1.
 */
int PhysicalRankToRootedRank(int rank, int phys_size, int lpo2_diff, int root) {
  return (rank - root + phys_size + (lpo2_diff == 0 ? 0 : 1)) % phys_size;
}

/* If lpo2_diff == 0, then we rotate the ranks such that
 * the process with rank 'root' becomes rank 0. Otherwise,
 * rank 'root' becomes rank 1.
 */
int RootedRankToPhysicalRank(int rooted_rank, int phys_size, int lpo2_diff, int root) {
  return (rooted_rank + root - (lpo2_diff == 0 ? 0 : 1) + phys_size) % phys_size;
}


/*
 * BcastScatterAllgather: Bcast algorithm with running time
 * O(alpha * log p + beta n).  p must not be a power of two!!!
 */
double BcastBinomialExpRunningTime(Comm const& comm, int sendcount,
                                   MPI_Datatype sendtype) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);

  int n = sendcount * datatype_size;               // In bytes

  return (optimized::kALPHA + optimized::kBETA* n) * tlx::integer_log2_ceil(size);
}

int BcastBinomial(void* buffer, int count, MPI_Datatype datatype, int root,
                  Comm const& comm) {
  return RBC::Bcast(buffer, count, datatype, root, comm);
}

int LogicalRankToPhysicalRank(int rank, int phys_size, int  /*lpo2*/, int lpo2_diff, int root) {
  const int rooted_rank = rank + std::min(lpo2_diff, rank + 1);
  return RootedRankToPhysicalRank(rooted_rank, phys_size, lpo2_diff, root);
}

int RootedRankToLogicalRank(int rooted_rank, int  /*lpo2*/, int lpo2_diff, int  /*root*/) {
  return rooted_rank < lpo2_diff * 2 ? rooted_rank / 2 : rooted_rank - lpo2_diff;
}


/* Let 'size' be the number of processes and lpo2 the
 * largest power of two which is smaller or equal to size.
 * If 'size' is a power of two, we do nothing.  Otherwise,
 * we handle process with rank 'root' as the second
 * process (the first process is 'root' - 1; we wrap the
 * last processes).  The first 'size' - lpo2 processes
 * with an even rank does not have any elements. In this
 * method, those even ranks receive the broadcasted
 * elements from their right neighbor. */
void ScatterToPowOfTwoProcs(void* sendbuf, int count,
                            int rank, int rank_rooted, int size, int lpo2,
                            MPI_Datatype datatype, Comm const& comm) {
  if (rank_rooted < (size - lpo2) * 2) {
    if (rank_rooted & 1) {
      // Odd rank_rooted
      Send(sendbuf, count, datatype, (rank + size - 1) % size,
           Tag_Const::BCAST, comm);
    } else {
      // Even rank_rooted
      Recv(sendbuf, count, datatype, (rank + 1) % size,
           Tag_Const::BCAST, comm, MPI_STATUS_IGNORE);
    }
  }
}


/*
 * BcastScatterAllgather: Bcast algorithm with running time
 * O(alpha * log p + beta n).  p must not be a power of two!!!
 */
double BcastScatterAllgatherExpRunningTime(Comm const& comm, int sendcount,
                                           MPI_Datatype sendtype) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);

  int n = sendcount * datatype_size;               // In bytes

  double algo = 2 * optimized::kALPHA* std::floor(std::log2(size)) +
                2 * (size - 1) / size * optimized::kBETA* n;

  if (!tlx::is_power_of_two(size)) {
    // Time to move data from/to non power of two nodes.
    algo += optimized::kBETA* n;
  }

  return algo;
}

int BcastScatterAllgather(void* buffer, int count, MPI_Datatype datatype, int root,
                          Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Bcast(const_cast<void*>(buffer), count,
                     datatype, root, comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);

  if (size == 1 || count == 0) return 0;

  const int lpo2 = tlx::round_down_to_power_of_two(size);
  const int lpo2_diff = size - lpo2;

  const int rank_rooted = PhysicalRankToRootedRank(rank, size, lpo2_diff, root);
  const int logical_rank = RootedRankToLogicalRank(rank_rooted, lpo2,
                                                   lpo2_diff, root);

  // We continue the algorithm with lpo2 number of
  // processes. The first lpo2_diff logical processes
  // with even ranks do not participate.  Note that we
  // have rotated process with rank root to rank 0.
  if ((rank_rooted < lpo2_diff * 2 && (rank_rooted & 1)) ||
      rank_rooted >= 2 * lpo2_diff) {
    /* Perform reduce-scatter operation */

    const int log_p = std::log2(size);

    // Will be filled with the number of elements on each level.
    std::vector<int> level_size(log_p);
    int rem_size = count;

    const int num_tailing_zeros = logical_rank == 0 ?
                                  log_p : static_cast<int>(tlx::ffs(logical_rank)) - 1;

    char* buffer_ptr = static_cast<char*>(buffer);

    for (int it = log_p - 1; it >= 0; --it) {
      level_size[it] = rem_size;

      const int right_count = rem_size / 2;
      const int left_count = rem_size - right_count;

      const bool right_cube = logical_rank & (1 << it);
      if (right_cube) {
        buffer_ptr += left_count * datatype_size;
        rem_size -= left_count;
      } else {
        rem_size -= right_count;
      }

      // We are still deactivated.
      if (it > num_tailing_zeros) {
        continue;
      }

      // We are active now.
      const int logical_target = logical_rank ^ (1 << it);
      const int phys_target = LogicalRankToPhysicalRank(logical_target, size, lpo2, lpo2_diff, root);
      // In our first active round, we receive
      // data. The process with rank 0 is excluded.
      if (it == num_tailing_zeros) {
        RecvNonZeroed(buffer_ptr, right_count, datatype,
                      phys_target, Tag_Const::BCAST,
                      comm, MPI_STATUS_IGNORE);
      } else {
        SendNonZeroed(buffer_ptr + left_count * datatype_size,
                      right_count, datatype, phys_target,
                      Tag_Const::BCAST,
                      comm);
      }
    }

    /* Perform allgather operation */

    for (int it = 0; it != log_p; ++it) {
      const int logical_target = logical_rank ^ 1 << it;
      const int phys_target = LogicalRankToPhysicalRank(logical_target, size, lpo2, lpo2_diff, root);
      const bool left_target = logical_target < logical_rank;
      const int target_size = level_size[it] - rem_size;

      if (left_target) {
        SendrecvNonZeroed(buffer_ptr,
                          rem_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          buffer_ptr - target_size * datatype_size,
                          target_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          comm,
                          MPI_STATUS_IGNORE);
        buffer_ptr -= target_size * datatype_size;
      } else {
        SendrecvNonZeroed(buffer_ptr,
                          rem_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          buffer_ptr + rem_size * datatype_size,
                          target_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          comm,
                          MPI_STATUS_IGNORE);
      }

      rem_size += target_size;
    }
  }

  /* Handle non power of two case */
  ScatterToPowOfTwoProcs(buffer, count, rank, rank_rooted,
                         size, lpo2, datatype, comm);

  return 0;
}

/*
 *
 * Blocking broadcast with equal amount of elements on each process
 * This method uses different implementations depending on the
 * size of comm and the input size.
 */
int Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
          Comm const& comm) {
  double binomial = BcastBinomialExpRunningTime(comm, count, datatype);
  double scat_allg = BcastScatterAllgatherExpRunningTime(comm, count, datatype);

  if (binomial < scat_allg) {
    return BcastBinomial(buffer, count, datatype, root, comm);
  } else {
    return BcastScatterAllgather(buffer, count, datatype, root, comm);
  }
}
}         // end namespace optimized

/*
 * Request for the broadcast
 */
class IbcastReq : public RequestSuperclass {
 public:
  IbcastReq(void* buffer, int count, MPI_Datatype datatype, int root,
            int tag, Comm const& omm);
  int test(int* flag, MPI_Status* status);

 private:
  void* m_buffer;
  MPI_Datatype m_datatype;
  int m_count, m_root, m_tag, m_own_height, m_size, m_rank, m_height, m_received, m_sends;
  Comm m_comm;
  bool m_send, m_completed, m_mpi_collective;
  Request m_recv_req;
  std::vector<Request> m_req_vector;
  MPI_Request m_mpi_req;
};
}  // namespace _internal

int Ibcast(void* buffer, int count, MPI_Datatype datatype,
           int root, Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IbcastReq>(buffer, count,
                                                      datatype, root, tag, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IbcastReq::IbcastReq(void* buffer, int count, MPI_Datatype datatype,
                                     int root, int tag, RBC::Comm const& comm) :
  m_buffer(buffer),
  m_datatype(datatype),
  m_count(count),
  m_root(root),
  m_tag(tag),
  m_own_height(0),
  m_size(0),
  m_rank(0),
  m_height(0),
  m_received(0),
  m_comm(comm),
  m_send(false),
  m_completed(false),
  m_mpi_collective(false) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Ibcast(buffer, count, datatype, root, comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif
  RBC::Comm_rank(comm, &m_rank);
  RBC::Comm_size(comm, &m_size);
  m_sends = 0;
  assert(m_size > 0);
  int temp_rank = (m_rank - root + m_size) % m_size;
  m_height = tlx::integer_log2_ceil(m_size);
  for (int i = 0; ((temp_rank >> i) % 2 == 0) && (i < m_height); i++)
    m_own_height++;
  if (m_rank == root)
    m_received = 1;
  else
    RBC::Irecv(buffer, count, datatype, MPI_ANY_SOURCE, tag, comm, &m_recv_req);
}

int RBC::_internal::IbcastReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (!m_received) {
    RBC::Test(&m_recv_req, &m_received, MPI_STATUS_IGNORE);
  }
  if (m_received && !m_send) {
    while (m_height > 0) {
      if (m_own_height >= m_height) {
        int temp_rank = m_rank - m_root;
        if (temp_rank < 0)
          temp_rank += m_size;
        int temp_dest = temp_rank + std::pow(2, m_height - 1);
        if (temp_dest < m_size) {
          int dest = (temp_dest + m_root) % m_size;
          m_req_vector.push_back(RBC::Request());
          RBC::Isend(m_buffer, m_count, m_datatype, dest, m_tag, m_comm, &m_req_vector.back());
        }
      }
      m_height--;
    }
    m_send = true;
  }
  if (m_send) {
    RBC::Testall(m_req_vector.size(), &m_req_vector.front(), flag, MPI_STATUSES_IGNORE);
    if (*flag == 1)
      m_completed = true;
  }
  return 0;
}
