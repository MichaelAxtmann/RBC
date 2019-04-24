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
#include <cstring>
#include <memory>

#include "../PointToPoint/Sendrecv.hpp"
#include "MachineConstants.hpp"
#include "RBC.hpp"
#include "tlx/math.hpp"

namespace RBC {
int Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
              MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.get());
  }

  int root = 0;
  RBC::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  RBC::Bcast(recvbuf, count, datatype, root, comm);
  return 0;
}

namespace _internal {
namespace optimized {
/* Let 'size' the number of processes and lpo2 the largest
 * power of two which is smaller or equal to size. The
 * first 'size' - lpo2 processes with an even rank send
 * their elements to their right neighbor and the receiver
 * reduces the incoming data with its own data. */
void AnyProcCountToPowOfTwoReducer(void* sendbuf,
                                   void* tmpbuf, int count,
                                   MPI_Datatype datatype,
                                   MPI_Op op, Comm const& comm) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  int lpo2 = tlx::round_down_to_power_of_two(size);

  if (rank < (size - lpo2) * 2) {
    if (rank & 1) {
      // Odd rank
      Recv(tmpbuf, count, datatype, rank - 1, Tag_Const::ALLREDUCE, comm,
           MPI_STATUS_IGNORE);

      MPI_Reduce_local(tmpbuf, sendbuf, count, datatype, op);
    } else {
      // Even rank
      Send(sendbuf, count, datatype, rank + 1, Tag_Const::ALLREDUCE, comm);
    }
  }
}

/* This function is a duplicate of a function in
 * Bcast.cpp. If we copy this function once more, we
 * have to make it reusable.
 *
 * Let 'size' be the number of processes and lpo2 the
 * largest power of two which is smaller or equal to
 * size. The first 'size' - lpo2 processes with an even
 * rank does not have any elements. The first 'size' -
 * lpo2 processes with an odd rank send their elements to
 * their left neighbor. */
void ScatterToPowOfTwoProcs(void* sendbuf, int count, int rank, int size,
                            MPI_Datatype datatype, Comm const& comm) {
  const int lpo2 = tlx::round_down_to_power_of_two(size);

  if (rank < (size - lpo2) * 2) {
    if (rank & 1) {
      // Odd rank
      Send(sendbuf, count, datatype, rank - 1, Tag_Const::ALLREDUCE, comm);
    } else {
      // Even rank
      Recv(sendbuf, count, datatype, rank + 1, Tag_Const::ALLREDUCE, comm,
           MPI_STATUS_IGNORE);
    }
  }
}

int LogicalRankToPhysicalRank(int rank, int lpo2_diff) {
  return rank + std::min(lpo2_diff, rank + 1);
}

int PhysicalRankToLogicalRank(int rank, int lpo2_diff) {
  return rank < lpo2_diff * 2 ? rank / 2 : rank - lpo2_diff;
}


/*
 * AllreduceScatterAllgather: Allgather algorithm with running time
 * O(alpha * log p + beta n).  p must not be a power of two!!!
 */
double AllreduceScatterAllgatherExpRunningTime(Comm const& comm, int sendcount,
                                               MPI_Datatype sendtype) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);

  int n = sendcount * datatype_size;               // In bytes

  double algo = 2 * kALPHA * std::floor(std::log2(size)) +
                2 * (size - 1) / size * kBETA * n;
  // We expect that the reduction takes beta/3 time.
  double calc = 1. / 3. * kBETA * n * std::floor(std::log2(size));

  if (!tlx::is_power_of_two(size)) {
    // Time to move data from/to non power of two nodes.
    algo += 2. * kBETA * n;
    calc += 1. / 3. * kBETA * n;
  }

  return algo + calc;
}

int AllreduceScatterAllgather(const void* sendbuf, void* recvbuf,
                              int count, MPI_Datatype datatype,
                              MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, count,
                         datatype, op, comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  if (count == 0) return 0;

  // Move local input to output buffer.
  memcpy(static_cast<char*>(recvbuf), sendbuf, recv_size);

  if (size == 1) return 0;

  std::unique_ptr<char[]> tmpbuf = std::make_unique<char[]>(recv_size);

  /* Handle power of two case */

  AnyProcCountToPowOfTwoReducer(recvbuf, tmpbuf.get(), count, datatype,
                                op, comm);

  const int lpo2 = tlx::round_down_to_power_of_two(size);
  const int lpo2_diff = size - lpo2;

  // We continue the algorithm with lpo2 number of processes.
  if ((rank < lpo2_diff * 2 && (rank & 1)) ||
      rank >= 2 * lpo2_diff) {
    /* Perform reduce-scatter operation */

    char* recvbuf_ptr = static_cast<char*>(recvbuf);

    const int log_p = std::log2(size);
    std::vector<int> level_size(log_p);
    int rem_size = count;
    const int logical_rank = PhysicalRankToLogicalRank(rank, lpo2_diff);
    for (int it = log_p - 1; it >= 0; --it) {
      const int logical_target = logical_rank ^ 1 << it;
      const int phys_target = LogicalRankToPhysicalRank(logical_target, lpo2_diff);
      const bool left_target = logical_target < logical_rank;

      level_size[it] = rem_size;
      const int right_count = rem_size / 2;
      const int left_count = rem_size - right_count;

      if (left_target) {
        SendrecvNonZeroed(recvbuf_ptr,
                          left_count,
                          datatype,
                          phys_target,
                          Tag_Const::ALLREDUCE,
                          tmpbuf.get(),
                          right_count,
                          datatype,
                          phys_target,
                          Tag_Const::ALLREDUCE,
                          comm,
                          MPI_STATUS_IGNORE);

        recvbuf_ptr += left_count * datatype_size;
        MPI_Reduce_local(tmpbuf.get(),
                         recvbuf_ptr, right_count, datatype, op);
        rem_size -= left_count;
      } else {
        SendrecvNonZeroed(recvbuf_ptr + left_count * datatype_size,
                          right_count,
                          datatype,
                          phys_target,
                          Tag_Const::ALLREDUCE,
                          tmpbuf.get(),
                          left_count,
                          datatype,
                          phys_target,
                          Tag_Const::ALLREDUCE,
                          comm,
                          MPI_STATUS_IGNORE);

        MPI_Reduce_local(tmpbuf.get(),
                         recvbuf_ptr, left_count, datatype, op);
        rem_size -= right_count;
      }
    }

    /* Perform allgather operation */

    for (int it = 0; it != log_p; ++it) {
      int logical_target = logical_rank ^ 1 << it;
      const int phys_target = LogicalRankToPhysicalRank(logical_target, lpo2_diff);
      bool left_target = logical_target < logical_rank;
      const int target_size = level_size[it] - rem_size;

      if (left_target) {
        SendrecvNonZeroed(recvbuf_ptr,
                          rem_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          recvbuf_ptr - target_size * datatype_size,
                          target_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          comm,
                          MPI_STATUS_IGNORE);
        recvbuf_ptr -= target_size * datatype_size;
      } else {
        SendrecvNonZeroed(recvbuf_ptr,
                          rem_size,
                          datatype,
                          phys_target,
                          Tag_Const::ALLGATHER,
                          recvbuf_ptr + rem_size * datatype_size,
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
  ScatterToPowOfTwoProcs(recvbuf, count, rank, size, datatype, comm);

  return 0;
}


/*
 * AllreduceHypercube: Allgather algorithm with running time
 * O(alpha * log p + beta n log p).  p must not be a power of two!!!
 */
double AllreduceHypercubeExpRunningTime(Comm const& comm, int sendcount,
                                        MPI_Datatype sendtype) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);

  int n = sendcount * datatype_size;               // In bytes

  double algo = (kALPHA + kBETA * n) * std::floor(std::log2(size));
  // We expect that the reduction takes beta/3 time.
  double calc = 1. / 3. * kBETA * n * std::floor(std::log2(size));

  if (!tlx::is_power_of_two(size)) {
    // Time to move data from/to non power of two nodes.
    algo += 2. * kBETA * n;
    calc += 1. / 3. * kBETA * n;
  }

  return algo + calc;
}

int AllreduceHypercube(const void* sendbuf, void* recvbuf,
                       int count, MPI_Datatype datatype,
                       MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, count,
                         datatype, op, comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  if (count == 0) return 0;

  // Move local input to output buffer.
  memcpy(recvbuf, sendbuf, recv_size);

  if (size == 1) return 0;

  std::unique_ptr<char[]> tmpbuf = std::make_unique<char[]>(recv_size);

  /* Handle power of two case */

  AnyProcCountToPowOfTwoReducer(recvbuf, tmpbuf.get(), count, datatype,
                                op, comm);

  const int lpo2 = tlx::round_down_to_power_of_two(size);
  const int lpo2_diff = size - lpo2;

  // We continue the algorithm with lpo2 number of processes.
  if ((rank < lpo2_diff * 2 && (rank & 1)) ||
      rank >= 2 * lpo2_diff) {
    /* Perform reduce-scatter operation */

    const int log_p = std::log2(size);
    const int logical_rank = PhysicalRankToLogicalRank(rank, lpo2_diff);
    for (int it = log_p - 1; it >= 0; --it) {
      const int logical_target = logical_rank ^ 1 << it;
      const int phys_target = LogicalRankToPhysicalRank(logical_target, lpo2_diff);
      const bool left_target = logical_target < logical_rank;

      if (left_target) {
        Sendrecv(recvbuf,
                 count,
                 datatype,
                 phys_target,
                 Tag_Const::ALLREDUCE,
                 tmpbuf.get(),
                 count,
                 datatype,
                 phys_target,
                 Tag_Const::ALLREDUCE,
                 comm,
                 MPI_STATUS_IGNORE);

        MPI_Reduce_local(tmpbuf.get(),
                         recvbuf, count, datatype, op);
      } else {
        Sendrecv(recvbuf,
                 count,
                 datatype,
                 phys_target,
                 Tag_Const::ALLREDUCE,
                 tmpbuf.get(),
                 count,
                 datatype,
                 phys_target,
                 Tag_Const::ALLREDUCE,
                 comm,
                 MPI_STATUS_IGNORE);

        MPI_Reduce_local(tmpbuf.get(),
                         recvbuf, count, datatype, op);
      }
    }
  }

  /* Handle non power of two case */
  ScatterToPowOfTwoProcs(recvbuf, count, rank, size, datatype, comm);

  return 0;
}

/*
 *
 * Blocking allgather with equal amount of elements on each process
 * This method uses different implementations depending on the
 * size of comm and the input size.
 */
int Allreduce(const void* sendbuf, void* recvbuf,
              int count, MPI_Datatype datatype,
              MPI_Op op, Comm const& comm) {
  double sc_allg = AllreduceScatterAllgatherExpRunningTime(comm, count, datatype);
  double hypercube = AllreduceHypercubeExpRunningTime(comm, count, datatype);

  if (hypercube < sc_allg) {
    return AllreduceHypercube(sendbuf, recvbuf, count, datatype, op, comm);
  } else {
    return AllreduceScatterAllgather(sendbuf, recvbuf, count, datatype, op, comm);
  }
}
}         // namespace optimized

/*
 * Request for the reduce
 */
class IallreduceReq : public RequestSuperclass {
 public:
  IallreduceReq(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                int tag, MPI_Op op, Comm const& comm);
  ~IallreduceReq();
  int test(int* flag, MPI_Status* status);

 private:
  bool m_first_round;
  Request m_reduce_request;
  Request m_bcast_request;
  const void* m_sendbuf;
  void* m_recvbuf;
  int m_count, m_tag;
  MPI_Datatype m_datatype;
  MPI_Op m_op;
  Comm m_comm;
  bool m_mpi_collective;
  MPI_Request m_mpi_req;
};
}  // namespace _internal

int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IallreduceReq>(sendbuf, recvbuf,
                                                          count, datatype, tag, op, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IallreduceReq::IallreduceReq(const void* sendbuf, void* recvbuf, int count,
                                             MPI_Datatype datatype, int tag, MPI_Op op,
                                             RBC::Comm const& comm) :
  m_first_round(true),
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_count(count),
  m_tag(tag),
  m_datatype(datatype),
  m_op(op),
  m_comm(comm),
  m_mpi_collective(false) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (m_comm.useMPICollectives()) {
    MPI_Iallreduce(const_cast<void*>(m_sendbuf), m_recvbuf, m_count, m_datatype,
                   m_op, m_comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif

  const int root = 0;
  RBC::Ireduce(m_sendbuf, m_recvbuf, m_count, m_datatype, m_op, root, m_comm, &m_reduce_request);
}

RBC::_internal::IallreduceReq::~IallreduceReq() { }

int RBC::_internal::IallreduceReq::test(int* flag, MPI_Status* status) {
  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (m_first_round) {
    RBC::Test(&m_reduce_request, flag, status);
    if (*flag) {
      RBC::Ibcast(m_recvbuf, m_count, m_datatype, 0, m_comm, &m_bcast_request);
      m_first_round = false;
    }
    *flag = 0;
  } else {
    RBC::Test(&m_bcast_request, flag, status);
  }

  return 0;
}
