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

#include "RBC.hpp"
#include "tlx/math.hpp"

namespace RBC {
int Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
           MPI_Op op, int root, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Reduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, root, comm.get());
  }

  const int tag = Tag_Const::REDUCE;
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);
  const size_t recv_size = static_cast<size_t>(count) * datatype_size;

  int root_rank = (rank - root + size) % size;

  if (size == 1) {
    std::memcpy(recvbuf, sendbuf, recv_size);
    return 0;
  }

  std::unique_ptr<char[]> recvbuf_arr = std::make_unique<char[]>(2 * recv_size);
  std::unique_ptr<char[]> tmp_arr = std::make_unique<char[]>(recv_size);

  std::memcpy(tmp_arr.get(), sendbuf, recv_size);

  // Perform first receive operation if appropriate.
  int src = root_rank ^ 1;
  bool second_buffer = false;
  bool is_recved = false;
  // Level of the tree. We start with level 1 as we have already executed level 0 if
  // appropriate.
  int i = 0;
  if (src > root_rank) {
    if (src < size) {
      int root_src = (src + root) % size;
      RBC::Recv(recvbuf_arr.get(), count, datatype, root_src, tag,
                comm, MPI_STATUS_IGNORE);
      is_recved = true;
      second_buffer = true;
    }
    i++;
  }

  MPI_Request request;
  while ((root_rank ^ (1 << i)) > root_rank) {
    src = root_rank ^ (1 << i);
    i++;
    if (src < size) {
      int root_src = (src + root) % size;
      // Receive and reduce at the same time -> overlapping.
      RBC::Irecv(recvbuf_arr.get() + recv_size * static_cast<size_t>(second_buffer),
                 count, datatype, root_src, tag, comm, &request);
      MPI_Reduce_local(recvbuf_arr.get() + recv_size * static_cast<size_t>(!second_buffer),
                       tmp_arr.get(), count, datatype, op);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
      second_buffer = !second_buffer;
    }
  }

  if (is_recved) {
    MPI_Reduce_local(recvbuf_arr.get() + recv_size * static_cast<size_t>(!second_buffer),
                     tmp_arr.get(), count, datatype, op);
  }

  // Send data
  if (root_rank > 0) {
    int dest = root_rank ^ (1 << i);
    int root_dest = (dest + root) % size;
    Send(tmp_arr.get(), count, datatype, root_dest, tag, comm);
  } else {
    std::memcpy(recvbuf, tmp_arr.get(), recv_size);
  }

  return 0;
}

namespace _internal {
/*
 * Request for the reduce
 */
class IreduceReq : public RequestSuperclass {
 public:
  IreduceReq(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
             int tag, MPI_Op op, int root, Comm const& comm);
  ~IreduceReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  int m_count, m_tag, m_root, m_rank, m_size, m_new_rank, m_height, m_own_height,
    m_datatype_size, m_receives;
  size_t m_recv_size;
  MPI_Datatype m_datatype;
  MPI_Op m_op;
  Comm m_comm;
  bool m_send, m_completed, m_mpi_collective;
  std::unique_ptr<char[]> m_recvbuf_arr, m_reduce_buf;
  Request m_send_req;
  std::vector<Request> m_recv_requests;
  MPI_Request m_mpi_req;
};
}  // namespace _internal

int Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IreduceReq>(sendbuf, recvbuf,
                                                       count, datatype, tag, op, root, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IreduceReq::IreduceReq(const void* sendbuf, void* recvbuf, int count,
                                       MPI_Datatype datatype, int tag, MPI_Op op, int root,
                                       RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_count(count),
  m_tag(tag),
  m_root(root),
  m_datatype(datatype),
  m_op(op),
  m_comm(comm),
  m_send(false),
  m_completed(false),
  m_mpi_collective(false),
  m_recvbuf_arr(nullptr),
  m_reduce_buf(nullptr) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm.get(),
                &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif
  RBC::Comm_rank(comm, &m_rank);
  RBC::Comm_size(comm, &m_size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  m_datatype_size = static_cast<int>(type_size);
  m_recv_size = static_cast<size_t>(count) * m_datatype_size;
  m_new_rank = (m_rank - root - 1 + m_size) % m_size;
  m_height = tlx::integer_log2_ceil(m_size);
  m_own_height = 0;
  if (m_new_rank == (m_size - 1)) {
    m_own_height = m_height;
  } else {
    for (int i = 0; ((m_new_rank >> i) % 2 == 1) && (i < m_height); i++)
      m_own_height++;
  }

  m_recvbuf_arr = std::make_unique<char[]>(m_recv_size * m_own_height);
  m_reduce_buf = std::make_unique<char[]>(m_recv_size);
  std::memcpy(m_reduce_buf.get(), sendbuf, m_recv_size);
}

RBC::_internal::IreduceReq::~IreduceReq() { }

int RBC::_internal::IreduceReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (m_height > 0 && m_recv_requests.size() == 0) {
    // Receive data
    int tmp_rank = m_new_rank;
    if (m_new_rank == m_size - 1)
      tmp_rank = std::pow(2, m_height) - 1;

    for (int i = m_own_height - 1; i >= 0; i--) {
      int tmp_src = tmp_rank - std::pow(2, i);
      if (tmp_src < m_new_rank) {
        m_recv_requests.push_back(RBC::Request());
        int src = (tmp_src + m_root + 1) % m_size;
        RBC::Irecv(m_recvbuf_arr.get() + (m_recv_requests.size() - 1) * m_recv_size, m_count,
                   m_datatype, src,
                   m_tag, m_comm, &m_recv_requests.back());
      } else {
        tmp_rank = tmp_src;
      }
    }
    m_receives = m_recv_requests.size();
  }

  if (!m_send) {
    int recv_finished;
    RBC::Testall(m_recv_requests.size(), &m_recv_requests.front(), &recv_finished,
                 MPI_STATUSES_IGNORE);
    if (recv_finished && m_receives > 0) {
      // Reduce received data and local data
      for (int i = 0; i < (m_receives - 1); i++) {
        MPI_Reduce_local(m_recvbuf_arr.get() + i * m_recv_size,
                         m_recvbuf_arr.get() + (i + 1) * m_recv_size, m_count, m_datatype, m_op);
      }
      MPI_Reduce_local(m_recvbuf_arr.get() + (m_receives - 1) * m_recv_size,
                       m_reduce_buf.get(), m_count, m_datatype, m_op);
    }

    // Send data
    if (recv_finished) {
      if (m_new_rank < m_size - 1) {
        int tmp_dest = m_new_rank + std::pow(2, m_own_height);
        if (tmp_dest > m_size - 1)
          tmp_dest = m_size - 1;
        int dest = (tmp_dest + m_root + 1) % m_size;
        RBC::Isend(m_reduce_buf.get(), m_count, m_datatype, dest, m_tag, m_comm, &m_send_req);
      }
      m_send = true;
    }
  }
  if (m_send) {
    if (m_new_rank == m_size - 1) {
      std::memcpy(m_recvbuf, m_reduce_buf.get(), m_recv_size);
      *flag = 1;
    } else {
      RBC::Test(&m_send_req, flag, MPI_STATUS_IGNORE);
    }
    if (*flag)
      m_completed = true;
  }
  return 0;
}
