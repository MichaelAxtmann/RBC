/*****************************************************************************
 * this file is part of the Project RBCn

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
#include <cmath>
#include <cstring>
#include <iostream>

#include "RBC.hpp"
#include "tlx/algorithm.hpp"
#include "tlx/math.hpp"


int Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
           void* recvbuf, int recvcount, MPI_Datatype recvtype,
           int root, RBC::Comm& comm) {
#ifndef NO_IBAST
  if (comm.useMPICollectives()) {
    return MPI_Gather(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                      root, comm.get());
  }
#endif

  int tag = RBC::Tag_Const::GATHER;
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  int new_rank = (rank - root + size) % size;
  int max_height = tlx::integer_log2_ceil(size);
  int own_height = 0;
  if (std::pow(2, max_height) < size)
    max_height++;
  for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
    own_height++;

  MPI_Aint lb, recv_size, send_size;
  MPI_Type_get_extent(recvtype, &lb, &recv_size);
  MPI_Type_get_extent(sendtype, &lb, &send_size);
  int recvtype_size = static_cast<int>(recv_size);
  int sendtype_size = static_cast<int>(send_size);

  int subtree_size = static_cast<int>(std::min(std::pow(2, own_height),
                                               static_cast<double>(size)));
  int total_recvcount = 0;
  assert(sendcount == recvcount);
  assert(sendtype_size == recvtype_size);
  total_recvcount = subtree_size * recvcount;
  recv_size = total_recvcount * recvtype_size;
  char* recv_buf = static_cast<char*>(recvbuf);
  if (rank != root) {
    recv_buf = static_cast<char*>(recvbuf);
  }

  // Copy send data into receive buffer
  std::memcpy(recv_buf, sendbuf, sendcount * sendtype_size);
  int received = sendcount;
  int height = 1;

  // If messages have to be received
  while (height <= own_height) {
    int tmp_src = new_rank + std::pow(2, height - 1);
    if (tmp_src < size) {
      int src = (tmp_src + root) % size;
      // Receive message
      MPI_Status status;
      Recv(recv_buf + received * sendtype_size, recvcount - received,
           sendtype, src, tag, comm, &status);
      int count;
      MPI_Get_count(&status, sendtype, &count);
      // Merge the received data
      received += count;
      height++;
    } else {
      // Source rank larger than comm size
      height++;
    }
  }

  // When all messages have been received
  if (rank == root) {
    // root doesn't send to anyone
    assert(total_recvcount == received);
  } else {
    // Send to parent node
    int tmp_dest = new_rank - std::pow(2, height - 1);
    int dest = (tmp_dest + root) % size;
    Send(recv_buf, received, sendtype, dest, tag, comm);
  }

  if (rank != root) {
    delete[] recv_buf;
  }
  return 0;
}

namespace RBC {
namespace _internal {
/*
 * Request for the gather
 */
class IgatherReq : public RequestSuperclass {
 public:
  IgatherReq(const void* sendbuf, int sendcount,
             MPI_Datatype sendtype, void* recvbuf, int recvcount,
             MPI_Datatype recvtype,
             int root, int tag,
             Comm const& comm);
  ~IgatherReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  int m_sendcount, m_recvcount, m_root, m_tag, m_own_height, m_size, m_rank, m_height,
    m_received, m_count, m_total_recvcount, m_new_rank, m_sendtype_size,
    m_recvtype_size, m_recv_size;
  MPI_Datatype m_sendtype, m_recvtype;
  Comm m_comm;
  bool m_receive, m_send, m_completed, m_mpi_collective;
  char* m_recv_buf;
  Request m_recv_req, m_send_req;
  MPI_Request m_mpi_req;
};
}  // namespace _internal
}  // namespace RBC
