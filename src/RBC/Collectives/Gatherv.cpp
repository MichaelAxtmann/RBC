/*****************************************************************************
 * this file is part of the Project RBC
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

namespace RBC {
int Gatherv(const void* sendbuf, int sendcount,
            MPI_Datatype sendtype, void* recvbuf,
            const int* recvcounts, const int* displs, MPI_Datatype recvtype,
            int root, Comm const& comm) {
#ifndef NO_IBAST
  if (comm.useMPICollectives()) {
    return MPI_Gatherv(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, const_cast<int*>(recvcounts),
                       const_cast<int*>(displs), recvtype, root, comm.get());
  }
#endif

  int tag = Tag_Const::GATHERV;
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

  int total_recvcount = 0;
  total_recvcount = 0;
  for (int i = 0; i < size; i++) {
    total_recvcount += recvcounts[i];
  }
  recv_size = total_recvcount * recvtype_size;
  char* recv_buf = nullptr;
  if (rank == root && recvcounts == nullptr) {
    assert(recvbuf != nullptr);
    recv_buf = static_cast<char*>(recvbuf);
  } else {
    recv_buf = new char[recv_size];
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
      // Range::Test if message can be received
      MPI_Status status;
      Probe(src, tag, comm, &status);
      // Receive message
      int count;
      MPI_Get_count(&status, sendtype, &count);
      Recv(recv_buf + received * sendtype_size, count,
           sendtype, src, tag, comm, MPI_STATUS_IGNORE);
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
    if (recvcounts != nullptr) {
      char* buf = static_cast<char*>(recvbuf);
      char* recv_ptr = recv_buf;
      for (int i = 0; i < size; i++) {
        std::memcpy(buf + displs[i] * sendtype_size, recv_ptr,
                    recvcounts[i] * sendtype_size);
        recv_ptr += recvcounts[i] * sendtype_size;
      }
    }
  } else {
    // Send to parent node
    int tmp_dest = new_rank - std::pow(2, height - 1);
    int dest = (tmp_dest + root) % size;
    Send(recv_buf, received, sendtype, dest, tag, comm);
  }

  if (rank != root || recvcounts != nullptr) {
    delete[] recv_buf;
  }
  return 0;
}

namespace _internal {
/*
 * Request for the gatherv
 */
class IgathervReq : public RequestSuperclass {
 public:
  IgathervReq(const void* sendbuf, int sendcount,
              MPI_Datatype sendtype, void* recvbuf, int recvcount,
              const int* recvcounts, const int* displs, MPI_Datatype recvtype,
              int root, int tag, Comm const& comm);
  ~IgathervReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  const int* m_recvcounts, * m_displs;
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

int Igatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
             void* recvbuf, const int* recvcounts, const int* displs,
             MPI_Datatype recvtype, int root, Comm const& comm,
             Request* request, int tag) {
  request->set(std::make_shared<_internal::IgathervReq>(sendbuf, sendcount,
                                                        sendtype, recvbuf, -1,
                                                        recvcounts, displs,
                                                        recvtype, root, tag,
                                                        comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IgathervReq::IgathervReq(const void* sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                         const int* recvcounts, const int* displs, MPI_Datatype recvtype,
                                         int root, int tag, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_recvcounts(recvcounts),
  m_displs(displs),
  m_sendcount(sendcount),
  m_recvcount(recvcount),
  m_root(root),
  m_tag(tag),
  m_own_height(0),
  m_size(0),
  m_rank(0),
  m_height(1),
  m_received(0),
  m_count(0),
  m_sendtype(sendtype),
  m_recvtype(recvtype),
  m_comm(comm),
  m_receive(false),
  m_send(false),
  m_completed(false),
  m_mpi_collective(false),
  m_recv_buf(nullptr) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                 recvcounts, displs, recvtype, root, comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif

  RBC::Comm_rank(comm, &m_rank);
  RBC::Comm_size(comm, &m_size);

  m_new_rank = (m_rank - root + m_size) % m_size;
  int max_height = tlx::integer_log2_ceil(m_size);
  if (std::pow(2, max_height) < m_size)
    max_height++;
  for (int i = 0; ((m_new_rank >> i) % 2 == 0) && (i < max_height); i++)
    m_own_height++;

  MPI_Aint lb, recv_size, send_size;
  MPI_Type_get_extent(recvtype, &lb, &recv_size);
  MPI_Type_get_extent(sendtype, &lb, &send_size);
  m_recvtype_size = static_cast<int>(recv_size);
  m_sendtype_size = static_cast<int>(send_size);

  m_total_recvcount = 0;
  for (int i = 0; i < m_size; i++)
    m_total_recvcount += recvcounts[i];

  recv_size = m_total_recvcount * m_recvtype_size;
  if (m_rank == root && recvcounts == nullptr) {
    assert(recvbuf != nullptr);
    m_recv_buf = static_cast<char*>(recvbuf);
  } else {
    m_recv_buf = new char[recv_size];
  }
  // Copy send data into receive buffer
  std::memcpy(m_recv_buf, sendbuf, sendcount * m_sendtype_size);
  m_received = sendcount;
}

RBC::_internal::IgathervReq::~IgathervReq() {
  if ((m_rank != m_root || m_recvcounts != nullptr) && m_recv_buf != nullptr)
    delete[] m_recv_buf;
}

int RBC::_internal::IgathervReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  // If messages have to be received
  if (m_height <= m_own_height) {
    if (!m_receive) {
      int tmp_src = m_new_rank + std::pow(2, m_height - 1);
      if (tmp_src < m_size) {
        int src = (tmp_src + m_root) % m_size;
        // Range::Test if message can be received
        MPI_Status probe_status;
        int ready;
        RBC::Iprobe(src, m_tag, m_comm, &ready, &probe_status);
        if (ready) {
          // Receive message with non-blocking receive
          MPI_Get_count(&probe_status, m_sendtype, &m_count);
          RBC::Irecv(m_recv_buf + m_received * m_sendtype_size, m_count,
                     m_sendtype, src, m_tag, m_comm, &m_recv_req);
          m_receive = true;
        }
      } else {
        // Source rank larger than comm size
        m_height++;
      }
    }
    if (m_receive) {
      // Range::Test if receive finished
      int finished;
      RBC::Test(&m_recv_req, &finished, MPI_STATUS_IGNORE);
      if (finished) {
        // Merge the received data
        m_received += m_count;
        m_height++;
        m_receive = false;
      }
    }
  }

  // If all messages have been received
  if (m_height > m_own_height) {
    if (m_rank == m_root) {
      // root doesn't send to anyone
      m_completed = true;
      //            if (total_recvcount != received)
      //            std::cout << W(rank) << W(size) << W(total_recvcount) << W(received) << std::endl;
      assert(m_total_recvcount == m_received);
      if (m_recvcounts != nullptr) {
        char* buf = static_cast<char*>(m_recvbuf);
        char* recv_ptr = m_recv_buf;
        for (int i = 0; i < m_size; i++) {
          std::memcpy(buf + m_displs[i] * m_sendtype_size, recv_ptr,
                      m_recvcounts[i] * m_sendtype_size);
          recv_ptr += m_recvcounts[i] * m_sendtype_size;
        }
      }
    } else {
      if (!m_send) {
        // Start non-blocking send to parent node
        int tmp_dest = m_new_rank - std::pow(2, m_height - 1);
        int dest = (tmp_dest + m_root) % m_size;
        RBC::Isend(m_recv_buf, m_received, m_sendtype, dest, m_tag, m_comm, &m_send_req);
        //                std::cout << W(rank) << W(dest) << W(received) << W(total_recvcount) << std::endl;
        m_send = true;
      }
      // Gatherv is completed when the send is finished
      int finished;
      RBC::Test(&m_send_req, &finished, MPI_STATUS_IGNORE);
      if (finished)
        m_completed = true;
    }
  }

  if (m_completed)
    *flag = 1;
  return 0;
}
