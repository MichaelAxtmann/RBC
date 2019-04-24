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
#include <cmath>
#include <cstring>
#include <iostream>

#include "RBC.hpp"
#include "tlx/math.hpp"

namespace RBC {
int Gatherm(const void* sendbuf, int sendcount,
            MPI_Datatype sendtype, void* recvbuf, int recvcount,
            int root,
            std::function<void(void*, void*, void*, void*, void*)> op,
            Comm const& comm) {
  if (recvcount == 0) {
    return 0;
  }

  int tag = Tag_Const::GATHERM;
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
  MPI_Type_get_extent(sendtype, &lb, &send_size);
  recv_size = send_size;
  int recvtype_size = static_cast<int>(recv_size);
  int sendtype_size = static_cast<int>(send_size);

  recv_size = recvcount * recvtype_size;
  char* recv_buf = nullptr;
  if (rank == root) {
    assert(recvbuf != nullptr);
    recv_buf = static_cast<char*>(recvbuf);
  } else {
    recv_buf = new char[recv_size];
  }
  std::unique_ptr<char[]> merge_buf(new char[recv_size]);

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
      if (count > 0) {
        op(recv_buf,
           recv_buf + received * sendtype_size,
           recv_buf + received * sendtype_size,
           recv_buf + (received + count) * sendtype_size,
           merge_buf.get());
        memcpy(recv_buf, merge_buf.get(), (received + count) * sendtype_size);
      }
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
    assert(recvcount == received);
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

namespace _internal {
/*
 * Request for the gatherm
 */
class IgathermReq : public RequestSuperclass {
 public:
  IgathermReq(const void* sendbuf, int sendcount,
              MPI_Datatype sendtype, void* recvbuf, int recvcount,
              MPI_Datatype recvtype, int root, int tag,
              std::function<void(void*, void*, void*, void*, void*)> op,
              Comm const& comm);
  ~IgathermReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  std::unique_ptr<char[]> m_mergebuf;
  int m_sendcount, m_recvcount, m_root, m_tag, m_own_height, m_size, m_rank, m_height,
    m_received, m_count, m_total_recvcount, m_new_rank, m_sendtype_size,
    m_recvtype_size, m_recv_size;
  MPI_Datatype m_sendtype, m_recvtype;
  std::function<void(void*, void*, void*, void*, void*)> m_op;
  Comm m_comm;
  bool m_receive, m_send, m_completed, m_mpi_collective;
  char* m_recv_buf;
  Request m_recv_req, m_send_req;
  MPI_Request m_mpi_req;
};
}  // namespace _internal

int Igatherm(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
             void* recvbuf, int recvcount, int root,
             std::function<void(void*, void*, void*, void*, void*)> op, Comm const& comm,
             Request* request, int tag) {
  request->set(std::make_shared<RBC::_internal::IgathermReq>(sendbuf, sendcount,
                                                             sendtype, recvbuf,
                                                             recvcount, sendtype,
                                                             root, tag,
                                                             op, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IgathermReq::IgathermReq(const void* sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                         MPI_Datatype recvtype,
                                         int root, int tag,
                                         std::function<void(void*, void*, void*, void*, void*)> op,
                                         RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_mergebuf(nullptr),
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
  m_op(op),
  m_comm(comm),
  m_receive(false),
  m_send(false),
  m_completed(false),
  m_mpi_collective(false),
  m_recv_buf(nullptr) {
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

  m_total_recvcount = recvcount;
  recv_size = m_total_recvcount * m_recvtype_size;
  if (m_rank == root) {
    assert(recvbuf != nullptr);
    m_recv_buf = static_cast<char*>(recvbuf);
  } else {
    m_recv_buf = new char[recv_size];
  }
  // Copy send data into receive buffer
  std::memcpy(m_recv_buf, sendbuf, sendcount * m_sendtype_size);
  m_received = sendcount;

  m_mergebuf.reset(new char[recv_size]);
}

RBC::_internal::IgathermReq::~IgathermReq() {
  if ((m_rank != m_root) && m_recv_buf != nullptr)
    delete[] m_recv_buf;
}

int RBC::_internal::IgathermReq::test(int* flag, MPI_Status* status) {
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
        m_op(m_recv_buf,
             m_recv_buf + m_received * m_sendtype_size,
             m_recv_buf + m_received * m_sendtype_size,
             m_recv_buf + (m_received + m_count) * m_sendtype_size,
             m_mergebuf.get());
        memcpy(m_recv_buf, m_mergebuf.get(), (m_received + m_count) * m_sendtype_size);
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
    } else {
      if (!m_send) {
        // Start non-blocking send to parent node
        int tmp_dest = m_new_rank - std::pow(2, m_height - 1);
        int dest = (tmp_dest + m_root) % m_size;
        RBC::Isend(m_recv_buf, m_received, m_sendtype, dest, m_tag, m_comm, &m_send_req);
        //                std::cout << W(rank) << W(dest) << W(received) << W(total_recvcount) << std::endl;
        m_send = true;
      }
      // Gatherm is completed when the send is finished
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
