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
#include <memory>

#include "RBC.hpp"
#include "tlx/math.hpp"

#include "BinaryTree.hpp"

namespace RBC {
int ScanAndBcast(const void* sendbuf, void* recvbuf_scan,
                 void* recvbuf_bcast, int count, MPI_Datatype datatype,
                 MPI_Op op, const Comm& comm) {
  if (comm.useMPICollectives()) {
    MPI_Scan(const_cast<void*>(sendbuf), recvbuf_scan, count, datatype, op, comm.get());

    int rank, size;
    Comm_rank(comm, &rank);
    Comm_size(comm, &size);
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    int datatype_size = static_cast<int>(type_size);
    int recv_size = count * datatype_size;
    std::memcpy(recvbuf_bcast, recvbuf_scan, recv_size);
    MPI_Bcast(recvbuf_bcast, count, datatype, size - 1, comm.get());
  }

  const int rank = comm.getRank();
  const int size = comm.getSize();
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);
  const int recv_size = count * datatype_size;

  if (comm.getSize() == 1) {
    std::memcpy(recvbuf_scan, sendbuf, recv_size);
    std::memcpy(recvbuf_bcast, sendbuf, recv_size);
    return 0;
  }

  const int tag = Tag_Const::SCANANDBCAST;

  int lchild = -1;
  int rchild = -1;
  int parent = -1;

  _internal::BinaryTree::Create(rank, size, &lchild, &rchild, &parent);

  std::unique_ptr<char[]> left_tree(new char[recv_size]);

  Request requests[4];
  int is_left_receiving = 0;
  int is_right_receiving = 0;

  std::memcpy(recvbuf_scan, sendbuf, recv_size);

  if (lchild != -1) {
    RBC::Irecv(left_tree.get(), count, datatype, lchild, tag, comm, requests);
    is_left_receiving = 1;
  }
  if (rchild != -1) {
    RBC::Irecv(recvbuf_bcast, count, datatype, rchild, tag, comm,
               requests + is_left_receiving);
    is_right_receiving = 1;
  }
  // var: up_recv
  RBC::Waitall(is_left_receiving + is_right_receiving, requests, MPI_STATUSES_IGNORE);

  if (is_left_receiving) {
    MPI_Reduce_local(left_tree.get(), recvbuf_scan, count, datatype, op);
  }
  if (is_right_receiving) {
    MPI_Reduce_local(recvbuf_scan, recvbuf_bcast, count, datatype, op);
  } else {
    std::memcpy(recvbuf_bcast, recvbuf_scan, recv_size);
  }

  // recvbuf_scan := own + left_subtree
  // left_subtree := left_subtree
  // recvbuf_bcast := sum of subtrees and own

  if (parent != -1) {
    // var: up_send
    RBC::Send(recvbuf_bcast, count, datatype, parent, tag, comm);
  }

  int recv_cnt = 0;

  // Receive reduces values of processes left to our
  // subtree. There are not processes left to our subtree if we
  // are a process of the leftmost front of the binary tree.
  if (parent != -1 && !tlx::is_power_of_two(rank + 1)) {
    // var: down_recv
    RBC::Irecv(left_tree.get(), count, datatype, parent, tag, comm, requests);
    // MPI_Reduce_local(left_tree.get(), recvbuf_scan, count, datatype, op);
    ++recv_cnt;
  }

  // recvbuf_scan := own + left_subtree
  // left_subtree := processes left to our subtree
  // recvbuf_bcast := sum of subtrees and own

  if (parent != -1) {
    RBC::Irecv(recvbuf_bcast, count, datatype, parent, tag, comm, requests + recv_cnt);
    ++recv_cnt;
  }

  // recvbuf_scan := own + left_subtree
  // left_subtree := processes left to our subtree
  // recvbuf_bcast := total sum

  RBC::Waitall(recv_cnt, requests, MPI_STATUSES_IGNORE);

  if (parent != -1 && !tlx::is_power_of_two(rank + 1)) {
    MPI_Reduce_local(left_tree.get(), recvbuf_scan, count, datatype, op);
  }

  int send_cnt = 0;
  // Receive reduces values of processes left to our
  // subtree. There are not processes left to our subtree if we
  // are a process of the leftmost front of the binary tree.
  if (is_left_receiving && !tlx::is_power_of_two(rank + 1) && parent != -1) {
    RBC::Isend(left_tree.get(), count, datatype, lchild, tag, comm,
               requests);
    ++send_cnt;
  }

  if (is_right_receiving) {
    RBC::Isend(recvbuf_scan, count, datatype, rchild, tag, comm,
               requests + send_cnt);
    ++send_cnt;
  }

  if (is_left_receiving) {
    RBC::Isend(recvbuf_bcast, count, datatype, lchild, tag, comm,
               requests + send_cnt);
    ++send_cnt;
  }

  if (is_right_receiving) {
    RBC::Isend(recvbuf_bcast, count, datatype, rchild, tag, comm,
               requests + send_cnt);
    ++send_cnt;
  }

  // var: down_send
  RBC::Waitall(send_cnt, requests, MPI_STATUSES_IGNORE);

  return 0;
}

namespace _internal {
// todo: Rewrite nonblocking algorithm and use Scan.cpp as a template (more efficient)!
/*
 * Request for the scan and broadcast
 */
class IscanAndBcastReq : public RequestSuperclass {
 public:
  IscanAndBcastReq(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast,
                   int count, MPI_Datatype datatype, int tag, MPI_Op op, Comm const& comm);
  ~IscanAndBcastReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf_scan, * m_recvbuf_bcast;
  int m_count, m_tag, m_rank, m_size, m_height, m_up_height, m_down_height, m_receives,
    m_sends, m_recv_size, m_datatype_size, m_up_heigth_cnt, m_tmp_rank, m_cur_height;
  MPI_Datatype m_datatype;
  MPI_Op m_op;
  Comm m_comm;
  std::vector<int> m_target_ranks;
  std::unique_ptr<char[]> m_recvbuf_arr, m_tmp_buf, m_scan_buf;
  char* m_tmp_buf2, * m_bcast_buf;
  bool m_upsweep, m_downsweep, m_send, m_send_up, m_completed, m_recv, m_downsweep_recvd;
  Request m_send_req, m_recv_req;
};
}  // namespace _internal

int IscanAndBcast(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast,
                  int count, MPI_Datatype datatype, MPI_Op op, Comm const& comm,
                  Request* request, int tag) {
  request->set(std::make_shared<_internal::IscanAndBcastReq>(sendbuf,
                                                             recvbuf_scan,
                                                             recvbuf_bcast,
                                                             count,
                                                             datatype,
                                                             tag,
                                                             op,
                                                             comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IscanAndBcastReq::IscanAndBcastReq(const void* sendbuf, void* recvbuf_scan,
                                                   void* recvbuf_bcast, int count,
                                                   MPI_Datatype datatype, int tag, MPI_Op op,
                                                   RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf_scan(recvbuf_scan),
  m_recvbuf_bcast(recvbuf_bcast),
  m_count(count),
  m_tag(tag),
  m_receives(0),
  m_sends(0),
  m_datatype(datatype),
  m_op(op),
  m_comm(comm),
  m_upsweep(true),
  m_downsweep(false),
  m_send(false),
  m_send_up(false),
  m_completed(false),
  m_recv(false),
  m_downsweep_recvd(false) {
  RBC::Comm_rank(comm, &m_rank);
  RBC::Comm_size(comm, &m_size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  m_datatype_size = static_cast<int>(type_size);
  m_recv_size = count * m_datatype_size;
  m_height = tlx::integer_log2_ceil(m_size);

  m_up_height = 0;
  if (m_rank == (m_size - 1)) {
    m_up_height = m_height;
  } else {
    for (int i = 0; ((m_rank >> i) % 2 == 1) && (i < m_height); i++)
      m_up_height++;
  }
  m_up_heigth_cnt = m_up_height - 1;
  m_tmp_rank = m_rank;
  if (m_rank == m_size - 1)
    m_tmp_rank = std::pow(2, m_height) - 1;

  m_recvbuf_arr = std::make_unique<char[]>(m_recv_size * (1 + m_up_height));
  m_tmp_buf = std::make_unique<char[]>(m_recv_size * 2);
  m_tmp_buf2 = m_tmp_buf.get() + m_recv_size;
  m_scan_buf = std::make_unique<char[]>(m_recv_size * 2);
  m_bcast_buf = m_scan_buf.get() + m_recv_size;
  m_down_height = 0;
  if (m_rank < m_size - 1)
    m_down_height = m_up_height + 1;

  std::memcpy(m_scan_buf.get(), sendbuf, m_recv_size);
}

RBC::_internal::IscanAndBcastReq::~IscanAndBcastReq() { }

int RBC::_internal::IscanAndBcastReq::test(int* flag, MPI_Status*  /*status*/) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }
  if (m_size == 1 && m_upsweep) {
    std::memcpy(m_recvbuf_scan, m_sendbuf,
                m_recv_size);
    std::memcpy(m_recvbuf_bcast, m_sendbuf,
                m_recv_size);
    m_upsweep = false;
    m_downsweep = false;
  }

  // upsweep phase
  if (m_upsweep) {
    if (m_up_heigth_cnt >= 0) {
      if (!m_recv) {
        int source = m_tmp_rank - std::pow(2, m_up_heigth_cnt);
        if (source < m_rank) {
          RBC::Irecv(m_recvbuf_arr.get() + m_recv_size * m_receives,
                     m_count, m_datatype, source,
                     m_tag, m_comm, &m_recv_req);
          // Save communication partner rank in vector
          m_target_ranks.push_back(source);
          m_recv = true;
          m_receives++;
        } else {
          // source rank is not in communicator (or own rank)
          m_tmp_rank = source;
          m_up_heigth_cnt--;
        }
      } else {
        int finished;
        RBC::Test(&m_recv_req, &finished, MPI_STATUS_IGNORE);
        if (finished) {
          assert(m_receives > 0);
          if (m_receives > 1) {
            MPI_Reduce_local(m_recvbuf_arr.get() + (m_receives - 2) * m_recv_size,
                             m_recvbuf_arr.get() + (m_receives - 1) * m_recv_size, m_count, m_datatype, m_op);
          }
          m_recv = false;
          m_up_heigth_cnt--;
        }
      }
    }

    // Everything received
    if (m_up_heigth_cnt < 0 && !m_send_up) {
      if (m_receives > 0) {
        MPI_Reduce_local(m_recvbuf_arr.get() + m_recv_size * (m_receives - 1),
                         m_scan_buf.get(), m_count, m_datatype, m_op);
      }

      // Send data
      if (m_rank < m_size - 1) {
        int dest = m_rank + std::pow(2, m_up_height);
        if (dest > m_size - 1)
          dest = m_size - 1;
        RBC::Isend(m_scan_buf.get(), m_count, m_datatype, dest, m_tag, m_comm, &m_send_req);
        m_target_ranks.push_back(dest);
      }
      m_send_up = true;
    }

    // End upsweep phase when data has been send
    if (m_send_up) {
      int finished = 1;
      if (m_rank < m_size - 1)
        RBC::Test(&m_send_req, &finished, MPI_STATUS_IGNORE);

      if (finished) {
        m_upsweep = false;
        m_downsweep = true;
        m_cur_height = m_height;
        if (m_rank == m_size - 1) {
          std::memcpy(m_bcast_buf, m_scan_buf.get(), m_recv_size);
          // set scan buf to "empty" elements
          for (int i = 0; i < m_recv_size; i++) {
            m_scan_buf[i] = 0;
          }
        }
      }
    }
  }

  // downsweep phase
  if (m_downsweep) {
    int finished1 = 0, finished2 = 0;
    if (m_down_height == m_cur_height) {
      // Communicate with higher ranks
      if (!m_send) {
        std::memcpy(m_tmp_buf.get(), m_scan_buf.get(), m_recv_size);
        int dest = m_target_ranks.back();
        RBC::Isend(m_tmp_buf.get(), m_count, m_datatype, dest, m_tag, m_comm, &m_send_req);
        RBC::Irecv(m_scan_buf.get(), m_count * 2, m_datatype, dest, m_tag, m_comm, &m_recv_req);
        m_send = true;
      }
      RBC::Test(&m_send_req, &finished1, MPI_STATUS_IGNORE);
      RBC::Test(&m_recv_req, &finished2, MPI_STATUS_IGNORE);
    } else if (m_receives >= m_cur_height) {
      // Communicate with lower ranks
      if (!m_send) {
        std::memcpy(m_tmp_buf.get(), m_scan_buf.get(), m_recv_size);
        std::memcpy(m_tmp_buf2, m_bcast_buf, m_recv_size);
        int dest = m_target_ranks[m_sends];
        RBC::Isend(m_tmp_buf.get(), m_count * 2, m_datatype, dest, m_tag, m_comm, &m_send_req);
        RBC::Irecv(m_scan_buf.get(), m_count, m_datatype, dest, m_tag, m_comm, &m_recv_req);
        m_sends++;
        m_send = true;
      }
      RBC::Test(&m_send_req, &finished1, MPI_STATUS_IGNORE);
      RBC::Test(&m_recv_req, &finished2, MPI_STATUS_IGNORE);
      if (finished1 && finished2) {
        if ((std::pow(2, m_up_height) - 1 == m_rank || m_rank == m_size - 1) &&
            !m_downsweep_recvd) {
          // tmp_buf has "empty" elements
          m_downsweep_recvd = true;
        } else {
          MPI_Reduce_local(m_tmp_buf.get(), m_scan_buf.get(), m_count, m_datatype, m_op);
        }
      }
    } else {
      m_cur_height--;
    }
    // Send and receive completed
    if (finished1 && finished2) {
      m_cur_height--;
      m_send = false;
    }
    // End downsweep phase
    if (m_cur_height == 0) {
      m_downsweep = false;
      char* buf = const_cast<char*>(static_cast<const char*>(m_sendbuf));
      std::memcpy(m_recvbuf_scan, buf, m_recv_size);
      if (m_rank != 0)
        MPI_Reduce_local(m_scan_buf.get(), m_recvbuf_scan, m_count, m_datatype, m_op);
      std::memcpy(m_recvbuf_bcast, m_bcast_buf, m_recv_size);
    }
  }

  if (!m_upsweep && !m_downsweep) {
    *flag = 1;
    m_completed = true;
  }
  return 0;
}
