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
#include "Scan.hpp"

namespace RBC {
int Scan(const void* sendbuf, void* recvbuf, int count,
         MPI_Datatype datatype, MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Scan(const_cast<void*>(sendbuf), recvbuf, count, datatype,
                    op, comm.get());
  }

  const int rank = comm.getRank();
  const int size = comm.getSize();
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);
  const int recv_size = count * datatype_size;

  if (comm.getSize() == 1) {
    std::memcpy(recvbuf, sendbuf, recv_size);
    return 0;
  }

  const int tag = Tag_Const::SCAN;

  int lchild = -1;
  int rchild = -1;
  int parent = -1;

  _internal::BinaryTree::Create(rank, size, &lchild, &rchild, &parent);

  std::unique_ptr<char[]> left_tree(new char[recv_size]);
  std::unique_ptr<char[]> right_tree(new char[recv_size]);

  Request requests[2];
  int is_left_receiving = 0;
  int is_right_receiving = 0;

  std::memcpy(recvbuf, sendbuf, recv_size);

  if (lchild != -1) {
    RBC::Irecv(left_tree.get(), count, datatype, lchild, tag, comm, requests);
    is_left_receiving = 1;
  }
  if (rchild != -1) {
    RBC::Irecv(right_tree.get(), count, datatype, rchild, tag, comm,
               requests + is_left_receiving);
    is_right_receiving = 1;
  }
  // var: up_recv
  RBC::Waitall(is_left_receiving + is_right_receiving, requests, MPI_STATUSES_IGNORE);

  if (is_left_receiving) {
    MPI_Reduce_local(left_tree.get(), recvbuf, count, datatype, op);
  }
  if (is_right_receiving) {
    MPI_Reduce_local(recvbuf, right_tree.get(), count, datatype, op);
  } else {
    std::memcpy(right_tree.get(), recvbuf, recv_size);
  }

  if (parent != -1) {
    // var: up_send
    RBC::Send(right_tree.get(), count, datatype, parent, tag, comm);
  }

  // Receive reduces values of processes left to our
  // subtree. There are not processes left to our subtree if we
  // are a process of the leftmost front of the binary tree.
  if (parent != -1 && !tlx::is_power_of_two(rank + 1)) {
    // var: down_recv
    RBC::Recv(left_tree.get(), count, datatype, parent, tag, comm, MPI_STATUS_IGNORE);
    MPI_Reduce_local(left_tree.get(), recvbuf, count, datatype, op);
  }

  int send_cnt = 0;
  // Receive reduces values of processes left to our
  // subtree. There are not processes left to our subtree if we
  // are a process of the leftmost front of the binary tree.
  if (is_left_receiving && !tlx::is_power_of_two(rank + 1) && parent != -1) {
    RBC::Isend(left_tree.get(), count, datatype, lchild, tag, comm, requests);
    ++send_cnt;
  }

  if (is_right_receiving) {
    RBC::Isend(recvbuf, count, datatype, rchild, tag, comm,
               requests + send_cnt);
    ++send_cnt;
  }

  // var: down_send
  if (send_cnt) {
    RBC::Waitall(send_cnt, requests, MPI_STATUSES_IGNORE);
  }

  return 0;
}

namespace _internal {
namespace optimized {
int Scan(const void* sendbuf, void* recvbuf, int count,
         MPI_Datatype datatype, MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Scan(const_cast<void*>(sendbuf), recvbuf, count, datatype,
                    op, comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  std::memcpy(recvbuf, sendbuf, recv_size);

  if (size == 0) return 0;

  std::unique_ptr<char[]> tmp_buf(new char[recv_size]);
  std::unique_ptr<char[]> scan_buf(new char[recv_size]);
  std::memcpy(scan_buf.get(), sendbuf, recv_size);

  int commute = 0;
  MPI_Op_commutative(op, &commute);

  int mask = 1;
  while (mask < size) {
    const int target = rank ^ mask;
    mask <<= 1;

    if (target < size) {
      RBC::Sendrecv(scan_buf.get(),
                    count,
                    datatype,
                    target,
                    Tag_Const::SCAN,
                    tmp_buf.get(),
                    count,
                    datatype,
                    target,
                    Tag_Const::SCAN,
                    comm,
                    MPI_STATUS_IGNORE);

      const bool left_target = target < rank;
      if (left_target) {
        MPI_Reduce_local(tmp_buf.get(), scan_buf.get(), count, datatype, op);
        MPI_Reduce_local(tmp_buf.get(), recvbuf, count, datatype, op);
      } else {
        if (commute) {
          MPI_Reduce_local(tmp_buf.get(), scan_buf.get(), count, datatype, op);
        } else {
          MPI_Reduce_local(scan_buf.get(), tmp_buf.get(), count, datatype, op);
          std::memcpy(scan_buf.get(), tmp_buf.get(), recv_size);
        }
      }
    }
  }

  return 0;
}
}  // namespace optimized
}  // namespace _internal

int Iscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
          MPI_Op op, Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IscanReq>(sendbuf, recvbuf,
                                                     count, datatype, tag, op, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IscanReq::IscanReq(const void* sendbuf, void* recvbuf, int count,
                                   MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_count(count),
  m_tag(tag),
  m_rank(comm.getRank()),
  m_size(comm.getSize()),
  m_lchild(-1),
  m_rchild(-1),
  m_parent(-1),
  m_is_left_receiving(0),
  m_is_right_receiving(0),
  m_send_cnt(0),
  m_datatype(datatype),
  m_op(op),
  m_comm(comm),
  m_completed(false),
  m_up_send(false),
  m_up_recv(false),
  m_down_recv(false),
  m_down_send(false),
  m_mpi_collective(false),
  m_left_tree(nullptr),
  m_right_tree(nullptr) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);
  recv_size = count * datatype_size;

  if (comm.getSize() == 1) {
    std::memcpy(recvbuf, sendbuf, recv_size);
    m_completed = true;
    return;
  }

  _internal::BinaryTree::Create(m_rank, m_size, &m_lchild, &m_rchild, &m_parent);

  m_left_tree.reset(new char[recv_size]);
  m_right_tree.reset(new char[recv_size]);

  std::memcpy(recvbuf, sendbuf, recv_size);

  if (m_lchild != -1) {
    RBC::Irecv(m_left_tree.get(), count, datatype, m_lchild, tag, comm, m_requests);
    m_is_left_receiving = 1;
  }
  if (m_rchild != -1) {
    RBC::Irecv(m_right_tree.get(), count, datatype, m_rchild, tag, comm,
               m_requests + m_is_left_receiving);
    m_is_right_receiving = 1;
  }

  if (m_lchild != -1 || m_rchild != -1) {
    m_up_recv = true;
  } else {
    // We do not have any children. In this case, we start sending
    // upwards right away.
    assert(!m_is_left_receiving);
    assert(!m_is_right_receiving);
    assert(m_parent != -1);

    std::memcpy(m_right_tree.get(), recvbuf, recv_size);
    RBC::Isend(m_right_tree.get(), count, datatype, m_parent, tag, comm, m_requests);
    m_up_send = true;
  }

  assert(m_up_recv || m_up_send);
}

RBC::_internal::IscanReq::~IscanReq() { }

int RBC::_internal::IscanReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  *flag = false;

  if (m_up_recv) {
    assert(m_is_left_receiving + m_is_right_receiving);
    int local_flag = 0;
    RBC::Testall(m_is_left_receiving + m_is_right_receiving, m_requests, &local_flag, MPI_STATUSES_IGNORE);
    if (local_flag) {
      if (m_is_left_receiving) {
        MPI_Reduce_local(m_left_tree.get(), m_recvbuf, m_count, m_datatype, m_op);
      }
      if (m_is_right_receiving) {
        MPI_Reduce_local(m_recvbuf, m_right_tree.get(), m_count, m_datatype, m_op);
      } else {
        std::memcpy(m_right_tree.get(), m_recvbuf, recv_size);
      }

      if (m_parent != -1) {
        RBC::Isend(m_right_tree.get(), m_count, m_datatype, m_parent, m_tag, m_comm, m_requests);
        m_up_recv = false;
        m_up_send = true;
        return 0;
      }

      // We are the root as we do not send a message upwards. If
      // we have a right child, we start the downward
      // phase. Otherwise, there are just two processes and
      // there is no downward phase.
      assert(m_parent == -1);
      if (m_is_right_receiving) {
        RBC::Isend(m_recvbuf, m_count, m_datatype, m_rchild, m_tag, m_comm,
                   m_requests);
        assert(m_send_cnt == 0);
        m_send_cnt = 1;
        m_up_recv = false;
        m_down_send = true;
        return 0;
      } else {
        m_up_recv = false;
        m_completed = true;
        *flag = true;
        return 0;
      }
    } else {
      return 0;
    }
  } else if (m_up_send) {
    int local_flag = 0;
    RBC::Test(m_requests, &local_flag, MPI_STATUS_IGNORE);
    if (local_flag) {
      // Receive reduces values of processes left to our
      // subtree. There are not processes left to our subtree if we
      // are a process of the leftmost front of the binary tree.
      if (m_parent != -1 && !tlx::is_power_of_two(m_rank + 1)) {
        RBC::Irecv(m_left_tree.get(), m_count, m_datatype, m_parent, m_tag, m_comm,
                   m_requests);
        m_up_send = false;
        m_down_recv = true;
        return 0;
      }

      assert(m_send_cnt == 0);
      // Receive reduces values of processes left to our
      // subtree. There are not processes left to our subtree if we
      // are a process of the leftmost front of the binary tree.
      if (m_is_left_receiving && !tlx::is_power_of_two(m_rank + 1) && m_parent != -1) {
        RBC::Isend(m_left_tree.get(), m_count, m_datatype, m_lchild, m_tag, m_comm, m_requests + m_send_cnt);
        ++m_send_cnt;
      }

      if (m_is_right_receiving) {
        RBC::Isend(m_recvbuf, m_count, m_datatype, m_rchild, m_tag, m_comm,
                   m_requests + m_send_cnt);
        ++m_send_cnt;
      }

      // var: down_send
      if (m_send_cnt) {
        m_up_send = false;
        m_down_send = true;
        return 0;
      } else {
        // We send a message upwards in the upward phase, we do not
        // receive a message in the down phase and we do not send a
        // message in the down phase. In this case, we are the
        // leftmost process.
        assert(m_parent != -1 && tlx::is_power_of_two(m_rank + 1));
        m_up_send = false;
        m_completed = true;
        *flag = true;
        return 0;
      }
    } else {
      return 0;
    }
  } else if (m_down_recv) {
    int local_flag = 0;
    RBC::Test(m_requests, &local_flag, MPI_STATUS_IGNORE);
    if (local_flag) {
      MPI_Reduce_local(m_left_tree.get(), m_recvbuf, m_count, m_datatype, m_op);

      assert(m_send_cnt == 0);
      // Receive reduces values of processes left to our
      // subtree. There are not processes left to our subtree if we
      // are a process of the leftmost front of the binary tree.
      if (m_is_left_receiving && !tlx::is_power_of_two(m_rank + 1) && m_parent != -1) {
        RBC::Isend(m_left_tree.get(), m_count, m_datatype, m_lchild, m_tag, m_comm, m_requests + m_send_cnt);
        ++m_send_cnt;
      }

      if (m_is_right_receiving) {
        RBC::Isend(m_recvbuf, m_count, m_datatype, m_rchild, m_tag, m_comm,
                   m_requests + m_send_cnt);
        ++m_send_cnt;
      }

      // var: down_send
      if (m_send_cnt) {
        m_down_recv = false;
        m_down_send = true;
        return 0;
      } else {
        m_down_recv = false;
        m_completed = true;
        *flag = true;
        return 0;
      }
    } else {
      return 0;
    }
  } else if (m_down_send) {
    assert(m_send_cnt);
    int local_flag = 0;
    RBC::Testall(m_send_cnt, m_requests, &local_flag, MPI_STATUSES_IGNORE);
    if (local_flag) {
      m_down_send = true;
      m_completed = true;
      *flag = true;
      return 0;
    } else {
      return 0;
    }
  }

  assert(false);
  return 0;
}
