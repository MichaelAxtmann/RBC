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

#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>

#include "RBC.hpp"
#include "tlx/math.hpp"

#include "BinaryTree.hpp"

namespace RBC {
namespace _internal {
/*
 * Request for the scan
 */
class IscanReq : public RequestSuperclass {
 public:
  IscanReq(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
           int tag, MPI_Op op, Comm const& comm);
  ~IscanReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  int m_count, m_tag, m_rank, m_size, recv_size, m_lchild, m_rchild, m_parent,
    m_is_left_receiving, m_is_right_receiving, m_send_cnt;
  MPI_Datatype m_datatype;
  MPI_Op m_op;
  Comm m_comm;
// States of the state machine.
  bool m_completed, m_up_send, m_up_recv, m_down_recv, m_down_send;
  bool m_mpi_collective;
  std::unique_ptr<char[]> m_left_tree, m_right_tree;
  Request m_requests[2];
  MPI_Request m_mpi_req;
};
}  // namespace _internal
}  // namespace RBC
