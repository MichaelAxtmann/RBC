/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

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
  MPI_Request m_requests[2];
  MPI_Request m_mpi_req;
};
}  // namespace _internal
}  // namespace RBC
