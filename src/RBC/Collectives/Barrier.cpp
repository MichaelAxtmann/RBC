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


#include "RBC.hpp"

namespace RBC {
int Barrier(Comm const& comm) {
  if (comm.useMPICollectives() || comm.splitMPIComm()) {
    MPI_Barrier(comm.get());
  } else {
    int a = 0, b = 0;
    RBC::Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, comm);
    RBC::Bcast( &a, 1, MPI_INT, 0, comm );
  }
  return 0;
}

namespace _internal {
/*
 * Request for the barrier
 */
class IbarrierReq : public RequestSuperclass {
 public:
  explicit IbarrierReq(RBC::Comm const& comm, int tag);
  int test(int* flag, MPI_Status* status);

 private:
  const int m_tag;
  Request request;
  MPI_Request mpi_req;
  bool mpi_collective;
  int a, b, c;
};
}  // namespace _internal

int Ibarrier(Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<RBC::_internal::IbarrierReq>(comm, tag));
  return 0;
}
}  // namespace RBC

RBC::_internal::IbarrierReq::IbarrierReq(RBC::Comm const& comm, int tag) :
  m_tag(tag),
  mpi_collective(false),
  a(0) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Ibarrier(comm.get(), &mpi_req);
    mpi_collective = true;
    return;
  }
#endif
  IscanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm, &request, tag);
}

int RBC::_internal::IbarrierReq::test(int* flag, MPI_Status* status) {
  if (mpi_collective)
    return MPI_Test(&mpi_req, flag, status);

  return Test(&request, flag, MPI_STATUS_IGNORE);
}
