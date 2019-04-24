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


#include "RBC.hpp"

namespace RBC {
int Barrier(Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Barrier(comm.get());
  }
  int a = 0, b = 0, c = 0;
  ScanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm);
  return 0;
}

namespace _internal {
/*
 * Request for the barrier
 */
class IbarrierReq : public RequestSuperclass {
 public:
  explicit IbarrierReq(RBC::Comm const& comm);
  int test(int* flag, MPI_Status* status);

 private:
  Request request;
  MPI_Request mpi_req;
  bool mpi_collective;
  int a, b, c;
};
}  // namespace _internal

int Ibarrier(Comm const& comm, Request* request) {
  request->set(std::make_shared<RBC::_internal::IbarrierReq>(comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IbarrierReq::IbarrierReq(RBC::Comm const& comm) :
  mpi_collective(false),
  a(0) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Ibarrier(comm.get(), &mpi_req);
    mpi_collective = true;
    return;
  }
#endif
  int tag = Tag_Const::BARRIER;
  IscanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm, &request, tag);
}

int RBC::_internal::IbarrierReq::test(int* flag, MPI_Status* status) {
  if (mpi_collective)
    return MPI_Test(&mpi_req, flag, status);

  return Test(&request, flag, MPI_STATUS_IGNORE);
}
