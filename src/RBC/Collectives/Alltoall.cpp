/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
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
int Alltoall(void* sendbuf,
             int sendcount,
             MPI_Datatype sendtype,
             void* recvbuf,
             int recvcount,
             MPI_Datatype recvtype,
             Comm comm) {
  if (comm.useMPICollectives()) {
    return MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf,
                        recvcount, recvtype, comm.get());
  }

  const auto rank = comm.getRank();
  const auto size = comm.getSize();

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int64_t send_datatype_size = static_cast<int>(type_size);
  MPI_Type_get_extent(recvtype, &lb, &type_size);
  int64_t recv_datatype_size = static_cast<int>(type_size);

  int target = rank > 0 ? size - rank : 0;

  for (int i = 0; i != size; ++i) {
    RBC::Sendrecv(static_cast<char*>(sendbuf) + send_datatype_size * sendcount * target,
                  sendcount,
                  sendtype,
                  target,
                  Tag_Const::ALLTOALL,
                  static_cast<char*>(recvbuf) + recv_datatype_size * recvcount * target,
                  recvcount,
                  recvtype,
                  target,
                  Tag_Const::ALLTOALL,
                  comm,
                  MPI_STATUS_IGNORE);

    ++target;
    if (target == size) {
      target = 0;
    }
  }

  return 0;
}

int Alltoallv(void* sendbuf,
              const int* sendcounts,
              const int* sdispls,
              MPI_Datatype sendtype,
              void* recvbuf,
              const int* recvcounts,
              const int* rdispls,
              MPI_Datatype recvtype,
              Comm comm) {
  if (comm.useMPICollectives()) {
    return MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype,
                         recvbuf, recvcounts, rdispls, recvtype, comm.get());
  }

  const auto rank = comm.getRank();
  const auto size = comm.getSize();

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int64_t send_datatype_size = static_cast<int>(type_size);
  MPI_Type_get_extent(recvtype, &lb, &type_size);
  int64_t recv_datatype_size = static_cast<int>(type_size);

  int target = rank > 0 ? size - rank : 0;

  std::vector<MPI_Request> requests(2 * size);

  for (int i = 0; i != size; ++i) {
    RBC::Irecv(static_cast<char*>(recvbuf) + recv_datatype_size * rdispls[target],
               recvcounts[target], recvtype, target,
               Tag_Const::ALLTOALL, comm, requests.data() + 2 * i);
    RBC::Isend(static_cast<char*>(sendbuf) + send_datatype_size * sdispls[target],
               sendcounts[target], sendtype, target,
               Tag_Const::ALLTOALL, comm, requests.data() + 2 * i + 1);

    ++target;
    if (target == size) {
      target = 0;
    }
  }

  MPI_Waitall(2 * size, requests.data(), MPI_STATUSES_IGNORE);

  return 0;
}
}  // end namespace RBC
