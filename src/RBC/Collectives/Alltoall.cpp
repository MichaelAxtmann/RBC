/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
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

  std::vector<MPI_Request> requests;
  requests.reserve(2 * size);

  for (int i = 0; i != size; ++i) {
    if (recvcounts[target] > 0) {
      requests.emplace_back();
      RBC::Irecv(static_cast<char*>(recvbuf) + recv_datatype_size * rdispls[target],
                 recvcounts[target], recvtype, target,
                 Tag_Const::ALLTOALL, comm, &requests.back());
    }

    if (sendcounts[target] > 0) {
      requests.emplace_back();
      RBC::Isend(static_cast<char*>(sendbuf) + send_datatype_size * sdispls[target],
                 sendcounts[target], sendtype, target,
                 Tag_Const::ALLTOALL, comm, &requests.back());
    }

    ++target;
    if (target == size) {
      target = 0;
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  return 0;
}
}  // end namespace RBC
