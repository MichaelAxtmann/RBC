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

#include <memory>

#include "RBC.hpp"
#include "Sendrecv.hpp"

namespace RBC {
int Sendrecv(void* sendbuf,
             int sendcount, MPI_Datatype sendtype,
             int dest, int sendtag,
             void* recvbuf, int recvcount, MPI_Datatype recvtype,
             int source, int recvtag,
             Comm const& comm, MPI_Status* status) {
  return MPI_Sendrecv(sendbuf,
                      sendcount, sendtype,
                      comm.RangeRankToMpiRank(dest), sendtag,
                      recvbuf, recvcount, recvtype,
                      comm.RangeRankToMpiRank(source), recvtag,
                      comm.get(), status);
}

namespace _internal {
/*
 * Request for the isend
 */
class IsendrecvReq : public RequestSuperclass {
 public:
  IsendrecvReq(void* sendbuf,
               int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
               void* recvbuf, int recvcount, MPI_Datatype recvtype,
               int source, int recvtag,
               Comm const& comm);
  int test(int* flag, MPI_Status* status);

 private:
  MPI_Request requests[2];
};


/*
 * Sendrecv operation which drops empty messages.
 */
int SendrecvNonZeroed(void* sendbuf,
                      int sendcount, MPI_Datatype sendtype,
                      int dest, int sendtag,
                      void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      int source, int recvtag,
                      Comm const& comm, MPI_Status* status) {
  if (sendcount > 0 && recvcount > 0) {
    return MPI_Sendrecv(sendbuf,
                        sendcount, sendtype,
                        comm.RangeRankToMpiRank(dest), sendtag,
                        recvbuf, recvcount, recvtype,
                        comm.RangeRankToMpiRank(source), recvtag,
                        comm.get(), status);
  } else if (sendcount > 0) {
    RBC::Send(sendbuf, sendcount, sendtype, dest, sendtag, comm);
  } else if (recvcount > 0) {
    RBC::Recv(recvbuf, recvcount, recvtype, source, recvtag, comm, MPI_STATUS_IGNORE);
  }

  // Case: Both messages are empty.
  return 0;
}
}     // end namespace _internal

int Isendrecv(void* sendbuf,
              int sendcount, MPI_Datatype sendtype,
              int dest, int sendtag,
              void* recvbuf, int recvcount, MPI_Datatype recvtype,
              int source, int recvtag,
              Comm const& comm, Request* request) {
  request->set(std::make_shared<_internal::IsendrecvReq>(sendbuf, sendcount, sendtype, dest,
                                                         sendtag, recvbuf, recvcount, recvtype,
                                                         source, recvtag, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IsendrecvReq::IsendrecvReq(void* sendbuf,
                                           int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                                           void* recvbuf, int recvcount, MPI_Datatype recvtype,
                                           int source, int recvtag,
                                           RBC::Comm const& comm) {
  void* sendbuf_ = const_cast<void*>(sendbuf);
  MPI_Isend(sendbuf_, sendcount, sendtype, comm.RangeRankToMpiRank(dest),
            sendtag, comm.get(), &requests[0]);
  MPI_Irecv(recvbuf, recvcount, recvtype, comm.RangeRankToMpiRank(source),
            recvtag, comm.get(), &requests[1]);
}

int RBC::_internal::IsendrecvReq::test(int* flag, MPI_Status*) {
  return MPI_Testall(2, requests, flag, MPI_STATUSES_IGNORE);
}
