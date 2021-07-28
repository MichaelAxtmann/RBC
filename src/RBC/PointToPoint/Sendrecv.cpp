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
