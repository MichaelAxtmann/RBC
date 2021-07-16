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
#include "Send.hpp"

namespace RBC {
int Send(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
         int tag, Comm const& comm) {
  return MPI_Send(const_cast<void*>(sendbuf), count, datatype,
                  comm.RangeRankToMpiRank(dest), tag, comm.get());
}

int Ssend(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
          int tag, Comm const& comm) {
  return MPI_Ssend(const_cast<void*>(sendbuf), count, datatype,
                   comm.RangeRankToMpiRank(dest), tag, comm.get());
}

namespace _internal {
/*
 * Request for the isend
 */
class IsendReq : public RequestSuperclass {
 public:
  IsendReq(const void* sendbuf, int count, MPI_Datatype datatype,
           int dest, int tag, Comm const& comm);
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  int m_count, m_dest, m_tag;
  MPI_Datatype m_datatype;
  Comm m_comm;
  // bool requested;
  MPI_Request m_request;
};

/*
 * Request for the issend
 */
class IssendReq : public RequestSuperclass {
 public:
  IssendReq(const void* sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, Comm const& comm);
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  int m_count;
  MPI_Datatype m_datatype;
  int m_dest, m_tag;
  Comm m_comm;
  // bool requested;
  MPI_Request m_request;
};


/*
 * Send operation which invokes MPI_Send if count > 0
 */
int SendNonZeroed(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
                  int tag, Comm const& comm) {
  if (count == 0) return 0;

  return MPI_Send(const_cast<void*>(sendbuf), count, datatype,
                  comm.RangeRankToMpiRank(dest), tag, comm.get());
}
}     // end namespace _internal

int Isend(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
          int tag, RBC::Comm const& comm, MPI_Request* request) {
  return MPI_Isend(const_cast<void*>(sendbuf), count, datatype,
                   comm.RangeRankToMpiRank(dest), tag, comm.get(), request);
}

int Isend(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
          int tag, Comm const& comm, Request* request) {
  request->set(std::make_shared<_internal::IsendReq>(sendbuf, count,
                                                     datatype, dest, tag, comm));
  return 0;
}

int Issend(const void* sendbuf, int count, MPI_Datatype datatype, int dest,
           int tag, Comm const& comm, Request* request) {
  request->set(std::make_shared<_internal::IssendReq>(sendbuf, count,
                                                      datatype, dest, tag, comm));
  return 0;
}

int Issend(const void* buffer, int count, MPI_Datatype datatype, int dest, int tag,
           RBC::Comm const& comm, MPI_Request* request) {
  void* buf = const_cast<void*>(buffer);
  MPI_Issend(buf, count, datatype, comm.RangeRankToMpiRank(dest),
             tag, comm.get(), request);
  return 0;
}
}  // namespace RBC

RBC::_internal::IsendReq::IsendReq(const void* sendbuf, int count, MPI_Datatype datatype,
                                   int dest, int tag, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_count(count),
  m_dest(dest),
  m_tag(tag),
  m_datatype(datatype),
  m_comm(comm) {     // , requested(false) {
  void* buf = const_cast<void*>(sendbuf);
  MPI_Isend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag, comm.get(), &m_request);
}

int RBC::_internal::IsendReq::test(int* flag, MPI_Status* status) {
  return MPI_Test(&m_request, flag, status);
}

RBC::_internal::IssendReq::IssendReq(const void* sendbuf, int count, MPI_Datatype datatype,
                                     int dest, int tag, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_count(count),
  m_datatype(datatype),
  m_dest(dest),
  m_tag(tag),
  m_comm(comm) {     // , requested(false) {
  void* buf = const_cast<void*>(sendbuf);
  MPI_Issend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag,
             comm.get(), &m_request);
}

int RBC::_internal::IssendReq::test(int* flag, MPI_Status* status) {
  return MPI_Test(&m_request, flag, status);
}
