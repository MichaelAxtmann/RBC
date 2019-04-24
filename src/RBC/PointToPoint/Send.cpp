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
