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


#include <cassert>
#include <memory>

#include <RBC.hpp>

#include "Recv.hpp"

namespace RBC {
int Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
         Comm const& comm, MPI_Status* status) {
  if (source == MPI_ANY_SOURCE) {
    return MPI_Recv(buf, count, datatype, source, tag, comm.get(), status);
  } else {
    return MPI_Recv(buf, count, datatype, comm.RangeRankToMpiRank(source),
                    tag, comm.get(), status);
  }
}

namespace _internal {
/*
 * Request for the receive
 */
class IrecvReq : public RequestSuperclass {
 public:
  IrecvReq(void* buffer, int count, MPI_Datatype datatype, int source,
           int tag, Comm const& comm);
  int test(int* flag, MPI_Status* status);

 private:
  void* m_buffer;
  int m_count, m_source, m_tag;
  MPI_Datatype m_datatype;
  Comm m_comm;
  bool m_receiving;
  MPI_Request m_request;
};

/*
 * Receive operation which invokes MPI_Recv if count > 0
 */
int RecvNonZeroed(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                  Comm const& comm, MPI_Status* status) {
  if (count == 0) return 0;

  if (source == MPI_ANY_SOURCE) {
    return MPI_Recv(buf, count, datatype, source, tag, comm.get(), status);
  } else {
    return MPI_Recv(buf, count, datatype, comm.RangeRankToMpiRank(source),
                    tag, comm.get(), status);
  }
}
}     // end namespace _internal

int Irecv(void* buffer, int count, MPI_Datatype datatype, int source, int tag,
          RBC::Comm const& comm, MPI_Request* request) {
  if (source == MPI_ANY_SOURCE) {
    return MPI_Irecv(buffer, count, datatype, source, tag, comm.get(), request);
  } else {
    return MPI_Irecv(buffer, count, datatype, comm.RangeRankToMpiRank(source),
                     tag, comm.get(), request);
  }
}

int Irecv(void* buffer, int count, MPI_Datatype datatype, int source, int tag,
          Comm const& comm, Request* request) {
  request->set(std::make_shared<_internal::IrecvReq>(buffer, count,
                                                     datatype, source, tag, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IrecvReq::IrecvReq(void* buffer, int count, MPI_Datatype datatype,
                                   int source, int tag, RBC::Comm const& comm) :
  m_buffer(buffer),
  m_count(count),
  m_source(source),
  m_tag(tag),
  m_datatype(datatype),
  m_comm(comm),
  m_receiving(false) {
  if (source != MPI_ANY_SOURCE) {
    MPI_Irecv(buffer, count, datatype, comm.RangeRankToMpiRank(source), tag,
              comm.get(), &m_request);
    m_receiving = true;
  } else {
    int x;
    this->test(&x, MPI_STATUS_IGNORE);
  }
}

int RBC::_internal::IrecvReq::test(int* flag, MPI_Status* status) {
  if (m_receiving) {
    return MPI_Test(&m_request, flag, status);
  } else {
    assert(m_source == MPI_ANY_SOURCE);
    int ready;
    MPI_Status stat;
    RBC::Iprobe(MPI_ANY_SOURCE, m_tag, m_comm, &ready, &stat);
    if (ready) {
      MPI_Irecv(m_buffer, m_count, m_datatype, stat.MPI_SOURCE, m_tag, m_comm.get(),
                &m_request);
      m_receiving = true;
    }
  }
  return 0;
}
