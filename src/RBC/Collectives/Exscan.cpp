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
#include <cmath>
#include <cstring>
#include <memory>

#include "RBC.hpp"

#include "Scan.hpp"

namespace RBC {
int Exscan(const void* sendbuf, void* recvbuf, int count,
           MPI_Datatype datatype, MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Exscan(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.get());
  }

  if (comm.getSize() == 1) {
    return 0;
  }

  const int tag = Tag_Const::EXSCAN;
  const int rank = comm.getRank();
  const int size = comm.getSize();
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  int msg_cnt = 0;
  std::unique_ptr<char[]> tmpbuf;
  RBC::Request requests[2];

  if (rank + 1 < size) {
    RBC::Isend(sendbuf, count, datatype, rank + 1, tag, comm, requests);
    ++msg_cnt;
  }
  if (rank > 0 && size == 2) {
    RBC::Irecv(recvbuf, count, datatype, rank - 1, tag, comm, requests + msg_cnt);
    ++msg_cnt;
  } else if (rank > 0) {
    tmpbuf.reset(new char[recv_size]);
    RBC::Irecv(tmpbuf.get(), count, datatype, rank - 1, tag, comm, requests + msg_cnt);
    ++msg_cnt;
  }

  RBC::Waitall(msg_cnt, requests, MPI_STATUSES_IGNORE);

  if (size == 2) {
    return 0;
  }

  if (rank > 0) {
    assert(size > 2);
    RBC::Comm subcomm;
    RBC::Comm_create_group(comm, &subcomm, 1, size - 1);
    RBC::Scan(tmpbuf.get(), recvbuf, count, datatype, op, subcomm);
  }

  return 0;
}

namespace _internal {
namespace optimized {
int Exscan(const void* sendbuf, void* recvbuf, int count,
           MPI_Datatype datatype, MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Exscan(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  if (size == 1 && count == 0) return 0;

  std::unique_ptr<char[]> tmp_buf = std::make_unique<char[]>(recv_size);
  std::unique_ptr<char[]> scan_buf = std::make_unique<char[]>(recv_size);
  std::memcpy(scan_buf.get(), sendbuf, recv_size);

  int commute = 0;
  MPI_Op_commutative(op, &commute);

  int mask = 1;
  int flag = 0;
  while (mask < size) {
    const int target = rank ^ mask;
    mask <<= 1;

    if (target < size) {
      Sendrecv(scan_buf.get(),
               count,
               datatype,
               target,
               Tag_Const::SCAN,
               tmp_buf.get(),
               count,
               datatype,
               target,
               Tag_Const::SCAN,
               comm,
               MPI_STATUS_IGNORE);

      const bool left_target = target < rank;
      if (left_target) {
        MPI_Reduce_local(tmp_buf.get(), scan_buf.get(), count, datatype, op);

        // Handle recvbuf in a special way
        if (rank) {
          if (flag) {
            MPI_Reduce_local(tmp_buf.get(), recvbuf, count, datatype, op);
          } else {
            std::memcpy(recvbuf, tmp_buf.get(), recv_size);
            flag = 1;
          }
        }
      } else {
        if (commute) {
          MPI_Reduce_local(tmp_buf.get(), scan_buf.get(), count, datatype, op);
        } else {
          MPI_Reduce_local(scan_buf.get(), tmp_buf.get(), count, datatype, op);
          std::memcpy(scan_buf.get(), tmp_buf.get(), recv_size);
        }
      }
    }
  }

  return 0;
}
}         // end namespace optimized

/*
 * Request for the exscan
 */
class IexscanReq : public RequestSuperclass {
 public:
  IexscanReq(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
             int tag, MPI_Op op, Comm const& comm);
  ~IexscanReq();
  int test(int* flag, MPI_Status* status);

 private:
  const void* m_sendbuf;
  void* m_recvbuf;
  int m_count, m_tag, m_rank, m_size, m_recv_size, m_msg_cnt, m_shift_phase, m_scan_phase;
  MPI_Datatype m_datatype;
  MPI_Op m_op;
  Comm m_comm;
  bool m_completed, m_mpi_collective;
  Request m_requests[2];
  MPI_Request m_mpi_req;
  std::unique_ptr<char[]> m_tmpbuf;
  std::unique_ptr<IscanReq> m_iscan_req;
};
}  // namespace _internal

int Iexscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IexscanReq>(sendbuf, recvbuf,
                                                       count, datatype, tag, op, comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IexscanReq::IexscanReq(const void* sendbuf, void* recvbuf, int count,
                                       MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_recvbuf(recvbuf),
  m_count(count),
  m_tag(tag),
  m_msg_cnt(0),
  m_shift_phase(false),
  m_scan_phase(false),
  m_datatype(datatype),
  m_op(op),
  m_comm(comm),
  m_completed(false),
  m_mpi_collective(false),
  m_tmpbuf(nullptr),
  m_iscan_req(nullptr) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Iexscan(sendbuf, recvbuf, count, datatype, op, comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif
  m_rank = comm.getRank();
  m_size = comm.getSize();
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  const int datatype_size = static_cast<int>(type_size);
  m_recv_size = count * datatype_size;

  if (m_size == 1) {
    m_completed = true;
    return;
  }

  m_shift_phase = true;

  m_msg_cnt = 0;
  if (m_rank + 1 < m_size) {
    RBC::Isend(sendbuf, count, datatype, m_rank + 1, tag, comm, m_requests);
    ++m_msg_cnt;
  }
  if (m_rank > 0 && m_size == 2) {
    RBC::Irecv(recvbuf, count, datatype, m_rank - 1, tag, comm, m_requests + m_msg_cnt);
    ++m_msg_cnt;
  } else if (m_rank > 0) {
    m_tmpbuf.reset(new char[m_recv_size]);
    RBC::Irecv(m_tmpbuf.get(), count, datatype, m_rank - 1, tag, comm, m_requests + m_msg_cnt);
    ++m_msg_cnt;
  }
}

RBC::_internal::IexscanReq::~IexscanReq() { }

int RBC::_internal::IexscanReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective) {
    return MPI_Test(&m_mpi_req, flag, status);
  }

  *flag = false;

  if (m_shift_phase) {
    assert(m_msg_cnt > 0);
    int local_flag = 0;
    RBC::Testall(m_msg_cnt, m_requests, &local_flag, MPI_STATUSES_IGNORE);
    if (local_flag) {
      if (m_size == 2 || m_rank == 0) {
        m_completed = true;
        m_shift_phase = false;
        *flag = true;
        return 0;
      } else {
        assert(m_size > 2 && m_rank > 0);
        RBC::Comm subcomm;
        RBC::Comm_create_group(m_comm, &subcomm, 1, m_size - 1);
        m_iscan_req.reset(new IscanReq(m_tmpbuf.get(), m_recvbuf,
                                       m_count, m_datatype, m_tag, m_op, subcomm));
        m_shift_phase = false;
        m_scan_phase = true;
        return 0;
      }
    } else {
      return 0;
    }
  }

  if (m_scan_phase) {
    int local_flag = 0;
    m_iscan_req->test(&local_flag, MPI_STATUS_IGNORE);
    if (local_flag) {
      m_scan_phase = false;
      m_iscan_req.reset(nullptr);
      m_completed = true;
      *flag = true;
      return 0;
    } else {
      return 0;
    }
  }

  assert(false);

  return 0;
}
