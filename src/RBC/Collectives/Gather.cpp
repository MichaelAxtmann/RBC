/*****************************************************************************
 * this file is part of the Project RBC

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
#include <iostream>

#include <RBC.hpp>
#include <tlx/algorithm.hpp>
#include <tlx/math.hpp>

#include "Collectives.hpp"

namespace RBC {

int Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
           void* recvbuf, int recvcount, MPI_Datatype recvtype,
           int root, RBC::Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Gather(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                      root, comm.get());
  }

  if (sendcount == 0) {
    return 0;
  }

  const int tag = Tag_Const::GATHER;
  const int rank = comm.getRank();
  const int size = comm.getSize();

  MPI_Aint lb, m_type_bytes_aint;
  MPI_Type_get_extent(sendtype, &lb, &m_type_bytes_aint);
  const size_t m_type_bytes = static_cast<int>(m_type_bytes_aint);

  char* recv_ptr = static_cast<char*>(recvbuf);
  const auto total_recv_bytes = m_type_bytes * size * sendcount;

  if (rank != root || root != 0) {

    recv_ptr = static_cast<char*>(malloc(total_recv_bytes));
    
  }
  
  memcpy(recv_ptr, sendbuf, m_type_bytes * sendcount);

  const int zeroed_rank = _internal::RemoveBinomTreeRoot(root, rank, size);

  const int tailing_zeros = tlx::ffs(zeroed_rank) - 1;
  const int iterations = zeroed_rank > 0 ?
    tailing_zeros : tlx::integer_log2_ceil(size);

  int recved_count = sendcount;

  for (int i = 0; i != iterations; ++i) {

    const int source = zeroed_rank + (1 << i);

    if (source >= size) {
      break;
    }
    
    const int rooted_source = _internal::AddBinomTreeRoot(root, source, size);

    const int count = std::min(sendcount << i, sendcount * (size - source));
    Recv(recv_ptr + m_type_bytes * recved_count, count, recvtype, rooted_source, tag,
             comm, MPI_STATUS_IGNORE);

    recved_count += count;
  }

  if (zeroed_rank > 0) {

    const int target = zeroed_rank - (1 << iterations);
    const int rooted_target = _internal::AddBinomTreeRoot(root, target, size);

    Send(recv_ptr, recved_count, sendtype, rooted_target, tag, comm);
  }

  // Reorder elements at root.
  if (rank == root && root != 0) {

    const size_t right_bytes = m_type_bytes * root * recvcount;
    const size_t left_bytes  = m_type_bytes * size * recvcount - right_bytes;
    
    memcpy(recvbuf, recv_ptr + left_bytes, right_bytes);
    memcpy(static_cast<char*>(recvbuf) + right_bytes, recv_ptr, left_bytes);
    
  }

  if (rank != root || root != 0) {
    assert(recv_ptr != recvbuf);
    free(recv_ptr);
    recv_ptr = static_cast<char*>(recvbuf);
  }

  assert(recv_ptr == recvbuf);
  
  return 0;
}

namespace _internal {
/*
 * Request for the gather
 */
class IgatherReq : public RequestSuperclass {
 public:
  IgatherReq(const void* sendbuf, int sendcount,
             MPI_Datatype sendtype, void* recvbuf, int recvcount,
             MPI_Datatype recvtype,
             int root, int tag,
             Comm const& comm);
  ~IgatherReq();
  int test(int* flag, MPI_Status* status);

 private:
  char* m_buffer;
  char* m_recv_ptr;
  MPI_Datatype m_datatype;
  int m_count, m_recved_count, m_root, m_tag, m_size, m_rank,
    m_zeroed_rank, m_iteration, m_iterations;
  size_t m_type_bytes;
  Comm m_comm;
  bool m_send_posted, m_completed, m_mpi_collective;
  MPI_Request m_request;
};
}  // namespace _internal

int Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        int root, Comm const &comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IgatherReq>(sendbuf, sendcount,
                  sendtype, recvbuf, recvcount, recvtype, root,
                  tag, comm));
  return 0;
};
}  // namespace RBC

RBC::_internal::IgatherReq::IgatherReq(const void *sendbuf, int sendcount,
                                       MPI_Datatype sendtype, void *recvbuf, int recvcount,
                                       MPI_Datatype recvtype, int root, int tag,
                                       RBC::Comm const &comm) :
  m_buffer(static_cast<char*>(recvbuf)),
  m_recv_ptr(nullptr),
  m_datatype(sendtype),
  m_count(sendcount),
  m_recved_count(0),
  m_root(root),
  m_tag(tag),
  m_size(0),
  m_rank(0),
  m_zeroed_rank(0),
  m_iteration(0),
  m_iterations(0),
  m_type_bytes(0),
  m_comm(comm),
  m_send_posted(false),
  m_completed(false),
  m_mpi_collective(false),
  m_request(MPI_REQUEST_NULL) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
    if (comm.useMPICollectives()) {
        MPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, comm.get(), &m_request);
        m_mpi_collective = true;
        return;
    }
#endif

  m_rank = comm.getRank();
  m_size = comm.getSize();

  if (m_count == 0) {

    m_completed = true;
    return;
    
  }

  MPI_Aint lb, m_type_bytes_aint;
  MPI_Type_get_extent(sendtype, &lb, &m_type_bytes_aint);
  m_type_bytes = static_cast<int>(m_type_bytes_aint);

  if (m_size == 1) {

    memcpy(m_buffer, sendbuf, m_type_bytes * m_count);
    m_completed = true;
    return;
  }

  m_recv_ptr = static_cast<char*>(recvbuf);

  if (m_rank != root || root != 0) {
  
    const auto total_recv_bytes = m_type_bytes * m_size * m_count;
    m_recv_ptr = static_cast<char*>(malloc(total_recv_bytes));
  }
  
  memcpy(m_recv_ptr, sendbuf, m_type_bytes * m_count);
    
  m_zeroed_rank = _internal::RemoveBinomTreeRoot(root, m_rank, m_size);

  const int tailing_zeros = tlx::ffs(m_zeroed_rank) - 1;
  m_iterations = m_zeroed_rank > 0 ?
    tailing_zeros : tlx::integer_log2_ceil(m_size);

  m_recved_count = sendcount;

  // Start sending a message.
  if (m_iteration != m_iterations) {

    const int source = m_zeroed_rank + (1 << m_iteration);

    if (source >= m_size) {
      m_iteration = m_iterations;
      return;
    }
    
    const int rooted_source = _internal::AddBinomTreeRoot(m_root, source, m_size);
    const int count = std::min(m_count << m_iteration, m_count * (m_size - source));

    Irecv(m_recv_ptr + m_type_bytes * m_recved_count, count, m_datatype, rooted_source, m_tag,
         m_comm, &m_request);

    m_recved_count += count;
    ++m_iteration;

    return;
  }

  // Send a message.
  if (m_zeroed_rank > 0) {

    const int target = m_zeroed_rank - (1 << m_iterations);
    const int rooted_target = _internal::AddBinomTreeRoot(m_root, target, m_size);

    Isend(m_recv_ptr, m_recved_count, m_datatype, rooted_target, m_tag, m_comm, &m_request);
    m_send_posted = true;
    
    return;
  }

  assert(m_size == 1);
  assert(m_recv_ptr == nullptr);
  m_completed = true;
  
};

RBC::_internal::IgatherReq::~IgatherReq() {
  assert(m_recv_ptr == nullptr);
}

int RBC::_internal::IgatherReq::test(int *flag, MPI_Status *status) {
    if (m_completed) {
        *flag = 1;
        return 0;
    }

    *flag = 0;

    if (m_mpi_collective) {
      const auto err = MPI_Test(&m_request, flag, status);
      m_completed = *flag;
      return err;
    }

  // Complete pending request.
  if (m_request != MPI_REQUEST_NULL) {

    int completed = 0;
    MPI_Test(&m_request, &completed, MPI_STATUS_IGNORE);
    if (completed) {
      m_request = MPI_REQUEST_NULL;
    }

  }

  // No pending request, receive data.
  if (m_request == MPI_REQUEST_NULL && m_iteration != m_iterations) {

    const int source = m_zeroed_rank + (1 << m_iteration);

    if (source < m_size) {

      const int rooted_source = _internal::AddBinomTreeRoot(m_root, source, m_size);
      const int count = std::min(m_count << m_iteration, m_count * (m_size - source));

      Irecv(m_recv_ptr + m_type_bytes * m_recved_count, count, m_datatype, rooted_source, m_tag,
            m_comm, &m_request);

      m_recved_count += count;
      ++m_iteration;

    } else {

      m_iteration = m_iterations;

    }
  }

  if (m_request == MPI_REQUEST_NULL && m_iteration == m_iterations) {
  
    if (m_zeroed_rank > 0 && !m_send_posted) {

      const int target = m_zeroed_rank - (1 << m_iterations);
      const int rooted_target = _internal::AddBinomTreeRoot(m_root, target, m_size);

      Isend(m_recv_ptr, m_recved_count, m_datatype, rooted_target, m_tag, m_comm, &m_request);
      m_send_posted = true;
        
    } else if (m_zeroed_rank > 0 && m_send_posted) {

      assert(m_recv_ptr != nullptr);
      free(m_recv_ptr);
      m_recv_ptr = nullptr;

      m_completed = true;
      *flag = 1;

    } else if (m_zeroed_rank == 0 && m_rank != 0) {

      const size_t right_bytes = m_type_bytes * m_root * m_count;
      const size_t left_bytes  = m_type_bytes * m_size * m_count - right_bytes;
    
      memcpy(m_buffer, m_recv_ptr + left_bytes, right_bytes);
      memcpy(static_cast<char*>(m_buffer) + right_bytes, m_recv_ptr, left_bytes);
        
      assert(m_recv_ptr != nullptr);
      free(m_recv_ptr);
      m_recv_ptr = nullptr;

      m_completed = true;
      *flag = 1;
    
    } else {
      assert(m_iteration == m_iterations && m_zeroed_rank == 0 && m_rank == 0);
        
      m_completed = true;
      *flag = 1;

    }

  }


  return 0;

}
