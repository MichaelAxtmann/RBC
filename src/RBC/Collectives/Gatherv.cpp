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

#include <RBC.hpp>
#include <tlx/algorithm.hpp>
#include <tlx/math.hpp>

#include "../PointToPoint/Recv.hpp"
#include "../PointToPoint/Send.hpp"
#include "Collectives.hpp"

namespace RBC {
int Gatherv(const void* sendbuf, int sendcount,
            MPI_Datatype sendtype, void* recvbuf,
            const int* recvcounts, const int* displs, MPI_Datatype recvtype,
            int root, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Gatherv(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf,
                       const_cast<int*>(recvcounts), const_cast<int*>(displs),
                       recvtype, root, comm.get());
  }

  const int tag = Tag_Const::GATHERV;
  const int rank = comm.getRank();
  const int size = comm.getSize();

  int total_recvcount = 0;
  if (rank == root ){
    total_recvcount = displs[size - 1] + recvcounts[size - 1];
  }

  Bcast(&total_recvcount, 1, MPI_INT, root, comm);

  if (total_recvcount == 0) {
    return 0;
  }

  MPI_Aint lb, type_bytes_aint;
  MPI_Type_get_extent(sendtype, &lb, &type_bytes_aint);
  const size_t type_bytes = static_cast<int>(type_bytes_aint);

  char* recv_ptr = static_cast<char*>(recvbuf);
  const auto total_recv_bytes = type_bytes * total_recvcount;

  if (rank != root || root != 0) {
    
    recv_ptr = static_cast<char*>(malloc(total_recv_bytes));
    
  }

  memcpy(recv_ptr, sendbuf, type_bytes * sendcount);

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

    MPI_Status status;
    Recv(recv_ptr + type_bytes * recved_count, total_recvcount - recved_count, sendtype,
         rooted_source, tag, comm, &status);

    int count = 0;
    MPI_Get_count(&status, sendtype, &count);
    recved_count += count;
    
  }

  if (zeroed_rank > 0) {

    const int target = zeroed_rank - (1 << iterations);
    const int rooted_target = _internal::AddBinomTreeRoot(root, target, size);

    Send(recv_ptr, recved_count, sendtype, rooted_target, tag, comm);
  }

  // Reorder elements at root.
  if (rank == root && root != 0) {

    const size_t right_bytes = type_bytes * displs[root];
    const size_t left_bytes = type_bytes * recved_count - right_bytes;
    
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
 * Request for the gatherv
 */
class IgathervReq : public RequestSuperclass {
 public:
  IgathervReq(const void* sendbuf, int sendcount,
              MPI_Datatype sendtype, void* recvbuf,
              const int* recvcounts, const int* displs, MPI_Datatype recvtype,
              int root, int tag, Comm const& comm);
  ~IgathervReq();
  int test(int* flag, MPI_Status* status);

 private:
  void InitGather();

  const void* m_sendbuf;
  void* m_buffer;
  void* m_recv_ptr;
  MPI_Datatype m_datatype;
  int m_count, m_total_recvcount, m_total_recv_count, m_recved_count, m_root, m_tag, m_size, m_rank,
    m_zeroed_rank, m_iteration, m_iterations;
  size_t m_type_bytes;
  Comm m_comm;
  bool m_bcast_finished, m_send_posted, m_recv_posted, m_completed, m_mpi_collective;
  MPI_Request m_request;
  RBC::Request m_bcast_request;
  const int* m_recvcounts;
  const int* m_displs;
};
}  // namespace _internal

int Igatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
             void* recvbuf, const int* recvcounts, const int* displs,
             MPI_Datatype recvtype, int root, Comm const& comm,
             Request* request, int tag) {
  request->set(std::make_shared<_internal::IgathervReq>(sendbuf, sendcount,
                                                        sendtype, recvbuf,
                                                        recvcounts, displs,
                                                        recvtype, root, tag,
                                                        comm));
  return 0;
}
}  // namespace RBC

RBC::_internal::IgathervReq::IgathervReq(const void* sendbuf, int sendcount,
                                         MPI_Datatype sendtype, void* recvbuf, 
                                         const int* recvcounts, const int* displs,
                                         MPI_Datatype /*recvtype*/,
                                         int root, int tag, RBC::Comm const& comm) :
  m_sendbuf(sendbuf),
  m_buffer(recvbuf),
  m_recv_ptr(nullptr),
  m_datatype(sendtype),
  m_count(sendcount),
  m_total_recv_count(-1),
  m_recved_count(0),
  m_root(root),
  m_tag(tag),
  m_size(0),
  m_zeroed_rank(0),
  m_iteration(0),
  m_iterations(0),
  m_type_bytes(0),
  m_comm(comm),
  m_bcast_finished(false),
  m_send_posted(false),
  m_recv_posted(false),
  m_completed(false),
  m_mpi_collective(false),
  m_request(MPI_REQUEST_NULL),
  m_recvcounts(recvcounts), m_displs(displs) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (comm.useMPICollectives()) {
    MPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                 recvcounts, displs, recvtype, root, comm.get(), &m_request);
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

    memcpy(m_buffer, m_sendbuf, m_count * m_type_bytes);
    m_completed = true;
    return;

  }

  if (m_rank == root ) {
    m_total_recvcount = displs[m_size - 1] + recvcounts[m_size - 1];
  }

  Ibcast(&m_total_recvcount, 1, MPI_INT, root, comm, &m_bcast_request, tag);
}

RBC::_internal::IgathervReq::~IgathervReq() {
  assert(m_recv_ptr == nullptr);
}

void RBC::_internal::IgathervReq::InitGather() {
  

  MPI_Aint lb, m_type_bytes_aint;
  MPI_Type_get_extent(m_datatype, &lb, &m_type_bytes_aint);
  m_type_bytes = static_cast<int>(m_type_bytes_aint);

  m_recv_ptr = static_cast<char*>(m_buffer);

  if (m_rank != m_root || m_root != 0) {
  
    m_recv_ptr = static_cast<char*>(malloc(m_type_bytes * m_total_recvcount));
  }
  
  memcpy(m_recv_ptr, m_sendbuf, m_type_bytes * m_count);
    
  m_zeroed_rank = _internal::RemoveBinomTreeRoot(m_root, m_rank, m_size);

  const int tailing_zeros = tlx::ffs(m_zeroed_rank) - 1;
  m_iterations = m_zeroed_rank > 0 ?
    tailing_zeros : tlx::integer_log2_ceil(m_size);

  m_recved_count = m_count;

  // Start sending a message.
  if (m_iteration != m_iterations) {

    const int source = m_zeroed_rank + (1 << m_iteration);

    if (source >= m_size) {
      m_iteration = m_iterations;
      return;
    }
    
    const int rooted_source = _internal::AddBinomTreeRoot(m_root, source, m_size);

    const auto cnt = m_total_recvcount - m_recved_count;
    auto ptr = static_cast<char*>(m_recv_ptr) + m_type_bytes * m_recved_count;

    Irecv(ptr, cnt, m_datatype, rooted_source, m_tag, m_comm, &m_request);
    
    m_recv_posted = true;

    ++m_iteration;

    return;
  }

  // Send a message.
  if (m_zeroed_rank > 0) {

    const int target = m_zeroed_rank - (1 << m_iterations);
    const int rooted_target = _internal::AddBinomTreeRoot(m_root, target, m_size);

    Isend(static_cast<char*>(m_recv_ptr), m_recved_count, m_datatype, rooted_target,
          m_tag, m_comm, &m_request);
    m_send_posted = true;

    return;
  }

  assert(m_size == 1);
  assert(m_recv_ptr == nullptr);
  m_completed = true;
}

int RBC::_internal::IgathervReq::test(int* flag, MPI_Status* status) {
  if (m_completed) {
    *flag = 1;
    return 0;
  }

  if (m_mpi_collective) {
    const auto err = MPI_Test(&m_request, flag, status);
    m_completed = *flag;
    return err;
  }

  *flag = 0;

  if (!m_bcast_finished) {
    
    int bcast_flag = 0;
    RBC::Test(&m_bcast_request, &bcast_flag, MPI_STATUS_IGNORE);
    
    if (bcast_flag == 1) {

    m_bcast_finished = true;
      assert(m_total_recvcount >= 0);
      
      if (m_total_recvcount == 0) {

        *flag = 1;
        m_completed = true;

        return 0;

      } else {
        
        InitGather();

      }
    } else {
      return 0;
    }
  }

  // Complete pending request.
  if (m_request != MPI_REQUEST_NULL) {

    int completed = 0;
    MPI_Status status;
    MPI_Test(&m_request, &completed, &status);
    
    if (completed) {

      if (m_recv_posted) {

        m_recv_posted = false;
        
        int count = 0;
        MPI_Get_count(&status, m_datatype, &count);
        
        m_recved_count += count;
        
      }
      
      m_request = MPI_REQUEST_NULL;
      
    }

  }

  // No pending request, receive data.
  if (m_request == MPI_REQUEST_NULL && m_iteration != m_iterations) {

    const int source = m_zeroed_rank + (1 << m_iteration);

    if (source < m_size) {

      const int rooted_source = _internal::AddBinomTreeRoot(m_root, source, m_size);
      // const int count = std::min(m_count << m_iteration, m_count * (m_size - source));

      const auto cnt = m_total_recvcount - m_recved_count;
      auto ptr = static_cast<char*>(m_recv_ptr) + m_type_bytes * m_recved_count;
      Irecv(ptr, cnt, m_datatype, rooted_source, m_tag, m_comm, &m_request);
      m_recv_posted = true;

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

      const size_t right_bytes = m_type_bytes * m_displs[m_root];
      const size_t left_bytes = m_type_bytes * m_recved_count - right_bytes;
    
      memcpy(m_buffer, static_cast<char*>(m_recv_ptr) + left_bytes, right_bytes);
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
