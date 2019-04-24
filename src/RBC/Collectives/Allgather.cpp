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

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>

#include "RBC.hpp"
#include "tlx/math.hpp"

#include "../PointToPoint/Sendrecv.hpp"

namespace RBC {
int Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
              void* recvbuf, int recvcount, MPI_Datatype recvtype,
              Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgather(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                         comm.get());
  }

  Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
         0, comm);
  int size;
  Comm_size(comm, &size);
  Bcast(recvbuf, size * recvcount, recvtype, 0, comm);
  return 0;
}

int Allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
               void* recvbuf, const int* recvcounts, const int* displs,
               MPI_Datatype recvtype, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgatherv(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf,
                          const_cast<int*>(recvcounts),
                          const_cast<int*>(displs), recvtype, comm.get());
  }

  Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
          0, comm);
  int size;
  Comm_size(comm, &size);
  int total_recvcount = 0;
  for (int i = 0; i < size; i++)
    total_recvcount += recvcounts[i];
  Bcast(recvbuf, total_recvcount, recvtype, 0, comm);
  return 0;
}

int Allgatherm(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
               void* recvbuf, int recvcount,
               std::function<void(void*, void*, void*, void*, void*)> op, Comm const& comm) {
  Gatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, 0,
          op, comm);
  Bcast(recvbuf, recvcount, sendtype, 0, comm);
  return 0;
}

namespace _internal {
namespace optimized {
/*
 * AllgatherPipeline: Allgather algorithm with running time
 * O(alpha * log p + beta n log p).  p must be a power of two!!!
 */
double AllgatherPipelineExpRunningTime(Comm const& comm, int sendcount, MPI_Datatype sendtype,
                                       bool* valid) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  // Valid if number of processes is a power of two.
  *valid = true;

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int n = sendcount * datatype_size * size;               // In bytes

  return kALPHA * (size - 1) +
         kBETA * n * (size - 1) / size;
}


int AllgatherPipeline(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgather(const_cast<void*>(sendbuf),
                         sendcount, sendtype, recvbuf, recvcount, recvtype,
                         comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = sendcount * datatype_size;

  if (sendcount == 0) {
    return 0;
  }

  // Move local input to output buffer.
  memcpy(static_cast<char*>(recvbuf) + recv_size * rank, sendbuf, recv_size);

  if (size == 1) {
    return 0;
  }

  const int target = (rank + 1) % size;
  const int source = (rank - 1 + size) % size;

  for (int it = 0; it != size - 1; ++it) {
    int recv_pe = (size + rank - it - 1) % size;
    int send_pe = (size + rank - it) % size;

    Sendrecv(static_cast<char*>(recvbuf) + recv_size * send_pe,
             sendcount,
             sendtype,
             target,
             Tag_Const::ALLGATHER,
             static_cast<char*>(recvbuf) + recv_size * recv_pe,
             recvcount,
             recvtype,
             source,
             Tag_Const::ALLGATHER,
             comm,
             MPI_STATUS_IGNORE);
  }

  return 0;
}

/*
 * AllgatherDissemination: Allgather algorithm with running time
 * O(alpha * log p + beta n).  Compared to the one for hypercubes,
 * we have to copy the data once
 */
double AllgatherDisseminationExpRunningTime(Comm const& comm, int sendcount, MPI_Datatype sendtype,
                                            bool* valid) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  *valid = true;

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int n = sendcount * datatype_size * size;               // In bytes

  // We expect that copying the data takes beta/6 time.
  return kALPHA * tlx::integer_log2_ceil(size) +
         kBETA * n * (size - 1) / size +
         kBETA * n / 6;
}

int AllgatherDissemination(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                           void* recvbuf, int recvcount, MPI_Datatype recvtype,
                           Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgather(const_cast<void*>(sendbuf),
                         sendcount, sendtype, recvbuf, recvcount, recvtype,
                         comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  if (sendcount == 0) {
    return 0;
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = recvcount * datatype_size;

  if (size == 1) {
    // Move local input to output buffer.
    memcpy(static_cast<char*>(recvbuf), sendbuf, recv_size);

    return 0;
  }

  std::unique_ptr<char[]> tmpbuf_arr = std::make_unique<char[]>(recv_size * size);

  // Move local input to temp buffer.
  memcpy(tmpbuf_arr.get(), sendbuf, recv_size);

  int cnt = recvcount;
  int offset = 1;

  // First floor(log(p)) rounds with exp increasing msg size.
  while (offset <= size / 2) {
    int source = (rank + offset) % size;
    // + size to avoid negative numbers
    int target = (rank - offset + size) % size;

    SendrecvNonZeroed(tmpbuf_arr.get(),
                      cnt,
                      sendtype,
                      target,
                      Tag_Const::ALLGATHER,
                      tmpbuf_arr.get() + cnt * datatype_size,
                      cnt,
                      sendtype,
                      source,
                      Tag_Const::ALLGATHER,
                      comm,
                      MPI_STATUS_IGNORE);

    cnt *= 2;
    offset *= 2;
  }

  // Last round to exchange remaining elements if size is not a power of two.
  if (offset < size) {
    const int remaining = recvcount * size - cnt;
    int source = (rank + offset) % size;
    // + size to avoid negative numbers
    int target = (rank - offset + size) % size;

    SendrecvNonZeroed(tmpbuf_arr.get(),
                      remaining,
                      sendtype,
                      target,
                      Tag_Const::ALLGATHER,
                      tmpbuf_arr.get() + cnt * datatype_size,
                      remaining,
                      sendtype,
                      source,
                      Tag_Const::ALLGATHER,
                      comm,
                      MPI_STATUS_IGNORE);
  }

  /* Copy data to recvbuf. All PEs but PE 0 have to move their
   * data in two steps */
  if (rank == 0) {
    memcpy(recvbuf, tmpbuf_arr.get(), recv_size * size);
  } else {
    memcpy(static_cast<char*>(recvbuf) + recv_size * rank, tmpbuf_arr.get(), recv_size * (size - rank));
    memcpy(static_cast<char*>(recvbuf), tmpbuf_arr.get() + recv_size * (size - rank), recv_size * rank);
  }

  return 0;
}


/* Allgather operation but any process is allowed to choose its own input size.
 * @param sendcount Number of elements provided by this process
 * @param recvcount Total number of distributed elements.
 */
int AllgathervDissemination(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, int recvcount, MPI_Datatype recvtype,
                            Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgather(const_cast<void*>(sendbuf),
                         sendcount, sendtype, recvbuf, recvcount, recvtype,
                         comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  if (recvcount == 0) {
    return 0;
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  const size_t datatype_size = static_cast<size_t>(type_size);
  const size_t recv_size = recvcount * datatype_size;
  const size_t send_size = sendcount * datatype_size;

  if (size == 1) {
    // Move local input to output buffer.
    memcpy(static_cast<char*>(recvbuf), sendbuf, recv_size);

    return 0;
  }

  int count_exscan = 0;
  RBC::_internal::optimized::Exscan(&sendcount, &count_exscan, 1,
                                    MPI_INT, MPI_SUM, comm);

  std::unique_ptr<char[]> tmpbuf_arr = std::make_unique<char[]>(recv_size);

  // Move local input to temp buffer.
  memcpy(tmpbuf_arr.get(), sendbuf, send_size);

  int cnt = sendcount;
  int offset = 1;

  // First floor(log(p)) rounds with exp increasing msg size.
  while (offset <= size / 2) {
    int source = (rank + offset) % size;
    // + size to avoid negative numbers
    int target = (rank - offset + size) % size;

    MPI_Request requests[2];
    RBC::Isend(tmpbuf_arr.get(), cnt, sendtype, target, Tag_Const::ALLGATHER, comm, requests);

    int recv_cnt = 0;
    MPI_Status status;
    RBC::Probe(source, Tag_Const::ALLGATHER, comm, &status);
    MPI_Get_count(&status, sendtype, &recv_cnt);

    RBC::Irecv(tmpbuf_arr.get() + cnt * datatype_size, recv_cnt, recvtype, source,
               Tag_Const::ALLGATHER, comm, requests + 1);

    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    cnt += recv_cnt;
    offset *= 2;
  }

  int remaining = recvcount - cnt;
  // Last round to exchange remaining elements if size is not a power of two.
  if (offset < size) {
    int source = (rank + offset) % size;
    // + size to avoid negative numbers
    int target = (rank - offset + size) % size;

    int remote_remaining = 0;
    Sendrecv(&remaining,
             1,
             MPI_INT,
             source,
             Tag_Const::ALLGATHER,
             &remote_remaining,
             1,
             MPI_INT,
             target,
             Tag_Const::ALLGATHER,
             comm,
             MPI_STATUS_IGNORE);

    SendrecvNonZeroed(tmpbuf_arr.get(),
                      remote_remaining,
                      sendtype,
                      target,
                      Tag_Const::ALLGATHER,
                      tmpbuf_arr.get() + cnt * datatype_size,
                      remaining,
                      sendtype,
                      source,
                      Tag_Const::ALLGATHER,
                      comm,
                      MPI_STATUS_IGNORE);
  }

  int sendcount_exscan = 0;
  RBC::_internal::optimized::Exscan(&sendcount, &sendcount_exscan,
                                    1, MPI_INT, MPI_SUM, comm);

  /* Copy data to recvbuf. All PEs but PE 0 have to move their
   * data in two steps */
  if (rank == 0) {
    memcpy(recvbuf, tmpbuf_arr.get(), recv_size);
  } else {
    const size_t byte_rotation = sendcount_exscan * datatype_size;
    memcpy(static_cast<char*>(recvbuf), tmpbuf_arr.get() + recv_size - byte_rotation, byte_rotation);
    memcpy(static_cast<char*>(recvbuf) + byte_rotation, tmpbuf_arr.get(), recv_size - byte_rotation);
  }

  return 0;
}

/*
 * AllgatherHypercube: Allgather algorithm with running time
 * O(alpha * log p + beta n log p).  p must be a power of two!!!
 */
double AllgatherHypercubeExpRunningTime(Comm const& comm, int sendcount,
                                        MPI_Datatype sendtype, bool* valid) {
  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  // Valid if number of processes is a power of two.
  *valid = (size & (size - 1)) == 0;

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int n = sendcount * datatype_size * size;               // In bytes

  return kALPHA * tlx::integer_log2_ceil(size) +
         kBETA * n * (size - 1) / size;
}


int AllgatherHypercube(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int recvcount, MPI_Datatype recvtype,
                       Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allgather(const_cast<void*>(sendbuf),
                         sendcount, sendtype, recvbuf, recvcount, recvtype,
                         comm.get());
  }

  int rank, size;
  Comm_rank(comm, &rank);
  Comm_size(comm, &size);

  if (sendcount == 0) {
    return 0;
  }

  // size must be a power of two to execute the hypercube version.
  if ((size & (size - 1))) {
    printf("%s \n", "Warning: p is not a power of two. Fallback to RBC::Allgather");
    return RBC::Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(sendtype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = recvcount * datatype_size;

  if (size == 1) {
    // Move local input to output buffer.
    memcpy(static_cast<char*>(recvbuf) + recv_size * rank, sendbuf, recv_size);
    return 0;
  }

  // Move local input to output buffer.
  memcpy(static_cast<char*>(recvbuf) + recv_size * rank, sendbuf, recv_size);

  int cnt = recvcount;
  char* recvbuf_ptr = static_cast<char*>(recvbuf) + recv_size * rank;
  const size_t log_p = std::log2(size);
  for (size_t it = 0; it != log_p; ++it) {
    int target = rank ^ 1 << it;
    bool left_target = target < rank;

    if (left_target) {
      SendrecvNonZeroed(recvbuf_ptr,
                        cnt,
                        sendtype,
                        target,
                        Tag_Const::ALLGATHER,
                        recvbuf_ptr - cnt * datatype_size,
                        cnt,
                        recvtype,
                        target,
                        Tag_Const::ALLGATHER,
                        comm,
                        MPI_STATUS_IGNORE);
      recvbuf_ptr -= cnt * datatype_size;
    } else {
      SendrecvNonZeroed(recvbuf_ptr,
                        cnt,
                        sendtype,
                        target,
                        Tag_Const::ALLGATHER,
                        recvbuf_ptr + cnt * datatype_size,
                        cnt,
                        recvtype,
                        target,
                        Tag_Const::ALLGATHER,
                        comm,
                        MPI_STATUS_IGNORE);
    }

    cnt *= 2;
  }

  return 0;
}

/*
 *
 * Blocking allgather with equal amount of elements on each process
 * This method uses different implementations depending on the
 * size of comm and the input size.
 */
int Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
              void* recvbuf, int recvcount, MPI_Datatype recvtype,
              Comm const& comm) {
  bool valid_pipeline;
  double pipeline = AllgatherPipelineExpRunningTime(comm, sendcount, sendtype,
                                                    &valid_pipeline);
  bool valid_hypercube;
  double hypercube = AllgatherHypercubeExpRunningTime(comm, sendcount, sendtype,
                                                      &valid_hypercube);
  bool valid_dissemination;
  double dissemination = AllgatherDisseminationExpRunningTime(comm, sendcount, sendtype,
                                                              &valid_dissemination);

  if (valid_hypercube) {
    if (hypercube < pipeline) {
      return AllgatherHypercube(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype,
                                comm);
    } else {
      return AllgatherDissemination(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcount, recvtype,
                                    comm);
    }
  } else {
    if (dissemination < pipeline) {
      return AllgatherDissemination(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcount, recvtype,
                                    comm);
    } else {
      return AllgatherPipeline(sendbuf, sendcount, sendtype,
                               recvbuf, recvcount, recvtype,
                               comm);
    }
  }
}
}         // end namespace optimized

/*
 * Request for the allgather
 */
class IallgatherReq : public RequestSuperclass {
 public:
  IallgatherReq(const void* sendbuf, int sendcount,
                MPI_Datatype sendtype, void* recvbuf, int recvcount,
                MPI_Datatype recvtype,
                int tag, Comm const& comm);

  int test(int* flag, MPI_Status* status) override;

 private:
  void* m_recvbuf;
  int m_tag, m_total, m_gather_completed, m_bcast_completed;
  MPI_Datatype m_recvtype;
  Comm m_comm;
  bool m_mpi_collective;
  Request m_req_gather, m_req_bcast;
  MPI_Request m_mpi_req;
};

/*
 * Request for the allgather
 */
class IallgathervReq : public RequestSuperclass {
 public:
  IallgathervReq(const void* sendbuf, int sendcount,
                 MPI_Datatype sendtype, void* recvbuf,
                 const int* recvcounts, const int* displs, MPI_Datatype recvtype,
                 int tag, Comm const& comm);

  int test(int* flag, MPI_Status* status) override;

 private:
  void* m_recvbuf;
  int m_tag, m_total_recvcount, m_gather_completed, m_bcast_completed;
  MPI_Datatype m_recvtype;
  Comm m_comm;
  bool m_mpicollective;
  Request m_req_gather, m_req_bcast;
  MPI_Request m_mpi_req;
};

/*
 * Request for the allgather
 */
class IallgathermReq : public RequestSuperclass {
 public:
  IallgathermReq(const void* sendbuf, int sendcount,
                 MPI_Datatype sendtype, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype,
                 int tag, std::function<void(void*, void*, void*, void*, void*)> op,
                 Comm const& comm);

  int test(int* flag, MPI_Status* status) override;

 private:
  void* m_recvbuf;
  int m_tag, m_total_recvcount, m_gather_completed, m_bcast_completed;
  MPI_Datatype m_recvtype;
  std::function<void(void*, void*, void*, void*, void*)> m_op;
  Comm m_comm;
  bool m_mpi_collective;
  Request m_req_gather, m_req_bcast;
  MPI_Request m_mpi_req;
};
}     // end namespace _internal

int Iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
               void* recvbuf, int recvcount, MPI_Datatype recvtype,
               Comm const& comm, Request* request, int tag) {
  request->set(std::make_shared<_internal::IallgatherReq>(sendbuf, sendcount,
                                                          sendtype, recvbuf, recvcount, recvtype,
                                                          tag, comm));
  return 0;
}

int Iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, const int* recvcounts, const int* displs,
                MPI_Datatype recvtype, Comm const& comm,
                Request* request, int tag) {
  request->set(std::make_shared<_internal::IallgathervReq>(sendbuf, sendcount,
                                                           sendtype, recvbuf,
                                                           recvcounts, displs,
                                                           recvtype, tag, comm));
  return 0;
}

int Iallgatherm(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                void* recvbuf, int recvcount,
                std::function<void(void*, void*, void*, void*, void*)> op, Comm const& comm,
                Request* request, int tag) {
  request->set(std::make_shared<_internal::IallgathermReq>(sendbuf, sendcount,
                                                           sendtype, recvbuf, recvcount, sendtype, tag,
                                                           op, comm));
  return 0;
}
}  // end namespace RBC

RBC::_internal::IallgatherReq::IallgatherReq(const void* sendbuf, int sendcount,
                                             MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                             MPI_Datatype recvtype, int tag, Comm const& comm) :
  m_recvbuf(recvbuf),
  m_tag(tag),
  m_total(comm.getSize() * recvcount),
  m_gather_completed(0),
  m_bcast_completed(0),
  m_recvtype(recvtype),
  m_comm(comm),
  m_mpi_collective(false) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (m_comm.useMPICollectives()) {
    MPI_Iallgather(sendbuf, sendcount, sendtype, m_recvbuf, recvcount, m_recvtype,
                   m_comm.get(), &m_mpi_req);
    m_mpi_collective = true;
    return;
  }
#endif
  int size;
  RBC::Comm_size(m_comm, &size);

  int root = 0;
  RBC::Igather(sendbuf, sendcount, sendtype, m_recvbuf, recvcount, m_recvtype,
               root, m_comm, &m_req_gather, m_tag);
}

int RBC::_internal::IallgatherReq::test(int* flag, MPI_Status* status) {
  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (m_bcast_completed) {
    *flag = 1;
    return 0;
  }

  if (!m_gather_completed) {
    RBC::Test(&m_req_gather, &m_gather_completed, MPI_STATUS_IGNORE);
    if (m_gather_completed) {
      RBC::Ibcast(m_recvbuf, m_total, m_recvtype, 0, m_comm,
                  &m_req_bcast, m_tag);
    }
  } else {
    RBC::Test(&m_req_bcast, &m_bcast_completed, MPI_STATUS_IGNORE);
  }

  if (m_bcast_completed)
    *flag = 1;

  return 0;
}


RBC::_internal::IallgathervReq::IallgathervReq(const void* sendbuf, int sendcount,
                                               MPI_Datatype sendtype, void* recvbuf,
                                               const int* recvcounts, const int* displs,
                                               MPI_Datatype recvtype,
                                               int tag, Comm const& comm) :
  m_recvbuf(recvbuf),
  m_tag(tag),
  m_total_recvcount(std::accumulate(recvcounts, recvcounts + comm.getSize(), 0)),
  m_gather_completed(0),
  m_bcast_completed(0),
  m_recvtype(recvtype),
  m_comm(comm),
  m_mpicollective(false) {
#ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
  if (m_comm.useMPICollectives()) {
    MPI_Iallgatherv(sendbuf, sendcount, sendtype, m_recvbuf,
                    recvcounts, displs, m_recvtype, m_comm.get(), &m_mpi_req);
    m_mpicollective = true;
    return;
  }
#endif
  int size;
  RBC::Comm_size(m_comm, &size);

  int root = 0;
  RBC::Igatherv(sendbuf, sendcount, sendtype, m_recvbuf, recvcounts, displs,
                m_recvtype, root, m_comm, &m_req_gather, m_tag);
  // m_total_recvcount = 0;
}

int RBC::_internal::IallgathervReq::test(int* flag, MPI_Status* status) {
  if (m_mpicollective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (m_bcast_completed) {
    *flag = 1;
    return 0;
  }

  if (!m_gather_completed) {
    RBC::Test(&m_req_gather, &m_gather_completed, MPI_STATUS_IGNORE);
    if (m_gather_completed) {
      std::cout << "send " << m_total_recvcount << std::endl;
      RBC::Ibcast(m_recvbuf, m_total_recvcount, m_recvtype, 0, m_comm,
                  &m_req_bcast, m_tag);
    }
  } else {
    RBC::Test(&m_req_bcast, &m_bcast_completed, MPI_STATUS_IGNORE);
  }

  if (m_bcast_completed)
    *flag = 1;

  return 0;
}

RBC::_internal::IallgathermReq::IallgathermReq(const void* sendbuf, int sendcount,
                                               MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                               MPI_Datatype recvtype,
                                               int tag,
                                               std::function<void(void*, void*, void*, void*, void*)> op,
                                               Comm const& comm) :
  m_recvbuf(recvbuf),
  m_tag(tag),
  m_total_recvcount(recvcount),
  m_gather_completed(0),
  m_bcast_completed(0),
  m_recvtype(recvtype),
  m_op(op),
  m_comm(comm),
  m_mpi_collective(false) {
// #ifndef NO_NONBLOCKING_COLL_MPI_SUPPORT
// This method is not part of MPI
// #endif
  int size;
  RBC::Comm_size(m_comm, &size);

  int root = 0;
  RBC::Igatherm(sendbuf, sendcount, sendtype, m_recvbuf, recvcount,
                root, m_op, m_comm, &m_req_gather, tag);
}

int RBC::_internal::IallgathermReq::test(int* flag, MPI_Status* status) {
  if (m_mpi_collective)
    return MPI_Test(&m_mpi_req, flag, status);

  if (m_bcast_completed) {
    *flag = 1;
    return 0;
  }

  if (!m_gather_completed) {
    RBC::Test(&m_req_gather, &m_gather_completed, MPI_STATUS_IGNORE);
    if (m_gather_completed) {
      RBC::Ibcast(m_recvbuf, m_total_recvcount, m_recvtype, 0, m_comm,
                  &m_req_bcast, m_tag);
    }
  } else {
    RBC::Test(&m_req_bcast, &m_bcast_completed, MPI_STATUS_IGNORE);
  }

  if (m_bcast_completed)
    *flag = 1;

  return 0;
}
