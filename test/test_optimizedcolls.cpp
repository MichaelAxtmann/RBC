/*****************************************************************************
 * This file is part of the Project JanusSortRBC
 *
 * Copyright (c) 2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "sstream"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <mpi.h>
#include <random>
#include <stdlib.h>
#include <vector>

#include "RBC.hpp"

#include "tlx/algorithm.hpp"
#include "tlx/math.hpp"

// #define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;
#define PRINT_ROOT(msg) ;

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "]";
  }
  return out;
}

void GenerateData(std::vector<long>& send, std::vector<long>& recv) {
  for (size_t idx = 0; idx != send.size(); ++idx) {
    send[idx] = rand();
  }
  for (size_t idx = 0; idx != recv.size(); ++idx) {
    recv[idx] = 0;
  }
}

#define PrintDistributed(string)              \
  { std::stringstream buffer;                 \
    buffer << rank << ": " << string << "\n"; \
    MPI_File_write_ordered(fh, buffer.str().c_str(), buffer.str().size(), MPI_CHAR, MPI_STATUS_IGNORE); }

void merge(const void* begin1, const void* end1,
           const void* begin2, const void* end2,
           void* out) {
  std::merge<const long*, const long*, long*>(static_cast<const long*>(begin1), static_cast<const long*>(end1),
                                              static_cast<const long*>(begin2), static_cast<const long*>(end2), static_cast<long*>(out), std::less<long>());
}


int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (argc != 3) {
    std::cout << "Error: We expect two arguments. " << std::endl
              << "First argument: Pass 0 for MPI and pass 1 for RBC." << std::endl
              << "Second argument: Input size." << std::endl;
  }
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  bool mpi = atoi(argv[1]);
  int el_cnt = atoi(argv[2]);

  MPI_File fh;
  int err = 0;
  if (mpi) {
    err = MPI_File_open(comm, "out_mpi.log", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  } else {
    err = MPI_File_open(comm, "out_rbc.log", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  }
  if (err) {
    MPI_Abort(MPI_COMM_WORLD, 911);
  }

  srand(rank + 1);

  MPI_Datatype type = MPI_LONG;

  std::vector<long> send(el_cnt);
  std::vector<long> recv(el_cnt* size);

  PRINT_ROOT("Iallgather");
  if (mpi) {
    std::vector<long> send(el_cnt);
    std::vector<long> recv(el_cnt* size);

    GenerateData(send, recv);

    RBC::Comm mpi_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &mpi_comm, true, true, true);
    RBC::Request request;
    RBC::Iallgather(send.data(), send.size(), type, recv.data(), send.size(), type, mpi_comm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgather: " << std::endl << send << std::endl << recv);
  } else {
    std::vector<long> send(el_cnt);
    std::vector<long> recv(el_cnt* size);

    GenerateData(send, recv);

    RBC::Request request;
    RBC::Iallgather(send.data(), send.size(), type, recv.data(), send.size(), type, rcomm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgather: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Allgatherv");
  if (mpi) {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    RBC::Comm mpi_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &mpi_comm, true, true, true);
    RBC::Allgatherv(send.data(), send.size(), type, recv.data(), recvcnts.data(), displs.data(), type, mpi_comm);
    PrintDistributed("Allgatherv: " << std::endl << send << std::endl << recv);
  } else {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    RBC::Allgatherv(send.data(), send.size(), type, recv.data(), recvcnts.data(), displs.data(), type, rcomm);
    PrintDistributed("Allgatherv: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Iallgatherv");
  if (mpi) {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    RBC::Comm mpi_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &mpi_comm, true, true, true);
    RBC::Request request;
    RBC::Iallgatherv(send.data(), send.size(), type, recv.data(), recvcnts.data(), displs.data(), type, mpi_comm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgatherv: " << std::endl << send << std::endl << recv);
  } else {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    RBC::Request request;
    RBC::Iallgatherv(send.data(), send.size(), type, recv.data(), recvcnts.data(), displs.data(), type, rcomm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgatherv: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Allgatherm");
  if (mpi) {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    auto merger = [](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    // todo out of place merging?
                    std::merge((long*)begin1, (long*)end1, (long*)begin2, (long*)end2, (long*)result);
                  };

    RBC::Comm mpi_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &mpi_comm, true, true, true);
    RBC::Allgatherm(send.data(), send.size(), type, recv.data(), recv.size(), merger, mpi_comm);
    PrintDistributed("Allgatherm: " << std::endl << send << std::endl << recv);
  } else {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    auto merger = [](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    // todo out of place merging?
                    std::merge((long*)begin1, (long*)end1, (long*)begin2, (long*)end2, (long*)result);
                  };

    RBC::Allgatherm(send.data(), send.size(), type, recv.data(), recv.size(), merger, rcomm);
    PrintDistributed("Allgatherm: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Iallgatherm");
  if (mpi) {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    auto merger = [](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    // todo out of place merging?
                    std::merge((long*)begin1, (long*)end1, (long*)begin2, (long*)end2, (long*)result);
                  };

    RBC::Comm mpi_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &mpi_comm, true, true, true);
    RBC::Request request;
    RBC::Iallgatherm(send.data(), send.size(), type, recv.data(), recv.size(), merger, mpi_comm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgatherm: " << std::endl << send << std::endl << recv);
  } else {
    std::vector<long> send(el_cnt + rank);
    std::vector<long> recv(el_cnt* size + (size - 1) * size / 2);

    std::vector<int> recvcnts(size);
    for (size_t i = 0; i != size; ++i) {
      recvcnts[i] = el_cnt + i;
    }

    std::vector<int> displs(size + 1);
    tlx::exclusive_scan(recvcnts.begin(), recvcnts.end(), displs.begin(), 0, std::plus<>{ });

    GenerateData(send, recv);

    auto merger = [](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    // todo out of place merging?
                    std::merge((long*)begin1, (long*)end1, (long*)begin2, (long*)end2, (long*)result);
                  };

    RBC::Request request;
    RBC::Iallgatherm(send.data(), send.size(), type, recv.data(), recv.size(), merger, rcomm, &request);
    RBC::Wait(&request, MPI_STATUS_IGNORE);
    PrintDistributed("Iallgatherm: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Allgather");
  if (mpi) {
    GenerateData(send, recv);
    MPI_Allgather(send.data(), el_cnt, type, recv.data(), el_cnt, type, comm);
    PrintDistributed("AllgatherDissemination: " << std::endl << send << std::endl << recv);

    if (tlx::is_power_of_two(size)) {
      GenerateData(send, recv);
      MPI_Allgather(send.data(), el_cnt, type, recv.data(), el_cnt, type, comm);
      PrintDistributed("AllgatherHypercube: " << std::endl << send << std::endl << recv);
    }

    GenerateData(send, recv);
    MPI_Allgather(send.data(), el_cnt, type, recv.data(), el_cnt, type, comm);
    PrintDistributed("AllgatherPipeline: " << std::endl << send << std::endl << recv);

    {
      std::vector<long> sendv(rand() % (el_cnt + 1));
      int send_cnt = sendv.size();
      std::vector<int> sizes(size);
      RBC::Allgather(&send_cnt, 1, MPI_INT, sizes.data(), 1, MPI_INT, rcomm);
      std::vector<int> sizes_exscan(size + 1, 0);
      tlx::exclusive_scan(sizes.begin(), sizes.end(), sizes_exscan.begin(), 0);
      std::vector<long> recvv(sizes_exscan.back());
      GenerateData(sendv, recvv);
      if (recvv.size()) MPI_Allgatherv(sendv.data(), sendv.size(),
                                       type, recvv.data(), sizes.data(), sizes_exscan.data(),
                                       type, comm);
      PrintDistributed("AllgathervDissemination: " << std::endl << sendv << std::endl << recvv);
    }
  } else {
    GenerateData(send, recv);
    RBC::_internal::optimized::AllgatherDissemination(send.data(), el_cnt, type, recv.data(), el_cnt, type, rcomm);
    PrintDistributed("AllgatherDissemination: " << std::endl << send << std::endl << recv);

    if (tlx::is_power_of_two(size)) {
      GenerateData(send, recv);
      RBC::_internal::optimized::AllgatherHypercube(send.data(), el_cnt, type, recv.data(), el_cnt, type, rcomm);
      PrintDistributed("AllgatherHypercube: " << std::endl << send << std::endl << recv);
    }

    GenerateData(send, recv);
    RBC::_internal::optimized::AllgatherPipeline(send.data(), el_cnt, type, recv.data(), el_cnt, type, rcomm);
    PrintDistributed("AllgatherPipeline: " << std::endl << send << std::endl << recv);

    {
      std::vector<long> sendv(rand() % (el_cnt + 1));
      int send_cnt = sendv.size();
      int recv_cnt = 0;
      RBC::Allreduce(&send_cnt, &recv_cnt, 1, MPI_INT, MPI_SUM, rcomm);
      std::vector<long> recvv(recv_cnt);
      GenerateData(sendv, recvv);
      RBC::_internal::optimized::AllgathervDissemination(sendv.data(), sendv.size(),
                                                         type, recvv.data(), recvv.size(), type, rcomm);
      PrintDistributed("AllgathervDissemination: " << std::endl << sendv << std::endl << recvv);
    }
  }

  PRINT_ROOT("(All)reduce");
  if (mpi) {
    GenerateData(send, recv);
    MPI_Allreduce(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("AllreduceScatterAllgather: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Allreduce(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("AllreduceHypercube: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Allreduce(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("AllreduceTwotree: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Allreduce(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("ScanAndBcast: " << std::endl << send << std::endl << recv);

    for (int root = 0; root != size; ++root) {
      GenerateData(send, recv);
      MPI_Reduce(send.data(), recv.data(), el_cnt, type, MPI_SUM, root, comm);
      PrintDistributed("ReduceTwotree: " << std::endl << send << std::endl << recv);
    }
  } else {
    PRINT_ROOT("scatter");
    GenerateData(send, recv);
    RBC::_internal::optimized::AllreduceScatterAllgather(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("AllreduceScatterAllgather: " << std::endl << send << std::endl << recv);

    PRINT_ROOT("hyper");
    GenerateData(send, recv);
    RBC::_internal::optimized::AllreduceHypercube(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("AllreduceHypercube: " << std::endl << send << std::endl << recv);

    PRINT_ROOT("twotree");
    GenerateData(send, recv);
    RBC::_internal::optimized::AllreduceTwotree(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("AllreduceTwotree: " << std::endl << send << std::endl << recv);

    PRINT_ROOT("twotree");
    GenerateData(send, recv);
    auto scan = recv;
    RBC::ScanAndBcast(send.data(), scan.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("ScanAndBcast: " << std::endl << send << std::endl << recv);

    for (int root = 0; root != size; ++root) {
      PRINT_ROOT("reduce twotree");
      GenerateData(send, recv);
      RBC::_internal::optimized::ReduceTwotree(send.data(), recv.data(), el_cnt, type, MPI_SUM, root, rcomm);
      PrintDistributed("ReduceTwotree: " << std::endl << send << std::endl << recv);
    }
  }

  PRINT_ROOT("Scan");
  if (mpi) {
    GenerateData(send, recv);
    MPI_Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("Scan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("Iscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("ScanOptimized: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("ScanTwotree: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("ScanAndBcast: " << std::endl << send << std::endl << recv);
  } else {
    GenerateData(send, recv);
    RBC::Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("Scan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    RBC::Request req;
    RBC::Iscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    PrintDistributed("Iscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    RBC::_internal::optimized::Scan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("ScanOptimized: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    RBC::_internal::optimized::ScanTwotree(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("ScanTwotree: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    auto bcast = recv;
    RBC::ScanAndBcast(send.data(), recv.data(), bcast.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("ScanAndBcast: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("Exscan");
  if (mpi) {
    GenerateData(send, recv);
    MPI_Exscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("Exscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Exscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("Iexscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    MPI_Exscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, comm);
    PrintDistributed("ExscanOptimized: " << std::endl << send << std::endl << recv);
  } else {
    GenerateData(send, recv);
    RBC::Exscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("Exscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    RBC::Request req;
    RBC::Iexscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    PrintDistributed("Iexscan: " << std::endl << send << std::endl << recv);

    GenerateData(send, recv);
    RBC::_internal::optimized::Exscan(send.data(), recv.data(), el_cnt, type, MPI_SUM, rcomm);
    PrintDistributed("ExscanOptimized: " << std::endl << send << std::endl << recv);
  }

  PRINT_ROOT("twotree");
  GenerateData(send, recv);
  RBC::_internal::optimized::ReduceTwotree(send.data(), recv.data(), el_cnt, type, MPI_SUM, 2, rcomm);
  PRINT_ROOT("Bcast");
  for (int root = 0; root != size; ++root) {
    if (mpi) {
      GenerateData(send, recv);
      MPI_Bcast(send.data(), el_cnt, type, root, comm);
      PrintDistributed("BcastBinomial: " << std::endl << send << std::endl << recv);

      GenerateData(send, recv);
      MPI_Bcast(send.data(), el_cnt, type, root, comm);
      PrintDistributed("BcastScatterAllgather: " << std::endl << send << std::endl << recv);

      GenerateData(send, recv);
      MPI_Bcast(send.data(), el_cnt, type, root, comm);
      PrintDistributed("BcastTwotree: " << std::endl << send << std::endl << recv);
    } else {
      GenerateData(send, recv);
      RBC::_internal::optimized::BcastBinomial(send.data(), el_cnt, type, root, rcomm);
      PrintDistributed("BcastBinomial: " << std::endl << send << std::endl << recv);

      GenerateData(send, recv);
      RBC::_internal::optimized::BcastScatterAllgather(send.data(), el_cnt, type, root, rcomm);
      PrintDistributed("BcastScatterAllgather: " << std::endl << send << std::endl << recv);

      GenerateData(send, recv);
      RBC::_internal::optimized::BcastTwotree(send.data(), el_cnt, type, root, rcomm);
      PrintDistributed("BcastTwotree: " << std::endl << send << std::endl << recv);
    }
  }

  // Finalize the MPI environment
  MPI_Finalize();
}
