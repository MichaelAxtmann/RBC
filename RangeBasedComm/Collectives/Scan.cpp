/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cassert>

#include "../RBC.hpp"

namespace RBC {

    int Scan(const void* sendbuf, void* recvbuf, int count,
            MPI_Datatype datatype, MPI_Op op, Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Scan(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, comm.mpi_comm);
        }

        int rank;
        Comm_rank(comm, &rank);
        MPI_Aint lb, type_size;
        MPI_Type_get_extent(datatype, &lb, &type_size);
        int datatype_size = static_cast<int> (type_size);
        int recv_size = count * datatype_size;
        char *scan_buf = new char[recv_size];

        Exscan(sendbuf, scan_buf, count, datatype, op, comm);
        char *buf = const_cast<char*> (static_cast<const char*> (sendbuf));
        std::memcpy(recvbuf, buf, recv_size);
        if (rank != 0)
            MPI_Reduce_local(scan_buf, recvbuf, count, datatype, op);

        delete[] scan_buf;
        return 0;
    }

    namespace _internal {

        namespace optimized {

            int Scan(const void* sendbuf, void* recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, Comm const &comm) {
                if (comm.useMPICollectives()) {
                    return MPI_Scan(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, comm.mpi_comm);
                }

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(datatype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int recv_size = count * datatype_size;

                std::memcpy(recvbuf, sendbuf, recv_size);

                if (size == 0) return 0;

                char *tmp_buf = new char[recv_size];
                char *scan_buf = new char[recv_size];
                std::memcpy(scan_buf, sendbuf, recv_size);

                int commute = 0;
                MPI_Op_commutative(op, &commute);

                int mask = 1;
                while (mask < size) {
                    const int target = rank ^ mask;
                    mask <<= 1;

                    if (target < size) {
                        RBC::Sendrecv(scan_buf,
                                count,
                                datatype,
                                target,
                                Tag_Const::SCAN,
                                tmp_buf,
                                count,
                                datatype,
                                target,
                                Tag_Const::SCAN,
                                comm,
                                MPI_STATUS_IGNORE);

                        const bool left_target = target < rank;
                        if (left_target) {
                            MPI_Reduce_local(tmp_buf, scan_buf, count, datatype, op);
                            MPI_Reduce_local(tmp_buf, recvbuf, count, datatype, op);
                        } else {
                            if (commute) {
                                MPI_Reduce_local(tmp_buf, scan_buf, count, datatype, op);
                            } else {
                                MPI_Reduce_local(scan_buf, tmp_buf, count, datatype, op);
                                std::memcpy(scan_buf, tmp_buf, recv_size);
                            }
                        }
                    }
                }
                
                delete[] tmp_buf;
                delete[] scan_buf;
                return 0;
            }
            
        } // end namespace optimized

        /*
         * Request for the scan
         */
        class IscanReq : public RequestSuperclass {
        public:
            IscanReq(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    int tag, MPI_Op op, Comm const &comm);
            ~IscanReq();
            int test(int *flag, MPI_Status *status);
        private:
            const void *sendbuf;
            void *recvbuf;
            int count, recv_size, rank, exscan_completed;
            MPI_Datatype datatype;
            MPI_Op op;
            bool completed, mpi_collective;
            char *scan_buf;
            Request req_exscan;
            MPI_Request mpi_req;
        };
    }

    int Iscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, Comm const &comm, Request* request, int tag) {
        request->set(std::make_shared<_internal::IscanReq>(sendbuf, recvbuf,
                count, datatype, tag, op, comm));
        return 0;
    }
}

RBC::_internal::IscanReq::IscanReq(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const &comm) :
sendbuf(sendbuf), recvbuf(recvbuf), count(count), exscan_completed(0),
datatype(datatype), op(op), completed(false), mpi_collective(false),
scan_buf(nullptr) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm.mpi_comm, &mpi_req);
        mpi_collective = true;
        return;
    }
#endif
    RBC::Comm_rank(comm, &rank);
    int datatype_size;
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_size = static_cast<int> (type_size);
    recv_size = count * datatype_size;
    scan_buf = new char[recv_size];
    RBC::Iexscan(sendbuf, scan_buf, count, datatype, op, comm, &req_exscan, tag);
}

RBC::_internal::IscanReq::~IscanReq() {
    if (scan_buf != nullptr)
        delete[] scan_buf;
}

int RBC::_internal::IscanReq::test(int* flag, MPI_Status* status) {
    if (completed) {
        *flag = 1;
        return 0;
    }

    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    if (!exscan_completed) {
        RBC::Test(&req_exscan, &exscan_completed, MPI_STATUS_IGNORE);
    } else {
        char *buf = const_cast<char*> (static_cast<const char*> (sendbuf));
        std::memcpy(recvbuf, buf, recv_size);
        if (rank != 0)
            MPI_Reduce_local(scan_buf, recvbuf, count, datatype, op);
        completed = true;
    }

    if (completed) {
        *flag = 1;
        return 0;
    }
    return 0;
}

