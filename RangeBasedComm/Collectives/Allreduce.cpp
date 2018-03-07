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

#include "../RBC.hpp"

namespace RBC {

    int Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Allreduce(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, comm.mpi_comm);
        }

        MPI_Aint lb, type_size;
        MPI_Type_get_extent(datatype, &lb, &type_size);
        int datatype_size = static_cast<int> (type_size);
        char* scan_buf = new char[count * datatype_size];

        ScanAndBcast(sendbuf, scan_buf, recvbuf, count, datatype, op,
                comm);

        delete[] scan_buf;
        return 0;
    }

    namespace _internal {

        /*
         * Request for the reduce
         */
        class IallreduceReq : public RequestSuperclass {
        public:
            IallreduceReq(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    int tag, MPI_Op op, Comm const &comm);
            ~IallreduceReq();
            int test(int *flag, MPI_Status *status);
        private:
            const void *sendbuf;
            void *recvbuf;
            int count, tag, rank, size, new_rank, height, own_height,
            datatype_size, recv_size, receives;
            MPI_Datatype datatype;
            MPI_Op op;
            Comm comm;
            bool send, completed, mpi_collective;
            char *recvbuf_arr, *reduce_buf, *scan_buf;
            Request request;
            std::vector<Request> recv_requests;
            MPI_Request mpi_req;
        };
    }

    int Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, Comm const &comm, Request* request, int tag) {
        request->set(std::make_shared<_internal::IallreduceReq>(sendbuf, recvbuf,
                count, datatype, tag, op, comm));
        return 0;
    }
}

RBC::_internal::IallreduceReq::IallreduceReq(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const &comm) :
sendbuf(sendbuf), recvbuf(recvbuf), count(count), tag(tag),
datatype(datatype), op(op), comm(comm), send(false), completed(false),
mpi_collective(false), recvbuf_arr(nullptr), reduce_buf(nullptr),
scan_buf(nullptr) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Iallreduce(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, comm.mpi_comm,
                &mpi_req);
        mpi_collective = true;
        return;
    }
#endif

    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_size = static_cast<int> (type_size);
    scan_buf = new char[count * datatype_size];

    RBC::IscanAndBcast(sendbuf, scan_buf, recvbuf, count, datatype, op,
            comm, &request, tag);
}

RBC::_internal::IallreduceReq::~IallreduceReq() {
    if (scan_buf != nullptr)
        delete[] scan_buf;
}

int RBC::_internal::IallreduceReq::test(int* flag, MPI_Status* status) {
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    RBC::Test(&request, flag, status);

    return 0;
}

