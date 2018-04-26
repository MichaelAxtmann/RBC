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

        int root = 0;
        RBC::Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        RBC::Bcast(recvbuf, count, datatype, root, comm);
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
            bool first_round;
            Request reduce_request;
            Request bcast_request;
            const void *sendbuf;
            void *recvbuf;
            int count, tag;
            MPI_Datatype datatype;
            MPI_Op op;
            Comm comm;
            bool mpi_collective;
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
        MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const &comm)
    : first_round(true)
    , sendbuf(sendbuf)
    , recvbuf(recvbuf)
    , count(count)
    , tag(tag)
    , datatype(datatype)
    , op(op)
    , comm(comm)
    , mpi_collective(false) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Iallreduce(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, comm.mpi_comm,
                &mpi_req);
        mpi_collective = true;
        return;
    }
#endif

    RBC::Ireduce(sendbuf, recvbuf, count, datatype, op, 0, comm, &reduce_request);
}

RBC::_internal::IallreduceReq::~IallreduceReq() {}

int RBC::_internal::IallreduceReq::test(int* flag, MPI_Status* status) {
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    if (first_round) {
        RBC::Test(&reduce_request, flag, status);
        if (*flag) {
            RBC::Ibcast(recvbuf, count, datatype, 0, comm, &bcast_request);
            first_round = false;
        }
        *flag = 0;
    } else {
        RBC::Test(&bcast_request, flag, status);
    }

    return 0;
}

