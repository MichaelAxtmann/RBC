/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <mpi.h>

#include "../RBC.hpp"

namespace RBC {

    int Barrier(Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Barrier(comm.mpi_comm);
        }
        int a = 0, b = 0, c = 0;
        ScanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm);
        return 0;
    }

    namespace _internal {

        /*
         * Request for the barrier
         */
        class IbarrierReq : public RequestSuperclass {
        public:
            IbarrierReq(RBC::Comm const &comm);
            int test(int *flag, MPI_Status *status);

        private:
            Request request;
            MPI_Request mpi_req;
            bool mpi_collective;
            int a, b, c;
        };
    }

    int Ibarrier(Comm const &comm, Request* request) {
        request->set(std::make_shared<RBC::_internal::IbarrierReq>(comm));
        return 0;
    }
}

RBC::_internal::IbarrierReq::IbarrierReq(RBC::Comm const &comm) : mpi_collective(false),
a(0) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Ibarrier(comm.mpi_comm, &mpi_req);
        mpi_collective = true;
        return;
    }
#endif
    int tag = Tag_Const::BARRIER;
    IscanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm, &request, tag);
}

int RBC::_internal::IbarrierReq::test(int* flag, MPI_Status* status) {
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    return Test(&request, flag, MPI_STATUS_IGNORE);
}

