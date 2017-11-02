/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include "../RBC.hpp"
#include "../Requests.hpp"

int RBC::Barrier(RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Barrier(comm.mpi_comm);
    } 
    int a = 0, b = 0, c = 0;
    RBC::ScanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm);
    return 0;
}


/*
 * Request for the barrier
 */
class Range_Requests::Ibarrier : public RBC::R_Req {
public:
    Ibarrier(RBC::Comm const &comm);
    int test(int *flag, MPI_Status *status);

private:
    RBC::Request request;
    MPI_Request mpi_req;
    bool mpi_collective;
    int a, b, c;
};

int RBC::Ibarrier(RBC::Comm const &comm, RBC::Request* request) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ibarrier(comm));
    return 0;
}

Range_Requests::Ibarrier::Ibarrier(RBC::Comm const &comm) : mpi_collective(false),
    a(0) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Ibarrier(comm.mpi_comm, &mpi_req);
        mpi_collective = true;
        return;
    } 
#endif
    int tag = RBC::Tag_Const::BARRIER;
    RBC::IscanAndBcast(&a, &b, &c, 1, MPI_INT, MPI_SUM, comm, &request, tag);
}

int Range_Requests::Ibarrier::test(int* flag, MPI_Status* status) {
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    return RBC::Test(&request, flag, MPI_STATUS_IGNORE);
}
