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
#include "../Requests.hpp"

int RBC::Send(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, RBC::Comm const &comm) {
    return MPI_Send(const_cast <void*>(sendbuf), count, datatype,
                    comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm);
};

int RBC::Ssend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, RBC::Comm const &comm) {
    return MPI_Ssend(const_cast <void*>(sendbuf), count, datatype,
                    comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm);
};

/*
 * Request for the isend
 */
class Range_Requests::Isend : public RBC::R_Req {
public:
    Isend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm);
    int test(int *flag, MPI_Status *status);

private:
    const void *sendbuf;
    int count, dest, tag;
    MPI_Datatype datatype;
    RBC::Comm comm;
    bool requested;
    MPI_Request request;
};

int RBC::Isend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, RBC::Comm const &comm, RBC::Request *request) {
    *request  = std::unique_ptr<R_Req>(new Range_Requests::Isend(sendbuf, count, 
            datatype, dest, tag, comm));
    return 0;
};

Range_Requests::Isend::Isend(const void *sendbuf, int count, MPI_Datatype datatype,
        int dest, int tag, RBC::Comm const &comm) : sendbuf(sendbuf), count(count),
        dest(dest), tag(tag), datatype(datatype), comm(comm), requested(false) {
    void* buf = const_cast<void*>(sendbuf);
    MPI_Isend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm, &request);
};

int Range_Requests::Isend::test(int *flag, MPI_Status *status) {
    return MPI_Test(&request, flag, status);
};


/*
 * Request for the issend
 */
class Range_Requests::Issend : public RBC::R_Req {
public:
    Issend(const void *sendbuf, int count, MPI_Datatype datatype,
            int dest, int tag, RBC::Comm const &comm);
    int test(int *flag, MPI_Status *status);

private:
    const void *sendbuf;
    int count;
    MPI_Datatype datatype;
    int dest, tag;
    RBC::Comm comm;
    bool requested;
    MPI_Request request;
};

int RBC::Issend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
        int tag, RBC::Comm const &comm, RBC::Request *request) {
    *request  = std::unique_ptr<R_Req>(new Range_Requests::Issend(sendbuf, count, 
            datatype, dest, tag, comm));
    return 0;
};

Range_Requests::Issend::Issend(const void *sendbuf, int count, MPI_Datatype datatype,
        int dest, int tag, RBC::Comm const &comm) : sendbuf(sendbuf), count(count),
        datatype(datatype), dest(dest), tag(tag), comm(comm), requested(false) {
    void* buf = const_cast<void*>(sendbuf);
    MPI_Issend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag,
            comm.mpi_comm, &request);
};

int Range_Requests::Issend::test(int *flag, MPI_Status *status) {
    return MPI_Test(&request, flag, status);
};
