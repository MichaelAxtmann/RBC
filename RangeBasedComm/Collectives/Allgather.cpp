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
#include "../RBC.hpp"
#include "../Requests.hpp"
#include <iostream>
#include <cassert>
#include <cstring>

int RBC::Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Allgather(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                comm.mpi_comm);
    }
    
    RBC::Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
            0, comm);
    int size;
    RBC::Comm_size(comm, &size);
    RBC::Bcast(recvbuf, size * recvcount, recvtype, 0, comm);
    return 0;
}

int RBC::Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs, 
        MPI_Datatype recvtype, RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Allgatherv(const_cast<void*>(sendbuf), sendcount, sendtype, recvbuf, const_cast<int*>(recvcounts), 
                const_cast<int*>(displs), recvtype, comm.mpi_comm);
    }

    RBC::Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
            0, comm);
    int size;
    RBC::Comm_size(comm, &size);
    int total_recvcount = 0;
    for (int i = 0; i < size; i++)
        total_recvcount += recvcounts[i % size];
    RBC::Bcast(recvbuf, total_recvcount, recvtype, 0, comm);
    return 0;
}

int RBC::Allgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount,
        std::function<void (void*, void*, void*)> op, RBC::Comm const &comm) {
    RBC::Gatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, 0, 
            op, comm);
    RBC::Bcast(recvbuf, recvcount, sendtype, 0, comm);
    return 0;
}

/*
 * Request for the allgather
 */
class Range_Requests::Iallgather : public RBC::R_Req {
public:
    Iallgather(const void *sendbuf, int sendcount, 
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        const int *recvcounts, const int *displs, MPI_Datatype recvtype,
        int tag, std::function<void (void*, void*, void*)> op,
        RBC::Comm const &comm, std::string collective_op);
    ~Iallgather();
    int test(int *flag, MPI_Status *status) override;
private:
    void *recvbuf;
    int tag, total_recvcount, gather_completed, bcast_completed;
    MPI_Datatype recvtype;
    std::function<void (void*, void*, void*)> op;
    RBC::Comm comm;
    bool mpi_collective;
    RBC::Request req_gather, req_bcast;
    MPI_Request mpi_req;
};

int RBC::Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype,
        RBC::Comm const &comm, RBC::Request* request, int tag) {
    std::function<void (void*, void*, void*)> op = 
        [](void* a, void* b, void* c) { return; };
    *request = std::unique_ptr<R_Req>(new Range_Requests::Iallgather(sendbuf, sendcount,
            sendtype, recvbuf, recvcount, nullptr, nullptr, recvtype,
            tag, op, comm, "gather"));
    return 0;
};

int RBC::Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int *recvcounts, const int *displs, 
        MPI_Datatype recvtype, RBC::Comm const &comm,
        RBC::Request* request, int tag) {
    std::function<void (void*, void*, void*)> op = 
        [](void* a, void* b, void* c) { return; };
    *request = std::unique_ptr<R_Req>(new Range_Requests::Iallgather(sendbuf, sendcount,
            sendtype, recvbuf, -1, recvcounts, displs, recvtype, tag, 
            op, comm, "gatherv"));
    return 0;
};

int RBC::Iallgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, 
        std::function<void (void*, void*, void*)> op, RBC::Comm const &comm,
        RBC::Request* request, int tag) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Iallgather(sendbuf, sendcount,
            sendtype, recvbuf, recvcount, nullptr, nullptr, sendtype, tag, 
            op, comm, "gatherm"));
    return 0;
};

Range_Requests::Iallgather::Iallgather(const void *sendbuf, int sendcount, 
        MPI_Datatype sendtype, void *recvbuf, int recvcount, 
        const int *recvcounts, const int *displs, MPI_Datatype recvtype,
        int tag, std::function<void (void*, void*, void*)> op,
        RBC::Comm const &comm, std::string collective_op) 
        : tag(tag), gather_completed(0), bcast_completed(0), recvtype(recvtype),
        mpi_collective(false) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        if (collective_op == "gather") {
            MPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                    comm.mpi_comm, &mpi_req);
            mpi_collective = true;
            return;
        } else if (collective_op == "gatherv") {
            MPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf,
                    recvcounts, displs, recvtype, comm.mpi_comm, &mpi_req);
            mpi_collective = true;
            return;
        } else {
            assert(collective_op == "gatherm");
        }
    }
#endif
    int size;
    RBC::Comm_size(comm, &size);
    
    int root = 0;
    if (collective_op == "gather") {
        RBC::Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                root, comm, &req_gather, tag); 
        total_recvcount = size * recvcount; 
    } else if (collective_op == "gatherv") {
        RBC::Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                recvtype, root, comm, &req_gather, tag);
        total_recvcount = 0;
        for (int i = 0; i < size; i++)
            total_recvcount += recvcounts[i % size];
    } else if (collective_op == "gatherm" ) {
        RBC::Igatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                root, op, comm, &req_gather, tag);         
        total_recvcount = recvcount;
    } else {
        assert(false && "bad collective operation");
    }
};

Range_Requests::Iallgather::~Iallgather() {
};

int Range_Requests::Iallgather::test(int *flag, MPI_Status *status) {    
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);
    
    if (bcast_completed) {
        *flag = 1;
        return 0;
    }
    
    if (!gather_completed) {
        RBC::Test(&req_gather, &gather_completed, MPI_STATUS_IGNORE);
        if (gather_completed) {
            RBC::Ibcast(recvbuf, total_recvcount, recvtype, 0, comm,
                    &req_bcast, tag);
        }
    } else {
        RBC::Test(&req_bcast, &bcast_completed, MPI_STATUS_IGNORE);
    }
    
    if (bcast_completed)
        *flag = 1;
    
    return 0;
    
};
