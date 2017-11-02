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
#include <cmath>
#include <cstring>

int RBC::Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.mpi_comm);
    } 
    
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    int datatype_size = static_cast<int>(type_size);
    char* scan_buf = new char[count * datatype_size];

    RBC::ScanAndBcast(sendbuf, scan_buf, recvbuf, count, datatype, op,
        comm);

    delete[] scan_buf;
    return 0;
}


/*
 * Request for the reduce
 */
class Range_Requests::Iallreduce : public RBC::R_Req {
public:
    Iallreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, RBC::Comm const &comm);
    ~Iallreduce();
    int test(int *flag, MPI_Status *status);
private:
    const void *sendbuf;
    void *recvbuf;
    int count, tag, rank, size, new_rank, height, own_height, 
        datatype_size, recv_size, receives;
    MPI_Datatype datatype;
    MPI_Op op;
    RBC::Comm comm;
    bool send, completed, mpi_collective;
    char *recvbuf_arr, *reduce_buf, *scan_buf;
    RBC::Request request;
    std::vector<RBC::Request> recv_requests;
    MPI_Request mpi_req;
};

int RBC::Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, 
        MPI_Op op, RBC::Comm const &comm, RBC::Request* request, int tag) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Iallreduce(sendbuf, recvbuf,
            count, datatype, tag, op, comm));
    return 0;
}

Range_Requests::Iallreduce::Iallreduce(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const &comm) : 
        sendbuf(sendbuf), recvbuf(recvbuf), count(count), tag(tag),
        datatype(datatype), op(op), comm(comm), send(false), completed(false),
        mpi_collective(false), recvbuf_arr(nullptr), reduce_buf(nullptr),
        scan_buf(nullptr) {    
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Iallreduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.mpi_comm,
                &mpi_req);
        mpi_collective = true;
        return;
    }
#endif
    
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_size = static_cast<int>(type_size);
    scan_buf = new char[count * datatype_size];
    
    RBC::IscanAndBcast(sendbuf, scan_buf, recvbuf, count, datatype, op,
        comm, &request, tag);
}

Range_Requests::Iallreduce::~Iallreduce() {
    if (scan_buf != nullptr)
        delete[] scan_buf;
}

int Range_Requests::Iallreduce::test(int* flag, MPI_Status* status) {
    if (mpi_collective) 
        return MPI_Test(&mpi_req, flag, status);
    
    RBC::Test(&request, flag, status);
    
    return 0;
}

