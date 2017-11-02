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

int RBC::Bcast(void* buffer, int count, MPI_Datatype datatype, int root, 
        RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Bcast(buffer, count, datatype, root, comm.mpi_comm);
    }
    int tag = RBC::Tag_Const::BCAST;
    MPI_Status status;
    int rank = 0;
    int size = 0;
    int own_height = 0;
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);    
    int temp_rank = (rank - root + size) % size;
    int height = ceil(log2(size));
    for (int i = 0; ((temp_rank >> i) % 2 == 0) && (i < height); i++)
        own_height++;
    if (rank != root){
        int temp_rank = rank - root;
        if (temp_rank < 0)
            temp_rank += size;
        int mask = 0x1;
        while ((temp_rank ^ mask) > temp_rank) {
            mask = mask << 1;
        }
        int temp_src = temp_rank ^ mask;
        int src = (temp_src + root) % size;
        RBC::Recv(buffer, count, datatype, src, tag, comm, &status);
    }

    while (height > 0) {
        if (own_height >= height) {
            int temp_rank = rank - root;
            if (temp_rank < 0)
                temp_rank += size;
            int temp_dest = temp_rank + pow(2, height - 1);
            if (temp_dest < size) {
                int dest = (temp_dest + root) % size;
                RBC::Ssend(buffer, count, datatype, dest, tag, comm);
            }
        }
        height--;
    }
    return 0;
}
    
/*
 * Request for the broadcast
 */
class Range_Requests::Ibcast : public RBC::R_Req {
public:
    Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
            int tag, RBC::Comm const &omm);
    int test(int *flag, MPI_Status *status);

private:
    void *buffer;
    MPI_Datatype datatype;
    int count, root, tag, own_height, size, rank, height, received, sends;
    RBC::Comm comm;
    bool send, completed, mpi_collective;
    RBC::Request recv_req;
    std::vector<RBC::Request> req_vector;
    MPI_Request mpi_req;
};

int RBC::Ibcast(void *buffer, int count, MPI_Datatype datatype,
        int root, RBC::Comm const &comm, RBC::Request *request, int tag) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ibcast(buffer, count, 
            datatype, root, tag, comm));
    return 0;
};

Range_Requests::Ibcast::Ibcast(void *buffer, int count, MPI_Datatype datatype,
        int root, int tag, RBC::Comm const &comm) : buffer(buffer), datatype(datatype),
        count(count), root(root), tag(tag), own_height(0), size(0), rank(0), 
        height(0), received(0), comm(comm), send(false), completed(false),
        mpi_collective(false) {
    
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Ibcast(buffer, count, datatype, root, comm.mpi_comm, &mpi_req);
        mpi_collective = true;
        return;
    }
#endif    
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);
    sends = 0;
    int temp_rank = (rank - root + size) % size;
    height = ceil(log2(size));
    for (int i = 0; ((temp_rank >> i) % 2 == 0) && (i < height); i++)
        own_height++;
    if (rank == root)
        received = 1;
    else
         RBC::Irecv(buffer, count, datatype, MPI_ANY_SOURCE, tag, comm, &recv_req);
};

int Range_Requests::Ibcast::test(int *flag, MPI_Status *status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);
    
    if (!received) {
        RBC::Test(&recv_req, &received, MPI_STATUS_IGNORE);
    }
    if (received && !send) {
        while (height > 0) {
            if (own_height >= height) {
                int temp_rank = rank - root;
                if (temp_rank < 0)
                    temp_rank += size;
                int temp_dest = temp_rank + pow(2, height - 1);
                if (temp_dest < size) {
                    int dest = (temp_dest + root) % size;
                    req_vector.push_back(RBC::Request());
                    RBC::Isend(buffer, count, datatype, dest, tag, comm, &req_vector.back());
                }
            }
            height--;
        }
        send = true;
    }
    if (send) {
        RBC::Testall(req_vector.size(), &req_vector.front(), flag, MPI_STATUSES_IGNORE);
        if (*flag == 1)
            completed = true;
    }
    return 0;
};

