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

int RBC::Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, int root, RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Reduce(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, root, comm.mpi_comm);
    } 
    
    int tag = RBC::Tag_Const::REDUCE;
    int rank, size;
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    int datatype_size = static_cast<int>(type_size);
    int recv_size = count * datatype_size;        

    int new_rank = (rank - root - 1 + size) % size;
    int height = ceil(log2(size));
    int own_height = 0;
    if (new_rank == (size - 1)) {
        own_height = height;
    } else {
        for (int i = 0; ((new_rank >> i) % 2 == 1) && (i < height); i++)
            own_height++;
    }
    char *recvbuf_arr = new char[recv_size * own_height];
    char *reduce_buf = new char[recv_size];
    std::memcpy(reduce_buf, sendbuf, recv_size);
    std::vector<RBC::Request> recv_requests;
    recv_requests.reserve(own_height);

    //Receive data
    int tmp_rank = new_rank;
    if (new_rank == size - 1)
        tmp_rank = pow(2, height) - 1;

    for (int i = own_height - 1; i >= 0; i--) {
        int tmp_src = tmp_rank - pow(2, i);
        if (tmp_src < new_rank) {
            recv_requests.push_back(RBC::Request());
            int src = (tmp_src + root + 1) % size;
            RBC::Irecv(recvbuf_arr + (recv_requests.size() - 1) * recv_size, count,
                    datatype, src,
                    tag, comm, &recv_requests.back());
        } else {
            tmp_rank = tmp_src;
        }
    }        
    RBC::Waitall(recv_requests.size(), &recv_requests.front(), MPI_STATUSES_IGNORE);

    if (recv_requests.size() > 0) {
        //Reduce received data and local data
        for (size_t i = 0; i < (recv_requests.size() - 1); i++) {
            MPI_Reduce_local(recvbuf_arr + i * recv_size,
                    recvbuf_arr + (i + 1) * recv_size, count, datatype, op);
        }
        MPI_Reduce_local(recvbuf_arr + (recv_requests.size() - 1) * recv_size,
                reduce_buf, count, datatype, op);
    }

    //Send data
    if (new_rank < size - 1) {
        int tmp_dest = new_rank + pow(2, own_height);
        if (tmp_dest > size - 1)
            tmp_dest = size - 1;
        int dest = (tmp_dest + root + 1) % size;
        RBC::Send(reduce_buf, count, datatype, dest, tag, comm);
    } else {
        std::memcpy(recvbuf, reduce_buf, recv_size);
    }

    delete[] recvbuf_arr;
    delete[] reduce_buf;
    return 0;
}


/*
 * Request for the reduce
 */
class Range_Requests::Ireduce : public RBC::R_Req {
public:
    Ireduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, int root, RBC::Comm const &comm);
    ~Ireduce();
    int test(int *flag, MPI_Status *status);
private:
    const void *sendbuf;
    void *recvbuf;
    int count, tag, root, rank, size, new_rank, height, own_height, 
        datatype_size, recv_size, receives;
    MPI_Datatype datatype;
    MPI_Op op;
    RBC::Comm comm;
    bool send, completed, mpi_collective;
    char *recvbuf_arr, *reduce_buf;
    RBC::Request send_req;
    std::vector<RBC::Request> recv_requests;
    MPI_Request mpi_req;
};

int RBC::Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, 
        MPI_Op op, int root, RBC::Comm const &comm, RBC::Request* request, int tag) {
    *request = std::unique_ptr<R_Req>(new Range_Requests::Ireduce(sendbuf, recvbuf,
            count, datatype, tag, op, root, comm));
    return 0;
}

Range_Requests::Ireduce::Ireduce(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, int root, RBC::Comm const &comm) : 
        sendbuf(sendbuf), recvbuf(recvbuf), count(count), tag(tag), root(root), 
        datatype(datatype), op(op), comm(comm), send(false), completed(false),
        mpi_collective(false), recvbuf_arr(nullptr), reduce_buf(nullptr) {    
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm.mpi_comm,
                &mpi_req);
        mpi_collective = true;
        return;
    }
#endif
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_size = static_cast<int>(type_size);
    recv_size = count * datatype_size;
    new_rank = (rank - root - 1 + size) % size;
    height = ceil(log2(size));
    own_height = 0;
    if (new_rank == (size - 1)) {
        own_height = height;
    } else {
        for (int i = 0; ((new_rank >> i) % 2 == 1) && (i < height); i++)
            own_height++;
    }

    recvbuf_arr = new char[recv_size * own_height];
    reduce_buf = new char[recv_size];
    std::memcpy(reduce_buf, sendbuf, recv_size);
}

Range_Requests::Ireduce::~Ireduce() {
    if (recvbuf_arr != nullptr)
        delete[] recvbuf_arr;
    if (reduce_buf != nullptr)
        delete[] reduce_buf;
}

int Range_Requests::Ireduce::test(int* flag, MPI_Status* status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    
    if (mpi_collective) 
        return MPI_Test(&mpi_req, flag, status);

    if (height > 0 && recv_requests.size() == 0) {
        //Receive data
        int tmp_rank = new_rank;
        if (new_rank == size - 1)
            tmp_rank = pow(2, height) - 1;

        for (int i = own_height - 1; i >= 0; i--) {
            int tmp_src = tmp_rank - pow(2, i);
            if (tmp_src < new_rank) {
                recv_requests.push_back(RBC::Request());
                int src = (tmp_src + root + 1) % size;
                RBC::Irecv(recvbuf_arr + (recv_requests.size() - 1) * recv_size, count,
                        datatype, src,
                        tag, comm, &recv_requests.back());
            } else {
                tmp_rank = tmp_src;
            }
        }
        receives = recv_requests.size();
    }

    if (!send) {
        int recv_finished;
        RBC::Testall(recv_requests.size(), &recv_requests.front(), &recv_finished,
                MPI_STATUSES_IGNORE);
        if (recv_finished && receives > 0) {
            //Reduce received data and local data
            for (int i = 0; i < (receives - 1); i++) {
                MPI_Reduce_local(recvbuf_arr + i * recv_size,
                        recvbuf_arr + (i + 1) * recv_size, count, datatype, op);
            }
            MPI_Reduce_local(recvbuf_arr + (receives - 1) * recv_size,
                    reduce_buf, count, datatype, op);
        }

        //Send data
        if (recv_finished) {
            if (new_rank < size - 1) {
                int tmp_dest = new_rank + pow(2, own_height);
                if (tmp_dest > size - 1)
                    tmp_dest = size - 1;
                int dest = (tmp_dest + root + 1) % size;
                RBC::Isend(reduce_buf, count, datatype, dest, tag, comm, &send_req);
            }
            send = true;
        }
    }
    if (send) {
        if (new_rank == size - 1) {
            std::memcpy(recvbuf, reduce_buf, recv_size);
            *flag = 1;
        } else
            RBC::Test(&send_req, flag, MPI_STATUS_IGNORE);
        if (*flag)
            completed = true;
    }
    return 0;
}

