/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include "../RBC.hpp"

#include <mpi.h>
#include <cmath>
#include <cstring>
#include <memory>

namespace RBC {

    int Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Reduce(const_cast<void*> (sendbuf), recvbuf, count, datatype, op, root, comm.mpi_comm);
        }

        int tag = Tag_Const::REDUCE;
        int rank, size;
        Comm_rank(comm, &rank);
        Comm_size(comm, &size);
        MPI_Aint lb, type_size;
        MPI_Type_get_extent(datatype, &lb, &type_size);
        int datatype_size = static_cast<int> (type_size);
        int recv_size = count * datatype_size;

        int root_rank = (rank - root + size) % size;

        if (size == 1) {
            std::memcpy(recvbuf, sendbuf, recv_size);
            return 0;
        }

        std::unique_ptr<char[]> recvbuf_arr = std::make_unique<char[]>(2 * recv_size);
        std::unique_ptr<char[]> tmp_arr = std::make_unique<char[]>(recv_size);

        std::memcpy(tmp_arr.get(), sendbuf, recv_size);

        // Perform first receive operation if appropriate.
        int src = root_rank ^ 1;
        bool second_buffer = false;
        bool is_recved = false;
        // Level of the tree. We start with level 1 as we have already executed level 0 if
        // appropriate.
        int i = 0;
        if (src > root_rank) {
            if (src < size) {
                int root_src = (src + root) % size;
                RBC::Recv(recvbuf_arr.get(), count, datatype, root_src, tag,
                        comm, MPI_STATUS_IGNORE);
                is_recved = true;
                second_buffer = true;
            }
            i++;
        }

        MPI_Request request;
        while ((root_rank ^ (1 << i)) > root_rank) {
            src = root_rank ^ (1 << i);
            i++;
            if (src < size) {
                int root_src = (src + root) % size;
                // Receive and reduce at the same time -> overlapping.
                RBC::Irecv(recvbuf_arr.get() + recv_size * (size_t)second_buffer, count, datatype, root_src, tag, comm, &request);
                MPI_Reduce_local(recvbuf_arr.get() + recv_size * (size_t)(!second_buffer),
                        tmp_arr.get(), count, datatype, op);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                second_buffer = !second_buffer;
            }
        }

        if (is_recved) {
            MPI_Reduce_local(recvbuf_arr.get() + recv_size * (size_t)(!second_buffer),
                    tmp_arr.get(), count, datatype, op);
        }

        //Send data
        if (root_rank > 0) {
            int dest = root_rank ^ (1 << i);
            int root_dest = (dest + root) % size;
            Send(tmp_arr.get(), count, datatype, root_dest, tag, comm);
        } else {
            std::memcpy(recvbuf, tmp_arr.get(), recv_size);
        }

        return 0;
    }

    namespace _internal {

        /*
         * Request for the reduce
         */
        class IreduceReq : public RequestSuperclass {
        public:
            IreduceReq(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                    int tag, MPI_Op op, int root, Comm const &comm);
            ~IreduceReq();
            int test(int *flag, MPI_Status *status);
        private:
            const void *sendbuf;
            void *recvbuf;
            int count, tag, root, rank, size, new_rank, height, own_height,
            datatype_size, recv_size, receives;
            MPI_Datatype datatype;
            MPI_Op op;
            Comm comm;
            bool send, completed, mpi_collective;
            std::unique_ptr<char[]> recvbuf_arr, reduce_buf;// todo reduce
            Request send_req;
            std::vector<Request> recv_requests;
            MPI_Request mpi_req;
        };
    }

    int Ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
            MPI_Op op, int root, Comm const &comm, Request* request, int tag) {
        request->set(std::make_shared<_internal::IreduceReq>(sendbuf, recvbuf,
                count, datatype, tag, op, root, comm));
        return 0;
    }
}

RBC::_internal::IreduceReq::IreduceReq(const void* sendbuf, void* recvbuf, int count,
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
    datatype_size = static_cast<int> (type_size);
    recv_size = count * datatype_size;
    new_rank = (rank - root - 1 + size) % size;
    height = std::ceil(std::log2(size));
    own_height = 0;
    if (new_rank == (size - 1)) {
        own_height = height;
    } else {
        for (int i = 0; ((new_rank >> i) % 2 == 1) && (i < height); i++)
            own_height++;
    }

    recvbuf_arr = std::make_unique<char[]>(recv_size * own_height);
    reduce_buf = std::make_unique<char[]>(recv_size);
    std::memcpy(reduce_buf.get(), sendbuf, recv_size);
}

RBC::_internal::IreduceReq::~IreduceReq() {
}

int RBC::_internal::IreduceReq::test(int* flag, MPI_Status* status) {
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
            tmp_rank = std::pow(2, height) - 1;

        for (int i = own_height - 1; i >= 0; i--) {
            int tmp_src = tmp_rank - std::pow(2, i);
            if (tmp_src < new_rank) {
                recv_requests.push_back(RBC::Request());
                int src = (tmp_src + root + 1) % size;
                RBC::Irecv(recvbuf_arr.get() + (recv_requests.size() - 1) * recv_size, count,
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
                MPI_Reduce_local(recvbuf_arr.get() + i * recv_size,
                        recvbuf_arr.get() + (i + 1) * recv_size, count, datatype, op);
            }
            MPI_Reduce_local(recvbuf_arr.get() + (receives - 1) * recv_size,
                    reduce_buf.get(), count, datatype, op);
        }

        //Send data
        if (recv_finished) {
            if (new_rank < size - 1) {
                int tmp_dest = new_rank + std::pow(2, own_height);
                if (tmp_dest > size - 1)
                    tmp_dest = size - 1;
                int dest = (tmp_dest + root + 1) % size;
                RBC::Isend(reduce_buf.get(), count, datatype, dest, tag, comm, &send_req);
            }
            send = true;
        }
    }
    if (send) {
        if (new_rank == size - 1) {
            std::memcpy(recvbuf, reduce_buf.get(), recv_size);
            *flag = 1;
        } else
            RBC::Test(&send_req, flag, MPI_STATUS_IGNORE);
        if (*flag)
            completed = true;
    }
    return 0;
}

