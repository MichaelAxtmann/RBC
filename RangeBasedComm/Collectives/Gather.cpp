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
#include <iostream>
#include <cassert>
#include <cstring>

#include "../RBC.hpp"

namespace RBC {

    int blockingGather(const void *sendbuf, int sendcount,
            MPI_Datatype sendtype, void *recvbuf, int recvcount,
            const int *recvcounts, const int *displs, MPI_Datatype recvtype,
            int root, std::function<void (void*, void*, void*) > op,
            Comm const &comm, std::string collective_op) {
        int tag = Tag_Const::GATHER;
        int rank, size;
        Comm_rank(comm, &rank);
        Comm_size(comm, &size);

        int new_rank = (rank - root + size) % size;
        int max_height = std::ceil(std::log2(size));
        int own_height = 0;
        if (std::pow(2, max_height) < size)
            max_height++;
        for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
            own_height++;

        MPI_Aint lb, recv_size, send_size;
        MPI_Type_get_extent(recvtype, &lb, &recv_size);
        MPI_Type_get_extent(sendtype, &lb, &send_size);
        int recvtype_size = static_cast<int> (recv_size);
        int sendtype_size = static_cast<int> (send_size);

        int subtree_size = static_cast<int> (std::min(std::pow(2, own_height),
                static_cast<double> (size)));
        int total_recvcount = 0;
        if (collective_op == "gather") {
            assert(sendcount == recvcount); // TODO: different send and recv types
            assert(sendtype_size == recvtype_size);
            total_recvcount = subtree_size * recvcount;
        } else if (collective_op == "gatherv") {
            total_recvcount = 0;
            for (int i = 0; i < size; i++) //TODO: test
                total_recvcount += recvcounts[i % size];
        } else if (collective_op == "gatherm") {
            total_recvcount = recvcount;
        } else {
            assert(false && "bad collective operation");
        }
        recv_size = total_recvcount * recvtype_size;
        char* recv_buf = nullptr;
        if (rank == root && recvcounts == nullptr) {
            assert(recvbuf != nullptr);
            recv_buf = static_cast<char*> (recvbuf);
        } else {
            recv_buf = new char[recv_size];
        }

        //Copy send data into receive buffer
        std::memcpy(recv_buf, sendbuf, sendcount * sendtype_size);
        int received = sendcount;
        int height = 1;

        //If messages have to be received
        while (height <= own_height) {
            int tmp_src = new_rank + std::pow(2, height - 1);
            if (tmp_src < size) {
                int src = (tmp_src + root) % size;
                //Range::Test if message can be received
                MPI_Status status;
                Probe(src, tag, comm, &status);
                //Receive message
                int count;
                MPI_Get_count(&status, sendtype, &count);
                Recv(recv_buf + received * sendtype_size, count,
                        sendtype, src, tag, comm, MPI_STATUS_IGNORE);
                //Merge the received data
                op(recv_buf, recv_buf + received * sendtype_size,
                        recv_buf + (received + count) * sendtype_size);
                received += count;
                height++;
            } else {
                //Source rank larger than comm size
                height++;
            }
        }

        //When all messages have been received
        if (rank == root) {
            //root doesn't send to anyone
            assert(total_recvcount == received);
            if (recvcounts != nullptr) {
                char *buf = static_cast<char*> (recvbuf);
                char *recv_ptr = recv_buf;
                for (int i = 0; i < size; i++) {
                    std::memcpy(buf + displs[i] * sendtype_size, recv_ptr,
                            recvcounts[i] * sendtype_size);
                    recv_ptr += recvcounts[i] * sendtype_size;
                }
            }
        } else {
            // Send to parent node
            int tmp_dest = new_rank - std::pow(2, height - 1);
            int dest = (tmp_dest + root) % size;
            Send(recv_buf, received, sendtype, dest, tag, comm);
        }

        if (rank != root || recvcounts != nullptr) {
            delete[] recv_buf;
        }
        return 0;
    }

    int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, Comm const &comm) {
#ifndef NO_IBAST
        if (comm.useMPICollectives()) {
            return MPI_Gather(const_cast<void*> (sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                    root, comm.mpi_comm);
        }
#endif
        std::function<void (void*, void*, void*) > op =
                [](void*, void*, void*) {
                    return;
                };
        blockingGather(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, recvtype, root,
                op, comm, "gather");
        return 0;
    }

    int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs,
            MPI_Datatype recvtype, int root, Comm const &comm) {
#ifndef NO_IBAST
        if (comm.useMPICollectives()) {
            return MPI_Gatherv(const_cast<void*> (sendbuf), sendcount, sendtype, recvbuf, const_cast<int*> (recvcounts),
                    const_cast<int*> (displs), recvtype, root, comm.mpi_comm);
        }
#endif
        std::function<void (void*, void*, void*) > op =
                [](void*, void*, void*) {
                    return;
                };
        blockingGather(sendbuf, sendcount,
                sendtype, recvbuf, -1, recvcounts, displs, recvtype, root,
                op, comm, "gatherv");
        return 0;
    }

    int Gatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, int root,
            std::function<void (void*, void*, void*) > op, Comm const &comm) {
        blockingGather(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, sendtype, root,
                op, comm, "gatherm");
        return 0;
    }

    namespace _internal {

        /*
         * Request for the gather
         */
        class IgatherReq : public RequestSuperclass {
        public:
            IgatherReq(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf, int recvcount,
                    const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                    int root, int tag, std::function<void (void*, void*, void*) > op,
                    Comm const &comm, std::string collective_op);
            ~IgatherReq();
            int test(int *flag, MPI_Status *status);
        private:
            const void *sendbuf;
            void *recvbuf;
            const int *recvcounts, *displs;
            int sendcount, recvcount, root, tag, own_height, size, rank, height,
            received, count, total_recvcount, new_rank, sendtype_size,
            recvtype_size, recv_size;
            MPI_Datatype sendtype, recvtype;
            std::function<void (void*, void*, void*) > op;
            Comm comm;
            bool receive, send, completed, mpi_collective;
            char *recv_buf;
            Request recv_req, send_req;
            MPI_Request mpi_req;
        };
    }

    int Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int root, Comm const &comm, Request* request, int tag) {
        std::function<void (void*, void*, void*) > op =
                [](void*, void*, void*) {
                    return;
                };
        request->set(std::make_shared<_internal::IgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, recvtype, root,
                tag, op, comm, "gather"));
        return 0;
    };

    int Igatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs,
            MPI_Datatype recvtype, int root, Comm const &comm,
            Request* request, int tag) {
        std::function<void (void*, void*, void*) > op =
                [](void*, void*, void*) {
                    return;
                };
        request->set(std::make_shared<_internal::IgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, -1, recvcounts, displs, recvtype, root, tag,
                op, comm, "gatherv"));
        return 0;
    };

    int Igatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, int root,
            std::function<void (void*, void*, void*) > op, Comm const &comm,
            Request* request, int tag) {
        request->set(std::make_shared<RBC::_internal::IgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, sendtype, root, tag,
                op, comm, "gatherm"));
        return 0;
    };
}

RBC::_internal::IgatherReq::IgatherReq(const void *sendbuf, int sendcount,
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        const int *recvcounts, const int *displs, MPI_Datatype recvtype,
        int root, int tag, std::function<void (void*, void*, void*) > op,
        RBC::Comm const &comm, std::string collective_op)
: sendbuf(sendbuf), recvbuf(recvbuf), recvcounts(recvcounts), displs(displs),
sendcount(sendcount), recvcount(recvcount), root(root), tag(tag),
own_height(0), size(0), rank(0), height(1), received(0), count(0),
sendtype(sendtype), recvtype(recvtype), op(op), comm(comm),
receive(false), send(false), completed(false), mpi_collective(false),
recv_buf(nullptr) {
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        if (collective_op == "gather") {
            MPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                    root, comm.mpi_comm, &mpi_req);
            mpi_collective = true;
            return;
        } else if (collective_op == "gatherv") {
            MPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf,
                    recvcounts, displs, recvtype, root, comm.mpi_comm, &mpi_req);
            mpi_collective = true;
            return;
        } else
            assert(collective_op == "gatherm");
    }
#endif

    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);

    new_rank = (rank - root + size) % size;
    int max_height = std::ceil(std::log2(size));
    if (std::pow(2, max_height) < size)
        max_height++;
    for (int i = 0; ((new_rank >> i) % 2 == 0) && (i < max_height); i++)
        own_height++;

    MPI_Aint lb, recv_size, send_size;
    MPI_Type_get_extent(recvtype, &lb, &recv_size);
    MPI_Type_get_extent(sendtype, &lb, &send_size);
    recvtype_size = static_cast<int> (recv_size);
    sendtype_size = static_cast<int> (send_size);

    int subtree_size = static_cast<int> (std::min(std::pow(2, own_height),
            static_cast<double> (size)));
    if (collective_op == "gather") {
        assert(sendcount == recvcount); // TODO: different send and recv types
        assert(sendtype_size == recvtype_size);
        total_recvcount = subtree_size * recvcount;
    } else if (collective_op == "gatherv") {
        total_recvcount = 0;
        for (int i = 0; i < size; i++) //TODO: test
            total_recvcount += recvcounts[i % size];
    } else if (collective_op == "gatherm") {
        total_recvcount = recvcount;
    } else {
        assert(false && "bad collective operation");
    }
    recv_size = total_recvcount * recvtype_size;
    if (rank == root && recvcounts == nullptr) {
        assert(recvbuf != nullptr);
        recv_buf = static_cast<char*> (recvbuf);
    } else
        recv_buf = new char[recv_size];
    //Copy send data into receive buffer
    std::memcpy(recv_buf, sendbuf, sendcount * sendtype_size);
    received = sendcount;
};

RBC::_internal::IgatherReq::~IgatherReq() {
    if ((rank != root || recvcounts != nullptr) && recv_buf != nullptr)
        delete[] recv_buf;
}

int RBC::_internal::IgatherReq::test(int *flag, MPI_Status *status) {
    if (completed) {
        *flag = 1;
        return 0;
    }

    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    //If messages have to be received
    if (height <= own_height) {
        if (!receive) {
            int tmp_src = new_rank + std::pow(2, height - 1);
            if (tmp_src < size) {
                int src = (tmp_src + root) % size;
                //Range::Test if message can be received
                MPI_Status status;
                int ready;
                RBC::Iprobe(src, tag, comm, &ready, &status);
                if (ready) {
                    //Receive message with non-blocking receive
                    MPI_Get_count(&status, sendtype, &count);
                    RBC::Irecv(recv_buf + received * sendtype_size, count,
                            sendtype, src, tag, comm, &recv_req);
                    receive = true;
                }
            } else {
                //Source rank larger than comm size
                height++;
            }
        }
        if (receive) {
            //Range::Test if receive finished
            int finished;
            RBC::Test(&recv_req, &finished, MPI_STATUS_IGNORE);
            if (finished) {
                //Merge the received data
                op(recv_buf, recv_buf + received * sendtype_size,
                        recv_buf + (received + count) * sendtype_size);
                received += count;
                height++;
                receive = false;
            }
        }
    }

    //If all messages have been received
    if (height > own_height) {
        if (rank == root) {
            //root doesn't send to anyone
            completed = true;
            //            if (total_recvcount != received)
            //            std::cout << W(rank) << W(size) << W(total_recvcount) << W(received) << std::endl;
            assert(total_recvcount == received);
            if (recvcounts != nullptr) {
                char *buf = static_cast<char*> (recvbuf);
                char *recv_ptr = recv_buf;
                for (int i = 0; i < size; i++) {
                    std::memcpy(buf + displs[i] * sendtype_size, recv_ptr,
                            recvcounts[i] * sendtype_size);
                    recv_ptr += recvcounts[i] * sendtype_size;
                }
            }
        } else {
            if (!send) {
                //Start non-blocking send to parent node
                int tmp_dest = new_rank - std::pow(2, height - 1);
                int dest = (tmp_dest + root) % size;
                RBC::Isend(recv_buf, received, sendtype, dest, tag, comm, &send_req);
                //                std::cout << W(rank) << W(dest) << W(received) << W(total_recvcount) << std::endl;
                send = true;
            }
            //Gather is completed when the send is finished
            int finished;
            RBC::Test(&send_req, &finished, MPI_STATUS_IGNORE);
            if (finished)
                completed = true;
        }
    }

    if (completed)
        *flag = 1;
    return 0;

}
