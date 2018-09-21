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

#include "../PointToPoint/Sendrecv.hpp"
#include "../RBC.hpp"

namespace RBC {

    int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Allgather(const_cast<void*> (sendbuf), sendcount, sendtype, recvbuf, recvcount, recvtype,
                    comm.mpi_comm);
        }

        Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                0, comm);
        int size;
        Comm_size(comm, &size);
        Bcast(recvbuf, size * recvcount, recvtype, 0, comm);
        return 0;
    }

    int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs,
            MPI_Datatype recvtype, Comm const &comm) {
        if (comm.useMPICollectives()) {
            return MPI_Allgatherv(const_cast<void*> (sendbuf), sendcount, sendtype, recvbuf, const_cast<int*> (recvcounts),
                    const_cast<int*> (displs), recvtype, comm.mpi_comm);
        }

        Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                0, comm);
        int size;
        Comm_size(comm, &size);
        int total_recvcount = 0;
        for (int i = 0; i < size; i++)
            total_recvcount += recvcounts[i % size];
        Bcast(recvbuf, total_recvcount, recvtype, 0, comm);
        return 0;
    }

    int Allgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount,
            std::function<void (void*, void*, void*) > op, Comm const &comm) {
        Gatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount, 0,
                op, comm);
        Bcast(recvbuf, recvcount, sendtype, 0, comm);
        return 0;
    }

    namespace _internal {

        namespace optimized {

/*
 * AllgatherPipeline: Allgather algorithm with running time
 * O(alpha * log p + beta n log p).  p must be a power of two!!!
 */
            double AllgatherPipelineExpRunningTime(Comm const & comm, int sendcount, MPI_Datatype sendtype,
                    bool* valid) {

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);

                // Valid if number of processes is a power of two.
                *valid = true;
                
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int n = sendcount * datatype_size * size; // In bytes

                return
                    Model_Const::ALPHA * (size - 1) +
                    Model_Const::BETA * n * (size - 1)/size;
            }

        
            int AllgatherPipeline(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm) {
                if (comm.useMPICollectives()) {
                    return MPI_Allgather(const_cast<void*> (sendbuf),
                            sendcount, sendtype, recvbuf, recvcount, recvtype,
                            comm.mpi_comm);
                }

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int recv_size = sendcount * datatype_size;

                if (sendcount == 0) {
                    return 0;
                }

                // Move local input to output buffer.
                memcpy((char*)recvbuf + recv_size * rank, sendbuf, recv_size);

                if (size == 1) {
                    return 0;
                }

                const int target = (rank + 1) % size;
                const int source = (rank - 1 + size) % size;
        
                for (int it = 0; it != size - 1; ++it) {
                    int recv_pe = (size + rank - it - 1) % size;
                    int send_pe = (size + rank - it) % size;
            
                    Sendrecv((char*)recvbuf + recv_size * send_pe,
                            sendcount,
                            sendtype,
                            target,
                            Tag_Const::ALLGATHER,
                            (char*)recvbuf + recv_size * recv_pe,
                            recvcount,
                            recvtype,
                            source,
                            Tag_Const::ALLGATHER,
                            comm,
                            MPI_STATUS_IGNORE);
                }

                return 0;
            }
    
	    /*
	     * AllgatherDissemination: Allgather algorithm with running time
	     * O(alpha * log p + beta n).  Compared to the one for hypercubes,
	     * we have to copy the data once
	     */
            double AllgatherDisseminationExpRunningTime(Comm const & comm, int sendcount, MPI_Datatype sendtype,
                    bool* valid) {

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);

                *valid = true;
                
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int n = sendcount * datatype_size * size; // In bytes

                // We expect that copying the data takes beta/6 time.
                return
                    Model_Const::ALPHA * std::ceil(std::log2(size)) +
                    Model_Const::BETA * n * (size - 1)/size +
                    Model_Const::BETA * n / 6;
            }
            
            int AllgatherDissemination(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm) {
                if (comm.useMPICollectives()) {
                    return MPI_Allgather(const_cast<void*> (sendbuf),
                            sendcount, sendtype, recvbuf, recvcount, recvtype,
                            comm.mpi_comm);
                }

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);

                if (sendcount == 0) {
                    return 0;
                }
                
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int recv_size = recvcount * datatype_size;

                if (size == 1) {
                    // Move local input to output buffer.
                    memcpy((char*)recvbuf + recv_size * rank, sendbuf, recv_size);

                    return 0;
                }

                char* tmpbuf_arr = new char[recv_size * size];

                // Move local input to temp buffer.
                memcpy((char*)tmpbuf_arr, sendbuf, recv_size);

                int cnt = recvcount;
                int offset = 1;

                // First floor(log(p)) rounds with exp increasing msg size.
                while (offset <= size / 2) {
                    int source = (rank + offset) % size;
                    // + size to avoid negative numbers
                    int target = (rank - offset + size) % size;

                    SendrecvNonZeroed(tmpbuf_arr,
                            cnt,
                            sendtype,
                            target,
                            Tag_Const::ALLGATHER,
                            tmpbuf_arr + cnt * datatype_size,
                            cnt,
                            sendtype,
                            source,
                            Tag_Const::ALLGATHER,
                            comm,
                            MPI_STATUS_IGNORE);

                    cnt *= 2;
                    offset *= 2;
                }

                // Last round to exchange remaining elements if size is not a power of two.
                const int remaining = recvcount * size - cnt;
                if (offset < size) {
                    int source = (rank + offset) % size;
                    // + size to avoid negative numbers
                    int target = (rank - offset + size) % size;

                    SendrecvNonZeroed(tmpbuf_arr,
                            remaining,
                            sendtype,
                            target,
                            Tag_Const::ALLGATHER,
                            tmpbuf_arr + cnt * datatype_size,
                            remaining,
                            sendtype,
                            source,
                            Tag_Const::ALLGATHER,
                            comm,
                            MPI_STATUS_IGNORE);
                }

                /* Copy data to recvbuf. All PEs but PE 0 have to move their
                 * data in two steps */
                if (rank == 0) {
                    memcpy(recvbuf, tmpbuf_arr, recv_size * size);
                } else {
                    memcpy((char*)recvbuf + recv_size * rank, tmpbuf_arr, recv_size * (size - rank));
                    memcpy((char*)recvbuf, tmpbuf_arr + recv_size * (size - rank), recv_size * rank);
                }

                delete[] tmpbuf_arr;
                return 0;
            }

/*
 * AllgatherHypercube: Allgather algorithm with running time
 * O(alpha * log p + beta n log p).  p must be a power of two!!!
 */
            double AllgatherHypercubeExpRunningTime(Comm const & comm, int sendcount, MPI_Datatype sendtype,
                    bool* valid) {

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);

                // Valid if number of processes is a power of two.
                *valid = (size & (size - 1)) == 0;
                
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int n = sendcount * datatype_size * size; // In bytes

                return
                    Model_Const::ALPHA * std::ceil(std::log2(size)) +
                    Model_Const::BETA * n * (size - 1)/size;
            }

            
            int AllgatherHypercube(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm) {
                if (comm.useMPICollectives()) {
                    return MPI_Allgather(const_cast<void*> (sendbuf),
                            sendcount, sendtype, recvbuf, recvcount, recvtype,
                            comm.mpi_comm);
                }

                int rank, size;
                Comm_rank(comm, &rank);
                Comm_size(comm, &size);

                if (sendcount == 0) {
                    return 0;
                }

                // size must be a power of two to execute the hypercube version.
                if ((size & (size - 1))) {
                    printf ("%s \n", "Warning: p is not a power of two. Fallback to RBC::Allgather");
                    return RBC::Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
                }
        
                MPI_Aint lb, type_size;
                MPI_Type_get_extent(sendtype, &lb, &type_size);
                int datatype_size = static_cast<int> (type_size);
                int recv_size = recvcount * datatype_size;

                if (size == 1) {
                    // Move local input to output buffer.
                    memcpy((char*)recvbuf + recv_size * rank, sendbuf, recv_size);
                    return 0;
                }

                // Move local input to output buffer.
                memcpy((char*)recvbuf + recv_size * rank, sendbuf, recv_size);

                int cnt = recvcount;
                char* recvbuf_ptr = (char*)recvbuf + recv_size * rank;
                const size_t log_p = std::log2(size);
                for (size_t it = 0; it != log_p; ++it) {
                    int target = (size_t)rank ^ (size_t)1 << it;
                    bool left_target = target < rank;

                    if (left_target) {
                        SendrecvNonZeroed(recvbuf_ptr,
                                cnt,
                                sendtype,
                                target,
                                Tag_Const::ALLGATHER,
                                recvbuf_ptr - cnt * datatype_size,
                                cnt,
                                recvtype,
                                target,
                                Tag_Const::ALLGATHER,
                                comm,
                                MPI_STATUS_IGNORE);
                        recvbuf_ptr -= cnt * datatype_size;
                    } else {
                        SendrecvNonZeroed(recvbuf_ptr,
                                cnt,
                                sendtype,
                                target,
                                Tag_Const::ALLGATHER,
                                recvbuf_ptr + cnt * datatype_size,
                                cnt,
                                recvtype,
                                target,
                                Tag_Const::ALLGATHER,
                                comm,
                                MPI_STATUS_IGNORE);
                    }
            
                    cnt *= 2;
                }

                return 0;
            }

/*
 *
 * Blocking allgather with equal amount of elements on each process
 * This method uses different implementations depending on the
 * size of comm and the input size.
 */
            int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    Comm const &comm) {
                bool valid_pipeline;
                double pipeline = AllgatherPipelineExpRunningTime(comm, sendcount, sendtype,
                        &valid_pipeline);
                bool valid_hypercube;
                double hypercube = AllgatherHypercubeExpRunningTime(comm, sendcount, sendtype,
                        &valid_hypercube);
                bool valid_dissemination;
                double dissemination = AllgatherDisseminationExpRunningTime(comm, sendcount, sendtype,
                        &valid_dissemination);

                if (valid_hypercube) {
                    if (hypercube < pipeline) {
                        return AllgatherHypercube(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype,
                                comm);
                    } else {
                        return AllgatherDissemination(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype,
                                comm);
                    }
                } else {
                    if (dissemination < pipeline) {
                        return AllgatherDissemination(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype,
                                comm);
                    } else {
                        return AllgatherPipeline(sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype,
                                comm);
                    }
                }
            }

        } // end namespace optimized
    
        /*
         * Request for the allgather
         */
        class IallgatherReq : public RequestSuperclass {
        public:
            IallgatherReq(const void *sendbuf, int sendcount,
                    MPI_Datatype sendtype, void *recvbuf, int recvcount,
                    const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                    int tag, std::function<void (void*, void*, void*) > op,
                    Comm const &comm, std::string collective_op);
            ~IallgatherReq();
            int test(int *flag, MPI_Status *status) override;
        private:
            void *recvbuf;
            int tag, total_recvcount, gather_completed, bcast_completed;
            MPI_Datatype recvtype;
            std::function<void (void*, void*, void*) > op;
            Comm comm;
            bool mpi_collective;
            Request req_gather, req_bcast;
            MPI_Request mpi_req;
        };
    }

    int Iallgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            Comm const &comm, Request* request, int tag) {
        std::function<void (void*, void*, void*) > op =
                [](void* a, void* b, void* c) {
                    return;
                };
        request->set(std::make_shared<_internal::IallgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, recvtype,
                tag, op, comm, "gather"));
        return 0;
    };

    int Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, const int *recvcounts, const int *displs,
            MPI_Datatype recvtype, Comm const &comm,
            Request* request, int tag) {
        std::function<void (void*, void*, void*) > op =
                [](void* a, void* b, void* c) {
                    return;
                };
        request->set(std::make_shared<_internal::IallgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, -1, recvcounts, displs, recvtype, tag,
                op, comm, "gatherv"));
        return 0;
    };

    int Iallgatherm(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount,
            std::function<void (void*, void*, void*) > op, Comm const &comm,
            Request* request, int tag) {
        request->set(std::make_shared<_internal::IallgatherReq>(sendbuf, sendcount,
                sendtype, recvbuf, recvcount, nullptr, nullptr, sendtype, tag,
                op, comm, "gatherm"));
        return 0;
    };
}

RBC::_internal::IallgatherReq::IallgatherReq(const void *sendbuf, int sendcount,
        MPI_Datatype sendtype, void *recvbuf, int recvcount,
        const int *recvcounts, const int *displs, MPI_Datatype recvtype,
        int tag, std::function<void (void*, void*, void*) > op,
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
    } else if (collective_op == "gatherm") {
        RBC::Igatherm(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                root, op, comm, &req_gather, tag);
        total_recvcount = recvcount;
    } else {
        assert(false && "bad collective operation");
    }
};

RBC::_internal::IallgatherReq::~IallgatherReq() {
};

int RBC::_internal::IallgatherReq::test(int *flag, MPI_Status *status) {
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
