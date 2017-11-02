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

int RBC::Sendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm, MPI_Status *status) {
    return MPI_Sendrecv(sendbuf,
                        sendcount, sendtype,
                        comm.RangeRankToMpiRank(dest), sendtag,
                        recvbuf, recvcount, recvtype,
                        comm.RangeRankToMpiRank(source), recvtag,
                        comm.mpi_comm, status);
}


/*
 * Request for the isend
 */
class Range_Requests::Isendrecv : public RBC::R_Req {
public:
    Isendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm);
    int test(int *flag, MPI_Status *status);

private:
    MPI_Request *requests;
};

int RBC::Isendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm, RBC::Request *request) {
    *request  = std::unique_ptr<R_Req>(new Range_Requests::Isendrecv(sendbuf, 
            sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, 
            source, recvtag, comm));
    return 0;
};

Range_Requests::Isendrecv::Isendrecv(void *sendbuf,
            int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
            void *recvbuf, int recvcount, MPI_Datatype recvtype,
            int source, int recvtag,
            RBC::Comm const &comm) {
    void* sendbuf_ = const_cast<void*>(sendbuf);    
    requests = new MPI_Request[2];
    MPI_Isend(sendbuf_, sendcount, sendtype, comm.RangeRankToMpiRank(dest), 
        sendtag, comm.mpi_comm, &requests[0]);       
    MPI_Irecv(recvbuf, recvcount, recvtype, comm.RangeRankToMpiRank(source),
        recvtag, comm.mpi_comm, &requests[1]);
};

int Range_Requests::Isendrecv::test(int *flag, MPI_Status *status) {
    return MPI_Testall(2, requests, flag, MPI_STATUSES_IGNORE);
};
