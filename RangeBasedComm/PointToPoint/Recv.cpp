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

int RBC::Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag,
        RBC::Comm const &comm, MPI_Status *status) {
    return MPI_Recv(buf, count, datatype, comm.RangeRankToMpiRank(source), tag, comm.mpi_comm, status);
}

/*
 * Request for the receive
 */
class Range_Requests::Irecv : public RBC::R_Req {
public:
    Irecv(void *buffer, int count, MPI_Datatype datatype, int source,
            int tag, RBC::Comm const &comm);
    int test(int *flag, MPI_Status *status);
private:
    void *buffer;
    int count, source, tag;
    MPI_Datatype datatype;
    RBC::Comm comm;
    bool receiving;
    MPI_Request request;
};

int RBC::Irecv(void* buffer, int count, MPI_Datatype datatype, int source, int tag,
        RBC::Comm const &comm, RBC::Request *request) {    
    *request = std::unique_ptr<R_Req>(new Range_Requests::Irecv(buffer, count, 
            datatype, source, tag, comm));
    return 0;
};

Range_Requests::Irecv::Irecv(void *buffer, int count, MPI_Datatype datatype,
        int source, int tag, RBC::Comm const &comm) : buffer(buffer), count(count),
        source(source), tag(tag), datatype(datatype), comm(comm), receiving(false) {      
    if (source != MPI_ANY_SOURCE) {
        MPI_Irecv(buffer, count, datatype, comm.RangeRankToMpiRank(source), tag,
                comm.mpi_comm, &request);
        receiving = true;
    } else {
        int x;
        this->test(&x, MPI_STATUS_IGNORE);
    }
};

int Range_Requests::Irecv::test(int *flag, MPI_Status *status) {    
    if (receiving) {
        return MPI_Test(&request, flag, status);
    } else {
        if (source != MPI_ANY_SOURCE) {
            MPI_Irecv(buffer, count, datatype, comm.RangeRankToMpiRank(source),
                    tag, comm.mpi_comm,
                      &request);
            receiving = true;
        } else {
            int ready;
            MPI_Status stat;
            RBC::Iprobe(MPI_ANY_SOURCE, tag, comm, &ready, &stat);
            if (ready) {
                MPI_Irecv(buffer, count, datatype, stat.MPI_SOURCE, tag, comm.mpi_comm,
                          &request);
                receiving = true;
            }
        }
    }
    return 0;
};
