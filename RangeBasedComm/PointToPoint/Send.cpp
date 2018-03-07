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

namespace RBC {

    int Send(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
            int tag, Comm const &comm) {
        return MPI_Send(const_cast<void*> (sendbuf), count, datatype,
                comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm);
    };

    int Ssend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
            int tag, Comm const &comm) {
        return MPI_Ssend(const_cast<void*> (sendbuf), count, datatype,
                comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm);
    };

    namespace _internal {

        /*
         * Request for the isend
         */
        class IsendReq : public RequestSuperclass {
        public:
            IsendReq(const void *sendbuf, int count, MPI_Datatype datatype,
                    int dest, int tag, Comm const &comm);
            int test(int *flag, MPI_Status *status);

        private:
            const void *sendbuf;
            int count, dest, tag;
            MPI_Datatype datatype;
            Comm comm;
            bool requested;
            MPI_Request request;
        };

        /*
         * Request for the issend
         */
        class IssendReq : public RequestSuperclass {
        public:
            IssendReq(const void *sendbuf, int count, MPI_Datatype datatype,
                    int dest, int tag, Comm const &comm);
            int test(int *flag, MPI_Status *status);

        private:
            const void *sendbuf;
            int count;
            MPI_Datatype datatype;
            int dest, tag;
            Comm comm;
            bool requested;
            MPI_Request request;
        };
    }

    int Isend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
            int tag, Comm const &comm, Request *request) {
        request->set(std::make_shared<_internal::IsendReq>(sendbuf, count,
                datatype, dest, tag, comm));
        return 0;
    };

    int Issend(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
            int tag, Comm const &comm, Request *request) {
        request->set(std::make_shared<_internal::IssendReq>(sendbuf, count,
                datatype, dest, tag, comm));
        return 0;
    };
}

RBC::_internal::IsendReq::IsendReq(const void *sendbuf, int count, MPI_Datatype datatype,
        int dest, int tag, RBC::Comm const &comm) : sendbuf(sendbuf), count(count),
dest(dest), tag(tag), datatype(datatype), comm(comm), requested(false) {
    void* buf = const_cast<void*> (sendbuf);
    MPI_Isend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag, comm.mpi_comm, &request);
};

int RBC::_internal::IsendReq::test(int *flag, MPI_Status *status) {
    return MPI_Test(&request, flag, status);
};

RBC::_internal::IssendReq::IssendReq(const void *sendbuf, int count, MPI_Datatype datatype,
        int dest, int tag, RBC::Comm const &comm) : sendbuf(sendbuf), count(count),
datatype(datatype), dest(dest), tag(tag), comm(comm), requested(false) {
    void* buf = const_cast<void*> (sendbuf);
    MPI_Issend(buf, count, datatype, comm.RangeRankToMpiRank(dest), tag,
            comm.mpi_comm, &request);
};

int RBC::_internal::IssendReq::test(int *flag, MPI_Status *status) {
    return MPI_Test(&request, flag, status);
};
