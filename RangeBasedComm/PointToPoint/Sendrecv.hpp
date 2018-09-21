/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#pragma once

#include "../RBC.hpp"

namespace RBC {

    namespace _internal {

        /*
         * Sendrecv operation which drops empty messages.
         */
        int SendrecvNonZeroed(void *sendbuf,
                int sendcount, MPI_Datatype sendtype,
                int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                Comm const &comm, MPI_Status *status);

    } // namespace _internal

} // namespace RBC

