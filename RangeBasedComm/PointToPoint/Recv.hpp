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
         * Receive operation which invokes MPI_Recv if count > 0
         */
        int RecvNonZeroed(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                Comm const &comm, MPI_Status *status);

    } // namespace _internal

} // namespace RBC
