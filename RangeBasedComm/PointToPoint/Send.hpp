/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2018, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#pragma once

#include "../RBC.hpp"

namespace RBC {

    namespace _internal {
        
        /*
         * Send operation which invokes MPI_Send if count > 0
         */
        int SendNonZeroed(const void *sendbuf, int count, MPI_Datatype datatype, int dest,
                int tag, Comm const &comm);

    } // namespace _internal

} // namespace RBC

