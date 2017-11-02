/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef REQUESTS_HPP
#define REQUESTS_HPP
#include "RBC.hpp"

/*
 * This class contains one request class for each non-blocking operation.
 */
class Range_Requests {
public:

    /*
     * The request classes for the diffent communication operations
     */    
    class Ibarrier;
    class Ibcast;
    class Igather;
    class Ireduce;
    class Iscan;
    class IscanAndBcast;
    class Isend;
    class Issend;
    class Irecv;
    class Isendrecv;
    class Iallreduce;
    class Iexscan;
    class Iallgather;
};

#endif /* REQUESTS_HPP */


