/*****************************************************************************
 * This file is part of the Project ShizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace Constants {
    //tags for communication
    const int 
            DISTR_SAMPLE_COUNT = 50,
            PIVOT_GATHER = 51, 
            PIVOT_BCAST = 52,
            CALC_EXCH = 60, 
            EXCHANGE_DATA_ASSIGNMENT = 61, 
            EXCHANGE_SMALL = 70, 
            EXCHANGE_LARGE = 71,
            TWO_PE = 80;
}
#endif /* CONSTANTS_HPP */

