/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SQUICK_HPP
#define SQUICK_HPP

#include <functional>
#include <iterator>
#include <type_traits>

#include "SQuick/QuickSort.hpp"

namespace SQuick {

/**
 * Helper function for creating a reusable parallel sorter.
 */
template<class value_type>
QuickSort<value_type> make_sorter(int seed = 1, long long min_samples = 64) {
    return QuickSort<value_type>(seed, min_samples, false);
}

/**
 * Configurable interface.
 */
template<class value_type, class Compare = std::less<value_type>>
    void sort(MPI_Comm mpi_comm, std::vector<value_type> &data,
              long long global_elements = -1, Compare comp = Compare()) {
    make_sorter<value_type>().sort(mpi_comm, data, global_elements, std::move(comp));
}

/**
 * Configurable interface.
 */
template<class value_type, class Compare = std::less<value_type>>
    void sort(RBC::Comm rbc_comm, std::vector<value_type> &data,
              long long global_elements = -1, Compare comp = Compare()) {
    make_sorter<value_type>().sort_rbc(rbc_comm, data, global_elements, std::move(comp));
}

} // namespace SQuick

#endif // SQUICK_HPP
