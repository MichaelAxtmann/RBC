/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#pragma once

// Avoid warnings with within header file of MPI
#if defined(__clang__)

#pragma clang diagnostic ignored "-Weverything"

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC system_header

#elif defined(_MSC_VER)

#endif

#include <mpi.h>
