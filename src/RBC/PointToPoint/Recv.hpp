/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include "RBC.hpp"

namespace RBC {
namespace _internal {
/*
 * Receive operation which invokes MPI_Recv if count > 0
 */
int RecvNonZeroed(void* buf, int count, MPI_Datatype datatype, int source, int tag,
                  Comm const& comm, MPI_Status* status);
}  // namespace _internal
}  // namespace RBC
