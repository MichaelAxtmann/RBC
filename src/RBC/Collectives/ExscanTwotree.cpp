/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2018-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>

#include "RBC.hpp"
#include "tlx/math.hpp"

#include "Twotree.hpp"

namespace RBC {
namespace _internal {
namespace Twotree {
int Exscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
           RBC::Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Exscan(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.get());
  }

  if (count == 0) {
    return 0;
  }

  int size = 0;
  Comm_size(comm, &size);
  if (size == 1) {
    return 0;
  }

  const int tag = Tag_Const::EXSCAN;
  const int rank = comm.getRank();
  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  int datatype_size = static_cast<int>(type_size);
  int recv_size = count * datatype_size;

  int msg_cnt = 0;
  std::unique_ptr<char[]> tmpbuf;
  RBC::Request requests[2];

  if (rank + 1 < size) {
    RBC::Isend(sendbuf, count, datatype, rank + 1, tag, comm, requests);
    ++msg_cnt;
  }
  if (rank > 0 && size == 2) {
    RBC::Irecv(recvbuf, count, datatype, rank - 1, tag, comm, requests + msg_cnt);
    ++msg_cnt;
  } else if (rank > 0) {
    tmpbuf.reset(new char[recv_size]);
    RBC::Irecv(tmpbuf.get(), count, datatype, rank - 1, tag, comm, requests + msg_cnt);
    ++msg_cnt;
  }

  RBC::Waitall(msg_cnt, requests, MPI_STATUSES_IGNORE);

  if (size == 2) {
    return 0;
  }

  if (rank > 0) {
    assert(size > 2);
    RBC::Comm subcomm;
    RBC::Comm_create_group(comm, &subcomm, 1, size - 1);
    return RBC::_internal::optimized::ScanTwotree(tmpbuf.get(), recvbuf,
                                                  count, datatype, op, subcomm);
  }
  return 0;
}
}  // end namespace Twotree

namespace optimized {
int ExscanTwotree(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                  RBC::Comm const& comm) {
  return _internal::Twotree::Exscan(sendbuf, recvbuf, count, datatype, op, comm);
}
}  // namespace optimized
}  // end namespace _internal
}  // end namespace RBC
