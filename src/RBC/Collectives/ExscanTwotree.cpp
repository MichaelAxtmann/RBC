/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2018-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

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
  MPI_Request requests[2];

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

  MPI_Waitall(msg_cnt, requests, MPI_STATUSES_IGNORE);

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
