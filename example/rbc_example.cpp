/*****************************************************************************
 * This file is part of the Project JanusSortRBC
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <iostream>

#include "RBC.hpp"

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int a = rank;
  int b = size - rank;

  std::cout << "Rank " << rank << ": a=" << a << std::endl;
  std::cout << "Rank " << rank << ": b=" << b << std::endl;

  RBC::Request requests[2];

  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  // Sub-group creation in constant time without communication.
  RBC::Comm comm3;
  int stride3 = 3;
  RBC::Comm_create_group(rcomm, &comm3, 0, size - 1, stride3);

  // Sub-group creation in constant time without communication.
  RBC::Comm comm2;
  int stride2 = 2;
  RBC::Comm_create_group(rcomm, &comm2, 0, size - 1, stride2);

  // Invoking nonblocking broadcast operation.
  if (rank % 2 == 0) {
    RBC::Ibcast(&a, 1, MPI_INT, 0, comm2, requests);
  }

  // Invoking nonblocking broadcast operation.
  if (rank % 3 == 0) {
    RBC::Ibcast(&b, 1, MPI_INT, 0, comm3, requests + 1);
  }

  // Wait until collective operations are finished.
  RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);

  std::cout << "Rank " << rank << ": a=" << a << std::endl;
  std::cout << "Rank " << rank << ": b=" << b << std::endl;

  // Finalize the MPI environment
  MPI_Finalize();
}
