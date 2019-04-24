/*****************************************************************************
 * This file is part of the Project JanusSortRBC
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

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
