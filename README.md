# Range-Based Communicators (RBC)

This is the implementation of the library Range-Based Communicators (RBC) presented in the paper [Lightweight MPI Communicators with Applications to Perfectly Balanced 'Schizophrenic' Quicksort](https://arxiv.org/abs/1710.08027),
which contains an in-depth description of its inner workings, as well as an extensive experimental performance evaluation.
This repository also contains an implementation of 'Schizophrenic' Quicksort (SQuick), a perfectly balanced distributed sorting algorithm.
SQuick is implemented with RBC communicators as well as MPI communicators.
Here's the abstract:

> MPI uses the concept of communicators to connect groups of processes.
> It provides nonblocking collective operations on communicators to overlap communication and computation.
> Flexible algorithms demand flexible communicators.
> E.g., a process can work on different subproblems within different process groups simultaneously, new process groups can be created, or the members of a process group can change.
> Depending on the number of communicators, the time for communicator creation can drastically increase the running time of the algorithm.
> Furthermore, a new communicator synchronizes all processes as communicator creation routines are blocking collective operations.
> 
> We present RBC, a communication library based on MPI, that creates range-based communicators in constant time without communication.
> These RBC communicators support (non)blocking point-to-point communication as well as (non)blocking collective operations.
> Our experiments show that the library reduces the time to create a new communicator by a factor of more than 400 whereas the running time of collective operations remains about the same.
> We propose ``Schizophrenic'' Quicksort, a distributed sorting algorithm that avoids any load imbalances.
> We improved the performance of this algorithm by a factor of 15 for moderate inputs by using RBC communicators.
> Finally, we discuss different approaches to bring nonblocking (local) communicator creation of lightweight (range-based) communicators into MPI.

## RBC Example

```C++
#include <mpi.h>
#include <vector>
#include "RangeBasedComm/RBC.hpp"

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    
    // Create a RBC communicator including one half of all ranks
    RBC::Comm global_comm, new_comm;
    RBC::Create_Comm_from_MPI(MPI_COMM_WORLD, &global_comm);
    int rank, size, first, last;
    RBC::Comm_rank(global_comm, &rank);
    RBC::Comm_size(global_comm, &size);
    if (rank < size / 2) {
        first = 0; 
        last  = size / 2 - 1;
    } else {
        first = size / 2; 
        last  = size - 1;
    }    
    RBC::Create_Comm(global_comm, &new_comm, first, last);
    
    // Non-blocking broadcast on new communicator
    std::vector<int> data(200, rank);
    int flag = 0, root = 0;
    RBC::Request req; 
    RBC::Ibcast(data.data(), data.size(), MPI_INT, root, new_comm, &req);
    while (!flag) {
        // do something else
        RBC::Test(&req, &flag, MPI_STATUS_IGNORE);
    }

    // Finalize the MPI environment
    MPI_Finalize();
}
```

Make sure to compile with C++11 support.


## SQuick with MPI communicators and RBC communicators

```C++
#include "Sort/SQuick.hpp"

MPI_Comm comm = ...;
std::vector<T> data = ...;
long long global_elements = data.size() * nprocs;

// sort with MPI_Comm
SQuick::sort(comm, data, global_elements[, comparator]);

// sort with RBC::Comm
RBC::Comm rcomm;
RBC::Create_Comm_from_MPI(comm, &rcomm);
SQuick::sort(rcomm, data, global_elements[, comparator]);

// sort with RBC::comm but use MPI communicators and collectives
RBC::Comm rcomm1;
bool is_use_mpi_comm = true;
bool is_use_mpi_coll = true;
RBC::Create_Comm_from_MPI(comm, &rcomm1,
                          is_use_mpi_comm, is_use_mpi_coll);
SQuick::sort(rcomm1, data, global_elements[, comparator]);
```

We give in [example.cpp](example.cpp) a full version of the SQuick example above. Please compile and execute the example with the following commands:
```
make all
mpirun -np 10 ./build/example
```

Make uses `mpic++` to compile the RBC library and the SQuick example.