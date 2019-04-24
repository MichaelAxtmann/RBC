# Range-Based Communicators (RBC)

**The algorithm Janus Quicksort (JQuick) has been moved to the repository [KaDiS](https://github.com/MichaelAxtmann/KaDiS).**

This is the implementation of the library Range-Based Communicators (RBC) presented in the paper [Lightweight MPI Communicators with Applications to Perfectly Balanced Janus Quicksort](https://arxiv.org/abs/1710.08027),
which contains an in-depth description of its inner workings, as well as an extensive experimental performance evaluation.
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
> We propose Janus Quicksort, a distributed sorting algorithm that avoids any load imbalances.
> We improved the performance of this algorithm by a factor of 15 for moderate inputs by using RBC communicators.
> Finally, we discuss different approaches to bring nonblocking (local) communicator creation of lightweight (range-based) communicators into MPI.

## RBC Example

```C++
#include <mpi.h>
#include <vector>
#include "RBC.hpp"

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

## Compilation

Make sure to compile with C++11 support.

We give in [example/rbc_example.cpp](example/rbc_example.cpp) a further example of the RBC library. Please compile and execute the example with the following commands:
```
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=mpic++
make rbcexample
mpirun -np 10 ./example
```

Use ```make rbc``` to compile the RBC library.

## Details

RBC uses the tags 1000060-1000079 internally. User of the RBC library must not use those tags when point-to-point communications or collective operations are performed on RBC communicators.
