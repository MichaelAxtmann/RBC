/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <mpi.h>
#include "../RBC.hpp"
#include "../Requests.hpp"
#include <cmath>
#include <cstring>
#include <cassert>

int RBC::Exscan(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Exscan(const_cast<void*>(sendbuf), recvbuf, count, datatype, op, comm.mpi_comm);
    } 
    
    int tag = RBC::Tag_Const::EXSCAN;
    int rank, size;
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    int datatype_size = static_cast<int>(type_size);
    int recv_size = count * datatype_size;
    int height = ceil(log2(size));

    int up_height = 0;
    if (rank == (size - 1)) {
        up_height = height;
    } else {
        for (int i = 0; ((rank >> i) % 2 == 1) && (i < height); i++)
            up_height++;
    }

    // last PE has to receive all messages that would be send to ranks >= size
    int tmp_rank = rank;
    if (rank == size - 1)
        tmp_rank = pow(2, height) - 1;

    char* recvbuf_arr = new char[recv_size * (1 + up_height)];
    char* tmp_buf = new char[recv_size];
    char* scan_buf = new char[recv_size];

    int down_height = 0;
    if (rank < size - 1)
        down_height = up_height + 1;

    std::memcpy(scan_buf, sendbuf, recv_size);
    if (size == 1) {
        std::memcpy(recvbuf, sendbuf, recv_size);
        // free buffer
        return 0;
    }

    //upsweep phase
    std::vector<int> target_ranks;
    std::vector<RBC::Request> recv_requests;
    recv_requests.reserve(up_height);

    for (int i = up_height - 1; i >= 0; i--) {
        int source = tmp_rank - pow(2, i);
        if (source < rank) {
            recv_requests.push_back(RBC::Request());
            RBC::Irecv(recvbuf_arr + (recv_requests.size() - 1) * recv_size,
                    count, datatype, source,
                    tag, comm, &recv_requests.back());
            //Save communication partner rank in vector
            target_ranks.push_back(source);
        } else {
            // source rank is not in communicator (or own rank)
            tmp_rank = source;
        }
    }

    RBC::Waitall(recv_requests.size(), &recv_requests.front(), MPI_STATUSES_IGNORE);

    if (recv_requests.size() > 0) {
        //Reduce received data and local data
        for (size_t i = 0; i < (recv_requests.size() - 1); i++) {
            MPI_Reduce_local(recvbuf_arr + i * recv_size,
                    recvbuf_arr + (i + 1) * recv_size, count, datatype, op);
        }
        MPI_Reduce_local(recvbuf_arr + (recv_requests.size() - 1) * recv_size,
                scan_buf, count, datatype, op);
    }

    //Send data
    if (rank < size - 1) {
        int dest = rank + pow(2, up_height);
        if (dest > size - 1)
            dest = size - 1;
        RBC::Send(scan_buf, count, datatype, dest, tag, comm);
        target_ranks.push_back(dest);
    }

    if (rank == size - 1) {
        // set scan buf to "empty" elements
        for (int i = 0; i < recv_size; i++) {
            scan_buf[i] = 0;
        }
    }
    int receives = recv_requests.size();

    //downsweep phase
    int sends = 0;
    bool downsweep_recvd = false;

    for (int cur_height = height; cur_height > 0; --cur_height) {
        if (cur_height == down_height) {
            //Communicate with higher ranks
            std::memcpy(tmp_buf, scan_buf, recv_size);
            int target = target_ranks.back();
            RBC::Sendrecv(tmp_buf, count, datatype, target, tag,
                scan_buf, count, datatype, target, tag, comm, MPI_STATUS_IGNORE);
        } else if (cur_height <= receives) {
            //Communicate with lower ranks
            std::memcpy(tmp_buf, scan_buf, recv_size);
            int target = target_ranks[sends];
            RBC::Sendrecv(tmp_buf, count, datatype, target, tag,
                scan_buf, count, datatype, target, tag, comm, MPI_STATUS_IGNORE);
            sends++;
            if ((pow(2, up_height) - 1 == rank || rank == size - 1)
                    && !downsweep_recvd) {
                // tmp_buf has "empty" elements
                downsweep_recvd = true;                    
            } else {    
                MPI_Reduce_local(tmp_buf, scan_buf, count, datatype, op);
            }
        }
    }
    //End downsweep phase        
    std::memcpy(recvbuf, scan_buf, recv_size);

    delete[] recvbuf_arr;
    delete[] tmp_buf;
    delete[] scan_buf;
    return 0;
}

/*
 * Request for the exscan
 */
class Range_Requests::Iexscan : public RBC::R_Req {
public:
    Iexscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            int tag, MPI_Op op, RBC::Comm const &comm);
    ~Iexscan();
    int test(int *flag, MPI_Status *status);
private:
    const void *sendbuf;
    void *recvbuf;
    int count, tag, rank, size, height, up_height, down_height, receives,
        sends, recv_size, datatype_size, up_height_cnt, tmp_rank, cur_height;
    MPI_Datatype datatype;
    MPI_Op op;
    RBC::Comm comm;
    std::vector<int> target_ranks;
    bool upsweep, downsweep, send, send_up, completed, mpi_collective, recv,
        downsweep_recvd;
    char *recvbuf_arr, *tmp_buf, *scan_buf;
    RBC::Request send_req, recv_req;
    std::vector<RBC::Request> recv_requests;
    MPI_Request mpi_req;
};

int RBC::Iexscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
        MPI_Op op, RBC::Comm const &comm, RBC::Request* request, int tag){
    *request = std::unique_ptr<R_Req>(new Range_Requests::Iexscan(sendbuf, recvbuf, 
            count, datatype, tag, op, comm));
    return 0;
}

Range_Requests::Iexscan::Iexscan(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, int tag, MPI_Op op, RBC::Comm const &comm)  :
        sendbuf(sendbuf), recvbuf(recvbuf), count(count), tag(tag),
        receives(0), sends(0),  datatype(datatype), op(op), comm(comm), upsweep(true), 
        downsweep(false), send(false), send_up(false), completed(false), mpi_collective(false),
        recv(false), downsweep_recvd(false), recvbuf_arr(nullptr), 
        tmp_buf(nullptr), scan_buf(nullptr) {    
#ifndef NO_IBCAST
    if (comm.useMPICollectives()) {
        MPI_Iscan(sendbuf, recvbuf, count, datatype, op, comm.mpi_comm, &mpi_req);
        mpi_collective = true;
        return;
    }
#endif
    RBC::Comm_rank(comm, &rank);
    RBC::Comm_size(comm, &size);
    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_size = static_cast<int>(type_size);
    recv_size = count * datatype_size;
    height = ceil(log2(size));

    up_height = 0;
    if (rank == (size - 1)) {
        up_height = height;
    } else {
        for (int i = 0; ((rank >> i) % 2 == 1) && (i < height); i++)
            up_height++;
    }
    up_height_cnt = up_height - 1; 
    
    // last PE has to receive all messages that would be send to ranks >= size
    tmp_rank = rank;
    if (rank == size - 1)
        tmp_rank = pow(2, height) - 1;
    
    recvbuf_arr = new char[recv_size * (1 + up_height)];
    tmp_buf = new char[recv_size];
    scan_buf = new char[recv_size];

    down_height = 0;
    if (rank < size - 1)
        down_height = up_height + 1;

    std::memcpy(scan_buf, sendbuf, recv_size);  
}

Range_Requests::Iexscan::~Iexscan() {
    if (recvbuf_arr != nullptr)
        delete[] recvbuf_arr;
    if (tmp_buf != nullptr)
        delete[] tmp_buf;
    if (scan_buf != nullptr)
        delete[] scan_buf;
}

int Range_Requests::Iexscan::test(int* flag, MPI_Status* status) {
    if (completed) {
        *flag = 1;
        return 0;
    }
    
    if (mpi_collective)
        return MPI_Test(&mpi_req, flag, status);

    if (size == 1 && upsweep) {
        std::memcpy(recvbuf, sendbuf,
                recv_size);
        upsweep = false;
        downsweep = false;
    }
    //upsweep phase
    if (upsweep) {
        if (up_height_cnt >= 0) {
            if (!recv) {
                int source = tmp_rank - pow(2, up_height_cnt);
                if (source < rank) {
                    RBC::Irecv(recvbuf_arr + recv_size * receives,
                            count, datatype, source,
                            tag, comm, &recv_req);
                    //Save communication partner rank in vector
                    target_ranks.push_back(source);
                    recv = true;
                    receives++;
                } else {
                    // source rank is not in communicator (or own rank)
                    tmp_rank = source;
                    up_height_cnt--;
                }
            } else {
                int finished;
                RBC::Test(&recv_req, &finished, MPI_STATUS_IGNORE);
                if (finished) {
                    assert(receives > 0);
                    if (receives > 1) {
                        MPI_Reduce_local(recvbuf_arr + (receives - 2) * recv_size,
                                recvbuf_arr + (receives - 1) * recv_size, count, datatype, op);
                    }
                    recv = false;
                    up_height_cnt--;
                }
            }
        }

        // Everything received
        if (up_height_cnt < 0 && !send_up) {
            if (receives > 0) {
                MPI_Reduce_local(recvbuf_arr + recv_size * (receives - 1),
                        scan_buf, count, datatype, op);
            }

            //Send data
            if (rank < size - 1) {
                int dest = rank + pow(2, up_height);
                if (dest > size - 1)
                    dest = size - 1;
                RBC::Isend(scan_buf, count, datatype, dest, tag, comm, &send_req);
                target_ranks.push_back(dest);
            }
            send_up = true;
        }

        //End upsweep phase when data has been send
        if (send_up) {
            int finished = 1;
            if (rank < size - 1)
                RBC::Test(&send_req, &finished, MPI_STATUS_IGNORE);

            if (finished) {
                upsweep = false;
                downsweep = true;
                cur_height = height;
                if (rank == size - 1) {
                    // set scan buf to "empty" elements
                    for (int i = 0; i < recv_size; i++) {
                        scan_buf[i] = 0;
                    }
                }
            }
        }
    }

    //downsweep phase
    if (downsweep) {
        int finished1 = 0, finished2 = 0;
        if (down_height == cur_height) {
            //Communicate with higher ranks
            if (!send) {
                std::memcpy(tmp_buf, scan_buf, recv_size);
                int dest = target_ranks.back();
                RBC::Isend(tmp_buf, count, datatype, dest, tag, comm, &send_req);
                RBC::Irecv(scan_buf, count, datatype, dest, tag, comm, &recv_req);
                send = true;
            }
            RBC::Test(&send_req, &finished1, MPI_STATUS_IGNORE);
            RBC::Test(&recv_req, &finished2, MPI_STATUS_IGNORE);
        } else if (receives >= cur_height) {
            //Communicate with lower ranks
            if (!send) {
                std::memcpy(tmp_buf, scan_buf, recv_size);
                int dest = target_ranks[sends];
                RBC::Isend(tmp_buf, count, datatype, dest, tag, comm, &send_req);
                RBC::Irecv(scan_buf, count, datatype, dest, tag, comm, &recv_req);
                sends++;
                send = true;
            }
            
            RBC::Test(&send_req, &finished1, MPI_STATUS_IGNORE);
            RBC::Test(&recv_req, &finished2, MPI_STATUS_IGNORE);
            if (finished1 && finished2) {
                if ((pow(2, up_height) - 1 == rank || rank == size - 1)
                        && !downsweep_recvd) {
                    // tmp_buf has "empty" elements
                    downsweep_recvd = true;                    
                } else {    
                    MPI_Reduce_local(tmp_buf, scan_buf, count, datatype, op);
                }
            }
        } else
            cur_height--;
        //Send and receive completed
        if (finished1 && finished2) {
            cur_height--;
            send = false;
        }
        //End downsweep phase
        if (cur_height == 0) {
            std::memcpy(recvbuf, scan_buf, recv_size);
            downsweep = false;
        }
    }

    if (!upsweep && !downsweep) {
        *flag = 1;
        completed = true;
    }
    return 0;
}

