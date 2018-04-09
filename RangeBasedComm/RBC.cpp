/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#include <cassert>
#include <memory>
#include <mpi.h>
#include <iostream>
#include <vector>

#include "RBC.hpp"
#include "RangeGroup.hpp"

RBC::Comm::Comm() : mpi_comm(MPI_COMM_NULL), rank(-1), size(0) {
}

RBC::Comm::~Comm() {
}

// Create from MPI_Comm
RBC::Comm::Comm(MPI_Comm mpi_comm, bool use_mpi_collectives,
        bool split_mpi_comm, bool use_comm_create)
        : mpi_comm(mpi_comm), use_mpi_collectives(use_mpi_collectives),
        split_mpi_comm(split_mpi_comm), use_comm_create(use_comm_create),
        is_mpi_comm(true), free_mpi_comm(false)  {
    int size;
    MPI_Comm_size(mpi_comm, &size);
    range_group = RangeGroup(0, size - 1, 1);
    init();
}

// Create from RBC::Comm without MPI_Comm split
RBC::Comm::Comm(RangeGroup range_group, Comm parent_comm)
        : mpi_comm(parent_comm.mpi_comm), use_mpi_collectives(parent_comm.use_mpi_collectives),
        split_mpi_comm(parent_comm.split_mpi_comm), use_comm_create(parent_comm.use_comm_create),
        is_mpi_comm(false), free_mpi_comm(false), 
        range_group(range_group) {
    init();
}

// Create from RBC::Comm with MPI_Comm_split
RBC::Comm::Comm(MPI_Comm mpi_comm, RBC::Comm parent_comm)
        : mpi_comm(mpi_comm), use_mpi_collectives(parent_comm.use_mpi_collectives),
        split_mpi_comm(parent_comm.split_mpi_comm), use_comm_create(parent_comm.use_comm_create),
        is_mpi_comm(true), free_mpi_comm(true) {    
    int size;
    MPI_Comm_size(mpi_comm, &size);
    range_group = RangeGroup(0, size - 1, 1);
    init();
}


void RBC::Comm::init() {
    assert(MPI_COMM_NULL != mpi_comm);
    int global_rank;
    MPI_Comm_rank(mpi_comm, &global_rank);
    if (range_group.IsMpiRankIncluded(global_rank))
        rank = MpiRankToRangeRank(global_rank);
    else
        rank = -1;
    size = range_group.getSize();    
    
    if (!is_mpi_comm)
        assert(!split_mpi_comm);
    if (free_mpi_comm)
        assert(is_mpi_comm);
}


int RBC::Create_Comm_from_MPI(MPI_Comm mpi_comm, RBC::Comm *rcomm, 
        bool use_mpi_collectives, bool split_mpi_comm, bool use_comm_create) {
    *rcomm = RBC::Comm(mpi_comm, use_mpi_collectives, split_mpi_comm,
            use_comm_create);
    return 0;
}

RBC::Comm RBC::Create_Comm_from_MPI(MPI_Comm mpi_comm,
        bool use_mpi_collectives, bool split_mpi_comm, bool use_comm_create) {
    return RBC::Comm(mpi_comm, use_mpi_collectives, split_mpi_comm,
            use_comm_create);
}

int RBC::Comm_create(RBC::Comm const &comm, RBC::Comm *new_comm,
        int first, int last, int stride) {
    if (comm.splitMPIComm()) {
        int rank;
        RBC::Comm_rank(comm, &rank);
        MPI_Group group, new_group;
        MPI_Comm_group(comm.mpi_comm, &group);
        int ranges[3] = {first, last, stride};
        MPI_Group_range_incl(group, 1, &ranges, &new_group);
        MPI_Comm new_mpi_comm;
        
        MPI_Comm_create(comm.mpi_comm, new_group, &new_mpi_comm);
        
        if (new_mpi_comm != MPI_COMM_NULL) {
            *new_comm = RBC::Comm(new_mpi_comm, comm);
        }
    } else {
        int rank;
        RBC::Comm_rank(comm, &rank);
        int mpi_rank = comm.RangeRankToMpiRank(rank);
        RangeGroup range_group = comm.range_group.Split(first, last, stride);
        if (range_group.IsMpiRankIncluded(mpi_rank)) {
            *new_comm = RBC::Comm(range_group, comm);
        }
    }
    return 0;
} 
 
RBC::Comm RBC::Comm_create(RBC::Comm const &comm,
        int first, int last, int stride) {
    RBC::Comm new_comm;
    RBC::Comm_create(comm, &new_comm, first, last, stride);
    return new_comm;    
}

int RBC::Comm_create_group(RBC::Comm const &comm, RBC::Comm *new_comm,
        int first, int last, int stride) {
    if (comm.splitMPIComm()) {
        MPI_Group group, new_group;
        MPI_Comm_group(comm.mpi_comm, &group);
        int ranges[3] = {first, last, stride};
        MPI_Group_range_incl(group, 1, &ranges, &new_group);
        MPI_Comm new_mpi_comm;
#ifndef NO_IBCAST
        MPI_Comm_create_group(comm.mpi_comm, new_group, 0, &new_mpi_comm);
#else
        MPI_Comm_create(comm.mpi_comm, new_group, &new_mpi_comm);
#endif        
        if (new_mpi_comm != MPI_COMM_NULL) {
            *new_comm = RBC::Comm(new_mpi_comm, comm);
        }
    } else {
        RBC::Comm_create(comm, new_comm, first, last, stride);
    }
    return 0;
}

int RBC::Split_Comm(Comm const &comm, int left_start, int left_end, int right_start, 
        int right_end, Comm* left_comm, Comm* right_comm) {
    if (!comm.splitMPIComm()) {        
        int rank;
        RBC::Comm_rank(comm, &rank);
        if (rank >= left_start && rank <= left_end)
            RBC::Comm_create(comm, left_comm, left_start, left_end);
        if (rank >= right_start && rank <= right_end)
            RBC::Comm_create(comm, right_comm, right_start, right_end);
    } else {
        //split MPI communicator
        assert(comm.is_mpi_comm);
        MPI_Comm mpi_comm = comm.mpi_comm, mpi_left, mpi_right;
        int rank, size;
        MPI_Comm_rank(mpi_comm, &rank);
        MPI_Comm_size(mpi_comm, &size);
        
        //create MPI communicators
        if (left_end < right_start) {
            //disjoint communicators
            if (comm.use_comm_create) {
                MPI_Group group, new_group = MPI_GROUP_EMPTY;
                MPI_Comm_group(comm.mpi_comm, &group);
                int ranges[2][3] = {
                    {left_start, left_end, 1},
                    {right_start, right_end, 1}};
                if (rank >= left_start && rank <= left_end)
                    MPI_Group_range_incl(group, 1, &ranges[0], &new_group);
                else if (rank >= right_start && rank <= right_end)
                    MPI_Group_range_incl(group, 1, &ranges[1], &new_group);

                MPI_Comm new_mpi_comm;
#ifndef NO_IBCAST
                MPI_Comm_create_group(comm.mpi_comm, new_group, 0, &new_mpi_comm);
#else
                MPI_Comm_create(comm.mpi_comm, new_group, &new_mpi_comm);
#endif
                if (rank >= left_start && rank <= left_end) {
                    mpi_left = new_mpi_comm;
                    mpi_right = MPI_COMM_NULL;
                } else {
                    mpi_left = MPI_COMM_NULL;
                    mpi_right = new_mpi_comm;
                }                
            } else {
                int color;
                if (rank >= left_start && rank <= left_end)
                    color = 1;
                else if (rank >= right_start && rank <= right_end)
                    color = 2;
                else
                    color = MPI_UNDEFINED;

                MPI_Comm new_comm;
                MPI_Comm_split(comm.mpi_comm, color, rank, &new_comm);

                if (color == 1) {
                    mpi_left = new_comm;
                    mpi_right = MPI_COMM_NULL;
                } else {
                    mpi_left = MPI_COMM_NULL;
                    mpi_right = new_comm;
                }
            }
        } else {
            //overlapping communicators
            if (comm.use_comm_create) {
                MPI_Group group, new_group_left = MPI_GROUP_EMPTY, new_group_right = MPI_GROUP_EMPTY;
                MPI_Comm_group(comm.mpi_comm, &group);
                int ranges[2][3] = {
                    {left_start, left_end, 1},
                    {right_start, right_end, 1}};
                if (rank >= left_start && rank <= left_end)
                    MPI_Group_range_incl(group, 1, &ranges[0], &new_group_left);
                if (rank >= right_start && rank <= right_end)
                    MPI_Group_range_incl(group, 1, &ranges[1], &new_group_right);

#ifndef NO_IBCAST
                if (rank >= left_start && rank <= left_end)
                    MPI_Comm_create_group(comm.mpi_comm, new_group_left, 0, &mpi_left);
                if (rank >= right_start && rank <= right_end)
                    MPI_Comm_create_group(comm.mpi_comm, new_group_right, 0, &mpi_right);
#else
                MPI_Comm_create(comm.mpi_comm, new_group_left, &mpi_left);
                MPI_Comm_create(comm.mpi_comm, new_group_right, &mpi_right);
#endif            
            } else {
                int color1, color2;
                if (rank >= left_start && rank <= left_end)
                    color1 = 1;
                else {
                    color1 = MPI_UNDEFINED;
                }

                if (rank >= right_start && rank <= right_end)
                    color2 = 2;
                else {
                    color2 = MPI_UNDEFINED;
                }

                MPI_Comm_split(comm.mpi_comm, color1, rank, &mpi_left);
                MPI_Comm_split(comm.mpi_comm, color2, rank, &mpi_right);
            }
        }
        
        if (rank >= left_start && rank <= left_end) {
            assert(mpi_left != MPI_COMM_NULL);
            *left_comm = RBC::Comm(mpi_left, comm);
        }
        if (rank >= right_start && rank <= right_end) {
            assert(mpi_right != MPI_COMM_NULL);
            *right_comm = RBC::Comm(mpi_right, comm);
        }
    }
    return 0;
}

int RBC::Comm_free(Comm &comm) {
    if (comm.is_mpi_comm && comm.free_mpi_comm) {
        return MPI_Comm_free(&comm.mpi_comm);
    }
    return 0;
}

const MPI_Comm& RBC::Comm::GetMpiComm() const {
  return mpi_comm;
}

int RBC::Comm::MpiRankToRangeRank(int mpi_rank) const {
    return range_group.MpiRankToRangeRank(mpi_rank);
}

int RBC::Comm::RangeRankToMpiRank(int range_rank) const {
    return range_group.RangeRankToMpiRank(range_rank);
}

bool RBC::Comm::useMPICollectives() const {
    return use_mpi_collectives && is_mpi_comm;
}

bool RBC::Comm::splitMPIComm() const {
    if (split_mpi_comm)
        assert(is_mpi_comm);
    return split_mpi_comm;
}

bool RBC::Comm::includesMpiRank(int rank) const {
    return range_group.IsMpiRankIncluded(rank);
}

bool RBC::Comm::isEmpty() const {
    return mpi_comm == MPI_COMM_NULL;
}

std::ostream& RBC::operator<<(std::ostream& os,
                                const RBC::Comm& comm) {
    os 
            << "(" 
            << comm.mpi_comm << ", " << comm.range_group
            << ")";
    return os;
}

RBC::Request::Request() {
}

void RBC::Request::set(const std::shared_ptr<_internal::RequestSuperclass>& req) {   
    req_ptr = req;
}

RBC::Request& RBC::Request::operator=(const Request& req) {
    this->set(req.req_ptr);
    return *this;
}

int RBC::Request::test(int* flag, MPI_Status* status) {
    *flag = 0;
    return req_ptr->test(flag, status);
}

int RBC::Comm_rank(RBC::Comm const &comm, int *rank) {
    *rank = comm.rank;
    return 0;
}

int RBC::Comm_size(RBC::Comm const &comm, int *size) {
    *size = comm.size;
    return 0;
}

int RBC::Iprobe(int source, int tag, RBC::Comm const &comm, int *flag, MPI_Status *status) {
    if (source != MPI_ANY_SOURCE)
        source = comm.RangeRankToMpiRank(source);
    MPI_Status tmp_status;
    int return_value = MPI_Iprobe(source, tag, comm.mpi_comm, flag, &tmp_status);
    if (*flag) {
        if (!comm.includesMpiRank(tmp_status.MPI_SOURCE))
            *flag = 0;
        else if (status != MPI_STATUS_IGNORE)
            *status = tmp_status;
    }
    return return_value;
}

int RBC::Probe(int source, int tag, RBC::Comm const &comm, MPI_Status *status) {
    if (source != MPI_ANY_SOURCE) {
        source = comm.RangeRankToMpiRank(source);
        return MPI_Probe(source, tag, comm.mpi_comm, status);
    }
    
    int flag = 0;
    while (!flag)
        RBC::Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, status);
    return 0;
}

int RBC::Test(RBC::Request *request, int *flag, MPI_Status *status) {
    return request->test(flag, status);
}

int RBC::Testall(int count, RBC::Request *array_of_requests, int* flag,
        MPI_Status array_of_statuses[]) {
    *flag = 1;
    for (int i = 0; i < count; i++) {
        int temp_flag;
        if (array_of_statuses == MPI_STATUSES_IGNORE) {
            Test(&array_of_requests[i], &temp_flag, MPI_STATUS_IGNORE);
        } else {
            Test(&array_of_requests[i], &temp_flag, &array_of_statuses[i]);
        }
        if (temp_flag == 0) {
            *flag = 0;
        }
    }
    return 0;
}

int RBC::Wait(RBC::Request *request, MPI_Status *status) {
    int flag = 0, return_value;
    while (flag == 0) {
        return_value = Test(request, &flag, status);
    }
    return return_value;
}

int RBC::Waitall(int count, RBC::Request array_of_requests[],
                 MPI_Status array_of_statuses[]) {
    // We are not allowed to call Testall until success
    // as Testall calls Test on requests which are already successful.
    int pending_requests = count;
    std::vector<bool> succs(count, false);
    while (pending_requests > 0) {
        for (int i = 0; i < count; i++) {
            if (succs[i]) continue;
            int succ;
            if (array_of_statuses == MPI_STATUSES_IGNORE) {
                Test(&array_of_requests[i], &succ, MPI_STATUS_IGNORE);
            } else {
                Test(&array_of_requests[i], &succ, &array_of_statuses[i]);
            }
            if (succ == 1) {
                --pending_requests;
                succs[i] = true;
            }
        }
    }
    return 0;
}

int RBC::get_Rank_from_Status(RBC::Comm const &comm, MPI_Status status) {
    return comm.MpiRankToRangeRank(status.MPI_SOURCE);
}
