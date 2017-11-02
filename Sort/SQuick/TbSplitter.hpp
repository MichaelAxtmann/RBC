/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (C) 2016 Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (C) 2016 Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#pragma once

#include <stddef.h>
#include <iostream>

#include <mpi.h>

template <class T>
class TbSplitter {
public:
    TbSplitter() {}
    
    TbSplitter(const T& splitter, const long long& gid) noexcept
        : splitter_(splitter),
          gid_(gid) {}

    long long GID() const {
        return gid_;
    }
    void set_gid(long long gid) {
        gid_ = gid;
    }

    const T& Splitter() const {
        return splitter_;
    }
    T& Splitter() {
        return splitter_;
    }
    void set_splitter(const T& splitter) {
        splitter_ = splitter;
    }
    
    static MPI_Datatype MpiType(const MPI_Datatype& base_type) {
        const int nitems = 2;
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2];
        types[0] = base_type;
        types[1] = MPI_LONG_LONG;
        MPI_Datatype mpi_tb_splitter_type;
        MPI_Aint offsets[2];

        offsets[0] = offsetof(TbSplitter<T>, splitter_);
        offsets[1] = offsetof(TbSplitter<T>, gid_);

        MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                               &mpi_tb_splitter_type);
        MPI_Type_commit(&mpi_tb_splitter_type);

        return mpi_tb_splitter_type;
    }

    template<class Compare>
    bool compare(const TbSplitter& b, Compare comp) const {
        return comp(this->splitter_, b.splitter_) ||
            (!comp(b.splitter_, this->splitter_) && this->gid_ < b.gid_);
    }
    
    /* Compare two elements of type T and break ties */
    friend bool operator<(const TbSplitter& a, const TbSplitter& b) {
        return a.compare(b, std::less<T>());
    }

private:
    T splitter_;
    long long gid_;
};
