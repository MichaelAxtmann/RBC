/******************************************************************************
 * tb_splitter.hpp
 *
 * Source of KaDiSo -- Karlsruhe Distributed Sorting Library
 *
 ******************************************************************************
 * Copyright (C) 2016 Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (C) 2016 Armin Wiebigke <armin.wiebigke@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 2015 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

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
    
    /* Compare two tie-breaking elements of type T */
    friend bool operator<(const TbSplitter& a, const TbSplitter& b) {
        return a.compare(b, std::less<T>());
        // return a.splitter_ < b.splitter_ ||
        //        (!(b.splitter_ < a.splitter_) && a.gid_ < b.gid_);
    }

private:
    T splitter_;
    long long gid_;
};

template <class T>
std::ostream& operator<<(std::ostream& os,
                         const TbSplitter<T>& tbs) {
    os << "TbSplitter{ splitter=" << tbs.splitter_ << " gid=" << tbs.gid_
       << " }";
    return os;
}
