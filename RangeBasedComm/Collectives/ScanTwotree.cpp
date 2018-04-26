/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2018, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include "../RBC.hpp"
#include "Twotree.hpp"

#include <mpi.h>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cassert>
#include "cstring"

namespace RBC {

namespace _internal {

namespace Twotree {

class ScanExecuter {
public:
    ScanExecuter() = delete;

    enum Color {Red = 0, Black = 1};
    
    static ScanExecuter get(const void *sendbuf, void *recvbuf, int local_el_cnt, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm) {
        const int tag = Tag_Const::SCANTWOTREE;
        int datatype_byte_cnt;
        int input_byte_cnt;
        int rank, nprocs;

        Comm_rank(comm, &rank);
        Comm_size(comm, &nprocs);

        MPI_Aint lb, type_size;
        MPI_Type_get_extent(datatype, &lb, &type_size);
        datatype_byte_cnt = static_cast<int> (type_size);
        input_byte_cnt = local_el_cnt * datatype_byte_cnt;

        std::unique_ptr<char[]> mysendbuf(new char[input_byte_cnt]);
        std::unique_ptr<char[]> tmpbuf(new char[input_byte_cnt]);

        std::memcpy(mysendbuf.get(), sendbuf, input_byte_cnt);
        std::memcpy(recvbuf, sendbuf, input_byte_cnt);
        
        const int max_package_el_cnt  = ScanExecuter::MaxPackageElCnt(nprocs, local_el_cnt, datatype_byte_cnt);
        const int package_cnt       = (local_el_cnt + max_package_el_cnt - 1) / max_package_el_cnt;
        const int bottom_package_cnt = (package_cnt + 1) / 2;
        const int top_package_cnt  = package_cnt - bottom_package_cnt;

        return ScanExecuter(std::move(mysendbuf), recvbuf, local_el_cnt, datatype, op, comm,
                                 tag, datatype_byte_cnt, input_byte_cnt, rank, nprocs,
                                 max_package_el_cnt, package_cnt, top_package_cnt, bottom_package_cnt,
                                 std::move(tmpbuf));
    }

    void execute() {

        /*
         * Input is stored in mysendbuf (copy of sendbuf), and recvbuf (input buffer).
         * Additional buffer tmpbuf.
         *
         * Reduce:
         * Send package from sendbuf to parent.
         * Receive package into tmpbuf.
         * Packages from left child are reduced with input elements into recvbuf.
         * Packages from left child and right child are reduced with input elements into sendbuf.
         *
         * Invariant after reduce:
         * mysendbuf contains input elements.
         * tmpbuf contains elements from left child.
         * recvbuf contains input elements combined with elements from left child.
         * sendbuf contains input elements combined with elements from left child and right child.
         *
         * Bcast:
         * - Receive scan:
         * Receive elements from parent into tmpbuf(=elements from left subtrees).
         * If we received elements from parent into tmpbuf (we have a subtree to our left)
         * we combine the received elements from tmpbuf with recvbuf(=input elements, elements from left child) into recvbuf(=input els, els from left child, els from left subtrees).
         * Afterwards, recvbuf contains the scan output. 
         * Otherwise, recvbuf already contained the scan output.
         * - Send scan:
         * Send tmpbuf(=elements from left subtrees) to left child if we already received a package from our parent.
         * Otherwise, we are one of the leftmost children of the tree 
         * and we did not receive any elements from our parent (as there are no subtrees to our left).
         * In this case, we send a message of size zero to our left child.
         * Send recvbuf(=input_els, els from left child, els from left subtrees) to right child.
         *
         * Output recvbuf.
         *
         * Tree:
         * We send even packages over bottom tree.
         * We send odd packages over top tree.
         * We send red packages (0) in even steps.
         * We send black packages (1) in odd steps.
         */

        const int lchild_top_delay = ChildDelay(tree.parent_top, tree.delay_top, tree.lchild_top,
                                               tree.incolor_top, tree.loutcolor_top);
        const int rchild_top_delay = ChildDelay(tree.parent_top, tree.delay_top, tree.rchild_top,
                                               tree.incolor_top, tree.routcolor_top);
        const int lchild_bottom_delay = ChildDelay(tree.parent_bottom, tree.delay_bottom, tree.lchild_bottom,
                                                 1^tree.incolor_top, tree.loutcolor_bottom);
        const int rchild_bottom_delay = ChildDelay(tree.parent_bottom, tree.delay_bottom, tree.rchild_bottom,
                                                 1^tree.incolor_top, tree.routcolor_bottom);

        // Red in edge
        int in_red_delay = -1;
        int in_red = -1;
        int in_red_top_tree = 1; // 0 for bottom tree, 1 fro top tree

        // Black in edge
        int in_black_delay = -1;
        int in_black = -1;
        int in_black_top_tree = 1; // 0 for bottom tree, 1 fro top tree

        // Red out edge
        int out_red_delay = -1;
        int out_red = -1;
        int out_red_top_tree = 1; // 0 for bottom tree, 1 fro top tree

        // Black out edge
        int out_black_delay = -1;
        int out_black = -1;
        int out_black_top_tree = 1; // 0 for bottom tree, 1 fro top tree

        // Number of steps
        int num_steps = 0;
        
        if (tree.parent_top != -1) {
            if (tree.incolor_top == Color::Red) {
                assert(in_red == -1);
                in_red = tree.parent_top;
                in_red_delay = tree.delay_top;
                num_steps = std::max(num_steps, in_red_delay + 2 * (top_package_cnt - 1) + 1);
            } else {
                assert(in_black == -1);
                assert(tree.incolor_top == Color::Black);
                in_black = tree.parent_top;
                in_black_delay = tree.delay_top;
                num_steps = std::max(num_steps, in_black_delay + 2 * (top_package_cnt - 1) + 1);
            }
        }
        if (tree.parent_bottom != -1) {
            if (tree.incolor_top == Color::Black) {
                assert(in_red == -1);
                in_red = tree.parent_bottom;
                in_red_delay = tree.delay_bottom;
                in_red_top_tree = 0;
                num_steps = std::max(num_steps, in_red_delay + 2 * (bottom_package_cnt - 1) + 1);
            } else {
                assert(in_black == -1);
                assert(tree.incolor_top == Color::Red);
                in_black = tree.parent_bottom;
                in_black_delay = tree.delay_bottom;
                in_black_top_tree = 0;
                num_steps = std::max(num_steps, in_black_delay + 2 * (bottom_package_cnt - 1) + 1);
            }
        }
        
        if (tree.lchild_top != -1) {
            if (tree.loutcolor_top == Color::Red) {
                assert(out_red == -1);
                out_red = tree.lchild_top;
                out_red_delay = lchild_top_delay;
                num_steps = std::max(num_steps, out_red_delay + 2 * (top_package_cnt - 1) + 1);
            } else {
                assert(out_black == -1);
                assert(tree.loutcolor_top == Color::Black);
                out_black = tree.lchild_top;
                out_black_delay = lchild_top_delay;
                num_steps = std::max(num_steps, out_black_delay + 2 * (top_package_cnt - 1) + 1);
            }
        }
        if (tree.rchild_top != -1) {
            if (tree.routcolor_top == Color::Red) {
                assert(out_red == -1);
                out_red = tree.rchild_top;
                out_red_delay = rchild_top_delay;
                num_steps = std::max(num_steps, out_red_delay + 2 * (top_package_cnt - 1) + 1);
            } else {
                assert(out_black == -1);
                assert(tree.routcolor_top == Color::Black);
                out_black = tree.rchild_top;
                out_black_delay = rchild_top_delay;
                num_steps = std::max(num_steps, out_black_delay + 2 * (top_package_cnt - 1) + 1);
            }
        }
        if (tree.lchild_bottom != -1) {
            if (tree.loutcolor_bottom == Color::Red) {
                assert(out_red == -1);
                out_red = tree.lchild_bottom;
                out_red_delay = lchild_bottom_delay;
                out_red_top_tree = 0;
                num_steps = std::max(num_steps, out_red_delay + 2 * (bottom_package_cnt - 1) + 1);
            } else {
                assert(out_black == -1);
                assert(tree.loutcolor_bottom == Color::Black);
                out_black = tree.lchild_bottom;
                out_black_delay = lchild_bottom_delay;
                out_black_top_tree = 0;
                num_steps = std::max(num_steps, out_black_delay + 2 * (bottom_package_cnt - 1) + 1);
            }
        }
        if (tree.rchild_bottom != -1) {
            if (tree.routcolor_bottom == Color::Red) {
                assert(out_red == -1);
                out_red = tree.rchild_bottom;
                out_red_delay = rchild_bottom_delay;
                out_red_top_tree = 0;
                num_steps = std::max(num_steps, out_red_delay + 2 * (bottom_package_cnt - 1) + 1);
            } else {
                assert(out_black == -1);
                assert(tree.routcolor_bottom == Color::Black);
                out_black = tree.rchild_bottom;
                out_black_delay = rchild_bottom_delay;
                out_black_top_tree = 0;
                num_steps = std::max(num_steps, out_black_delay + 2 * (bottom_package_cnt - 1) + 1);
            }
        }

        // Reduce
        for (int step = num_steps - 1; step >= 0; --step) {
            // Handle red edges
            if (step % 2 == 0) {
                ReduceSendRecv(step - in_red_delay + in_red_top_tree, in_red,
                               step - out_red_delay + out_red_top_tree, out_red);
            }

            // Handle black edges
            else {
                ReduceSendRecv(step - in_black_delay + in_black_top_tree, in_black,
                               step - out_black_delay + out_black_top_tree, out_black);
            }
        }

        // Bcast
        for (int step = 0; step < num_steps; ++step) {
            // Handle red edges
            if (step % 2 == 0) {
                BcastSendRecv(step - out_red_delay + out_red_top_tree, out_red,
                              step - in_red_delay + in_red_top_tree, in_red);
            }

            // Handle black edges
            else {
                BcastSendRecv(step - out_black_delay + out_black_top_tree, out_black,
                              step - in_black_delay + in_black_top_tree, in_black);
            }
        }
    }
private:

    ScanExecuter(std::unique_ptr<char[]> sendbuf, void *recvbuf, const int local_el_cnt,
                      const MPI_Datatype datatype, const MPI_Op op, RBC::Comm const &comm,
                      const int tag, const int datatype_byte_cnt, const int input_byte_cnt,
                      const int rank, const int nprocs,
                      const int max_package_el_cnt, const int package_cnt,
                      const int top_package_cnt, const int bottom_package_cnt,
                      std::unique_ptr<char[]> tmpbuf)
    : has_received_from_top_parent(false)
    , has_received_from_bottom_parent(false)
    , sendbuf(std::move(sendbuf))
    , recvbuf((char*)recvbuf)
    , local_el_cnt(local_el_cnt)
    , datatype(datatype)
    , op(op)
    , comm(comm)
    , tag(tag)
    , datatype_byte_cnt(datatype_byte_cnt)
    , input_byte_cnt(input_byte_cnt)
    , rank(rank)
    , nprocs(nprocs)
    , max_package_el_cnt(max_package_el_cnt)
    , package_cnt(package_cnt)
    , top_package_cnt(top_package_cnt)
    , bottom_package_cnt(bottom_package_cnt)
    , tmpbuf(std::move(tmpbuf))
    , tree(rank, nprocs) {
    }

    int ChildDelay(int parent, int delay_parent, int child, int incolor, int outcolor) {
        // We do not have that child.
        if (child == -1) {
            return -1;
        }

        // We are root node.
        if (parent == -1) {
            return outcolor;
        }

        // We are not a root node.
        else {
            return delay_parent + (incolor == outcolor ? 2 : 1);
        }
    }

    void ReduceSendRecv(int send_package_id, int target, int recv_package_id, int source) {
        MPI_Request requests[2];
        int is_sending = 0;
        int is_receiving = 0;
        bool is_receiving_from_left = 0;
        // Sending
        void* send_send_ptr;
        // Receiving
        void* recv_tmp_ptr;
        void* recv_recv_ptr;
        void* recv_send_ptr;
        int recv_el_cnt = 0;
        if (send_package_id >= 0 && send_package_id < package_cnt && target != -1) {
            is_sending = 1;
            int el_cnt = PackageElCnt(send_package_id);
            int send_offset = ElementOffset(send_package_id);
            send_send_ptr = SendbufPtr(send_offset);
            MPI_Isend(send_send_ptr, el_cnt, datatype, comm.RangeRankToMpiRank(target), tag, comm.mpi_comm, requests);
        }
        if (recv_package_id >= 0 && recv_package_id < package_cnt && source != -1) {
            is_receiving = 1;
            recv_el_cnt = PackageElCnt(recv_package_id);
            int recv_offset = ElementOffset(recv_package_id);
            recv_tmp_ptr = TmpbufPtr(recv_offset);
            recv_recv_ptr = RecvbufPtr(recv_offset);
            recv_send_ptr = SendbufPtr(recv_offset);
            is_receiving_from_left = source < rank;
            MPI_Irecv(recv_tmp_ptr, recv_el_cnt, datatype, comm.RangeRankToMpiRank(source), tag, comm.mpi_comm, requests + is_sending);
        }
        MPI_Waitall(is_sending + is_receiving, requests, MPI_STATUSES_IGNORE);
        if (is_receiving) {
            MPI_Reduce_local(recv_tmp_ptr, recv_send_ptr, recv_el_cnt, datatype, op);
            if (is_receiving_from_left) {
                MPI_Reduce_local(recv_tmp_ptr, recv_recv_ptr, recv_el_cnt, datatype, op);
            }
        }
    }

    void BcastSendRecv(int send_package_id, int target, int recv_package_id, int source) {
        MPI_Request requests[2];
        MPI_Status statuses[2];
        int is_sending = 0;
        int is_receiving = 0;
        bool receive_from_top = recv_package_id % 2 == 1;
        bool sending_to_top = send_package_id % 2 == 1;
        void* tmp_ptr;
        void* recv_recv_scan_ptr;
        int recv_el_cnt = 0;
        if (send_package_id >= 0 && send_package_id < package_cnt && target != -1) {
            is_sending = 1;
            int el_cnt = PackageElCnt(send_package_id);
            int send_offset = ElementOffset(send_package_id);
            const bool is_sending_to_left = target < rank;
            if (is_sending_to_left) {
                void* send_tmp_ptr = TmpbufPtr(send_offset);
                if (sending_to_top && !has_received_from_top_parent) {
                    el_cnt = 0;
                } else if (!sending_to_top && !has_received_from_bottom_parent) {
                    el_cnt = 0;
                }
                MPI_Isend(send_tmp_ptr, el_cnt, datatype, comm.RangeRankToMpiRank(target), tag, comm.mpi_comm, requests);
            } else {
                void* send_recv_ptr = RecvbufPtr(send_offset);
                MPI_Isend(send_recv_ptr, el_cnt, datatype, comm.RangeRankToMpiRank(target), tag, comm.mpi_comm, requests);
            }
        }
        if (recv_package_id >= 0 && recv_package_id < package_cnt && source != -1) {
            is_receiving = 1;
            recv_el_cnt = PackageElCnt(recv_package_id);
            int recv_offset = ElementOffset(recv_package_id);
            recv_recv_scan_ptr = RecvbufPtr(recv_offset);
            tmp_ptr = TmpbufPtr(recv_offset);
            MPI_Irecv(tmp_ptr, recv_el_cnt, datatype, comm.RangeRankToMpiRank(source), tag, comm.mpi_comm, requests + is_sending);
        }
        MPI_Waitall(is_sending + is_receiving, requests, statuses);
        if (is_receiving) {
            int count = 0;
            MPI_Get_count(statuses + is_sending, datatype, &count);
            if (count > 0) {
                MPI_Reduce_local(tmp_ptr, recv_recv_scan_ptr, recv_el_cnt, datatype, op);
                if (receive_from_top) {
                    has_received_from_top_parent = true;
                } else {
                    has_received_from_bottom_parent = true;
                }
            }
        }
    }

    void* SendbufPtr(size_t el_offset) {
        return sendbuf.get() + el_offset * datatype_byte_cnt;
    }

    void* TmpbufPtr(size_t el_offset) {
        return tmpbuf.get() + el_offset * datatype_byte_cnt;
    }

    void* RecvbufPtr(size_t el_offset) {
        return recvbuf + el_offset * datatype_byte_cnt;
    }

    int PackageElCnt(int package_id) {
        assert(package_id >= 0);
        assert(package_id < package_cnt);
        if (package_id == package_cnt - 1) {
            const int package_el_cnt = local_el_cnt - (package_cnt - 1) * max_package_el_cnt;
            assert(package_el_cnt <= max_package_el_cnt);
            return package_el_cnt;
        } else {
            return max_package_el_cnt;
        }
    }

    int ElementOffset(int package_id) {
        return package_id * max_package_el_cnt;
    }

    static int MaxPackageElCnt(int nprocs, int local_el_cnt, int bytes_per_el) {
        const double n = (double)local_el_cnt * bytes_per_el;
        const double d = std::ceil(std::log2((double)nprocs + 1.));
        const double k = 2. * std::sqrt(n * 2 * (d - 1.) * Network_Const::TB / Network_Const::TS);
        const int package_cnt = std::max<int>(1, k);
        return package_cnt;
    }

    bool has_received_from_top_parent;
    bool has_received_from_bottom_parent;

    std::unique_ptr<char[]> sendbuf;
    char* recvbuf;
    const int local_el_cnt;
    const MPI_Datatype datatype;
    const MPI_Op op;
    RBC::Comm const &comm;

    const int tag;
    const int datatype_byte_cnt;
    const int input_byte_cnt;
    const int rank;
    const int nprocs;

    const int max_package_el_cnt;
    const int package_cnt;
    const int top_package_cnt;
    const int bottom_package_cnt;

    std::unique_ptr<char[]> tmpbuf;

    void PrintArray(std::string name, const int* arr) {
        std::cout << "PE: " << rank << " " << name << " ";
        for (auto it = arr; it != arr + local_el_cnt; ++it) {
            std::cout << *it << " ";
        }
    }

    void PrintArrays() {
        PrintArray("sendbuf", (const int*)sendbuf.get());
        PrintArray("recvbuf", (const int*)recvbuf);
        PrintArray("tmpbuf", (const int*)tmpbuf.get());
    }
    
    const RBC::_internal::Twotree::Twotree tree;
};

int Scan(const void *sendbuf, void *recvbuf, int local_el_cnt, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm) {
    if (comm.useMPICollectives()) {
        return MPI_Scan(const_cast<void*> (sendbuf), recvbuf, local_el_cnt, datatype, op, comm.mpi_comm);
    }

    if (local_el_cnt == 0) {
        return 0;
    }

    int nprocs = 0;
    Comm_size(comm, &nprocs);
    if (nprocs <= 2) {
        RBC::Scan(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
    }

    auto executer = ScanExecuter::get(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
    executer.execute();
    return 0;
}

} // end namespace Twotree

} // end namespace _internal

int ScanTwotree(const void *sendbuf, void *recvbuf, int local_el_cnt, MPI_Datatype datatype, MPI_Op op, RBC::Comm const &comm) {
    return _internal::Twotree::Scan(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
}
} // end namespace RBC
