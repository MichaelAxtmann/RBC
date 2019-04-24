/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2018-2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>

#include "RBC.hpp"
#include "Twotree.hpp"

namespace RBC {
namespace _internal {
namespace Twotree {
class AllreduceExecuter {
 public:
  AllreduceExecuter() = delete;

  static AllreduceExecuter get(const void* sendbuf, void* recvbuf,
                               int local_el_cnt, MPI_Datatype datatype, MPI_Op op, RBC::Comm const& comm) {
    const int tag = Tag_Const::ALLREDUCETWOTREE;
    int datatype_byte_cnt;
    size_t input_byte_cnt;
    int rank, nprocs;

    Comm_rank(comm, &rank);
    Comm_size(comm, &nprocs);

    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_byte_cnt = static_cast<int>(type_size);
    input_byte_cnt = static_cast<size_t>(local_el_cnt) * datatype_byte_cnt;

    std::unique_ptr<char[]> tmpbuf(new char[input_byte_cnt]);

    const int package_cnt = MaxPackageElCount(nprocs, input_byte_cnt);
    const int max_package_el_cnt = (local_el_cnt + package_cnt - 1) / package_cnt;

    const int bottom_package_cnt = (package_cnt + 1) / 2;
    const int top_package_cnt = package_cnt - bottom_package_cnt;

    return AllreduceExecuter(sendbuf, recvbuf, local_el_cnt, datatype, op, comm,
                             tag, datatype_byte_cnt, input_byte_cnt, rank, nprocs,
                             max_package_el_cnt, package_cnt, top_package_cnt, bottom_package_cnt,
                             std::move(tmpbuf));
  }

  int PackageCnt() const {
    return m_package_cnt;
  }

  void execute() {
    /*
     * Reduce:
     * Reduced result is stored in recvbuf.
     * We receive data in tmpbuf and merge recvbuf and tmpbuf into recvbuf.
     * We send data from recvbuf.
     *
     * Bcast:
     * Received result is stored in recvbuf.
     * We receive data in recvbuf.
     * We send data from recvbuf.
     *
     * Tree:
     * We send even packages over bottom tree.
     * We send odd packages over top tree.
     * We send red packages (0) in even steps.
     * We send black packages (1) in odd steps.
     */

    // Move local data to tmpbuf
    std::memcpy(m_recvbuf, m_sendbuf, m_input_byte_cnt);

    const int lchild_top_delay = ChildDelay(m_tree.m_parent_top,
                                            m_tree.m_delay_top,
                                            m_tree.m_lchild_top,
                                            m_tree.m_incolor_top,
                                            m_tree.m_loutcolor_top);
    const int rchild_top_delay = ChildDelay(m_tree.m_parent_top,
                                            m_tree.m_delay_top,
                                            m_tree.m_rchild_top,
                                            m_tree.m_incolor_top,
                                            m_tree.m_routcolor_top);
    const int lchild_bottom_delay = ChildDelay(m_tree.m_parent_bottom,
                                               m_tree.m_delay_bottom,
                                               m_tree.m_lchild_bottom,
                                               1 ^ m_tree.m_incolor_top,
                                               m_tree.m_loutcolor_bottom);
    const int rchild_bottom_delay = ChildDelay(m_tree.m_parent_bottom,
                                               m_tree.m_delay_bottom,
                                               m_tree.m_rchild_bottom,
                                               1 ^ m_tree.m_incolor_top,
                                               m_tree.m_routcolor_bottom);

    // Red in edge
    int in_red_delay = -1;
    int in_red = -1;
    int in_red_top_tree = 1;     // 0 for bottom tree, 1 fro top tree

    // Black in edge
    int in_black_delay = -1;
    int in_black = -1;
    int in_black_top_tree = 1;     // 0 for bottom tree, 1 fro top tree

    // Red out edge
    int out_red_delay = -1;
    int out_red = -1;
    int out_red_top_tree = 1;     // 0 for bottom tree, 1 fro top tree

    // Black out edge
    int out_black_delay = -1;
    int out_black = -1;
    int out_black_top_tree = 1;     // 0 for bottom tree, 1 fro top tree

    // Number of steps
    int num_steps = 0;

    if (m_tree.m_parent_top != -1) {
      if (m_tree.m_incolor_top == Color::Red) {
        assert(in_red == -1);
        in_red = m_tree.m_parent_top;
        in_red_delay = m_tree.m_delay_top;
        num_steps = std::max(num_steps, in_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(in_black == -1);
        assert(m_tree.m_incolor_top == Color::Black);
        in_black = m_tree.m_parent_top;
        in_black_delay = m_tree.m_delay_top;
        num_steps = std::max(num_steps, in_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_parent_bottom != -1) {
      if (m_tree.m_incolor_top == Color::Black) {
        assert(in_red == -1);
        in_red = m_tree.m_parent_bottom;
        in_red_delay = m_tree.m_delay_bottom;
        in_red_top_tree = 0;
        num_steps = std::max(num_steps, in_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(in_black == -1);
        assert(m_tree.m_incolor_top == Color::Red);
        in_black = m_tree.m_parent_bottom;
        in_black_delay = m_tree.m_delay_bottom;
        in_black_top_tree = 0;
        num_steps = std::max(num_steps, in_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }

    if (m_tree.m_lchild_top != -1) {
      if (m_tree.m_loutcolor_top == Color::Red) {
        assert(out_red == -1);
        out_red = m_tree.m_lchild_top;
        out_red_delay = lchild_top_delay;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_loutcolor_top == Color::Black);
        out_black = m_tree.m_lchild_top;
        out_black_delay = lchild_top_delay;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_rchild_top != -1) {
      if (m_tree.m_routcolor_top == Color::Red) {
        assert(out_red == -1);
        out_red = m_tree.m_rchild_top;
        out_red_delay = rchild_top_delay;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_routcolor_top == Color::Black);
        out_black = m_tree.m_rchild_top;
        out_black_delay = rchild_top_delay;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_lchild_bottom != -1) {
      if (m_tree.m_loutcolor_bottom == Color::Red) {
        assert(out_red == -1);
        out_red = m_tree.m_lchild_bottom;
        out_red_delay = lchild_bottom_delay;
        out_red_top_tree = 0;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_loutcolor_bottom == Color::Black);
        out_black = m_tree.m_lchild_bottom;
        out_black_delay = lchild_bottom_delay;
        out_black_top_tree = 0;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_rchild_bottom != -1) {
      if (m_tree.m_routcolor_bottom == Color::Red) {
        assert(out_red == -1);
        out_red = m_tree.m_rchild_bottom;
        out_red_delay = rchild_bottom_delay;
        out_red_top_tree = 0;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_routcolor_bottom == Color::Black);
        out_black = m_tree.m_rchild_bottom;
        out_black_delay = rchild_bottom_delay;
        out_black_top_tree = 0;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }

    // Reduce
    for (int step = num_steps - 1; step >= 0; --step) {
      if (step % 2 == 0) {
        // Handle red edges
        ReduceSendRecv(step - in_red_delay + in_red_top_tree, in_red,
                       step - out_red_delay + out_red_top_tree, out_red);
      } else {
        // Handle black edges
        ReduceSendRecv(step - in_black_delay + in_black_top_tree, in_black,
                       step - out_black_delay + out_black_top_tree, out_black);
      }
    }

    // Bcast
    for (int step = 0; step < num_steps; ++step) {
      if (step % 2 == 0) {
        // Handle red edges
        BcastSendRecv(step - out_red_delay + out_red_top_tree, out_red,
                      step - in_red_delay + in_red_top_tree, in_red);
      } else {
        // Handle black edges
        BcastSendRecv(step - out_black_delay + out_black_top_tree, out_black,
                      step - in_black_delay + in_black_top_tree, in_black);
      }
    }
  }

 private:
  AllreduceExecuter(const void* sendbuf, void* recvbuf, const int local_el_cnt,
                    const MPI_Datatype datatype, const MPI_Op op, RBC::Comm const& comm,
                    const int tag, const int datatype_byte_cnt, const int input_byte_cnt,
                    const int rank, const int nprocs,
                    const int max_package_el_cnt, const int package_cnt,
                    const int top_package_cnt, const int bottom_package_cnt,
                    std::unique_ptr<char[]> tmpbuf) :
    m_sendbuf(static_cast<const char*>(sendbuf)),
    m_recvbuf(static_cast<char*>(recvbuf)),
    m_local_el_cnt(local_el_cnt),
    m_datatype(datatype),
    m_op(op),
    m_comm(comm),
    m_tag(tag),
    m_datatype_byte_cnt(datatype_byte_cnt),
    m_input_byte_cnt(input_byte_cnt),
    m_rank(rank),
    m_nprocs(nprocs),
    m_max_package_el_cnt(max_package_el_cnt),
    m_package_cnt(package_cnt),
    m_top_package_cnt(top_package_cnt),
    m_bottom_package_cnt(bottom_package_cnt),
    m_tmpbuf(std::move(tmpbuf)),
    m_tree(rank, nprocs) { }

  int ChildDelay(int parent, int delay_parent, int child, int incolor, int outcolor) {
    // We do not have that child.
    if (child == -1) {
      return -1;
    }

    if (parent == -1) {
      // We are root node.
      return outcolor;
    } else {
      // We are not a root node.
      return delay_parent + (incolor == outcolor ? 2 : 1);
    }
  }

  void ReduceSendRecv(int send_package_id, int target, int recv_package_id, int source) {
    MPI_Request requests[2];
    int is_sending = 0;
    int is_receiving = 0;
    void* send_ptr;
    void* tmp_ptr;
    void* recv_ptr;
    int recv_el_cnt = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      int el_cnt = PackageElCnt(send_package_id);
      int send_offset = ElementOffset(send_package_id);
      send_ptr = RecvbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target), m_tag,
                m_comm.get(), requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      recv_el_cnt = PackageElCnt(recv_package_id);
      int recv_offset = ElementOffset(recv_package_id);
      tmp_ptr = TmpbufPtr(recv_offset);
      recv_ptr = RecvbufPtr(recv_offset);
      MPI_Irecv(tmp_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source), m_tag,
                m_comm.get(), requests + is_sending);
    }
    MPI_Waitall(is_sending + is_receiving, requests, MPI_STATUSES_IGNORE);
    if (is_receiving) {
      MPI_Reduce_local(tmp_ptr, recv_ptr, recv_el_cnt, m_datatype, m_op);
    }
  }

  void BcastSendRecv(int send_package_id, int target, int recv_package_id, int source) {
    MPI_Request requests[2];
    int is_sending = 0;
    int is_receiving = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      int el_cnt = PackageElCnt(send_package_id);
      int send_offset = ElementOffset(send_package_id);
      void* send_ptr = RecvbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                m_tag, m_comm.get(), requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      int recv_el_cnt = PackageElCnt(recv_package_id);
      int recv_offset = ElementOffset(recv_package_id);
      void* recv_ptr = RecvbufPtr(recv_offset);
      MPI_Irecv(recv_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source),
                m_tag, m_comm.get(), requests + is_sending);
    }
    MPI_Waitall(is_sending + is_receiving, requests, MPI_STATUSES_IGNORE);
  }

  void* TmpbufPtr(size_t el_offset) {
    return m_tmpbuf.get() + el_offset * m_datatype_byte_cnt;
  }

  void* RecvbufPtr(size_t el_offset) {
    return m_recvbuf + el_offset * m_datatype_byte_cnt;
  }

  int PackageElCnt(int package_id) const {
    assert(package_id >= 0);
    assert(package_id < m_package_cnt);
    if (package_id == m_package_cnt - 1) {
      const int package_el_cnt = m_local_el_cnt - (m_package_cnt - 1) * m_max_package_el_cnt;
      assert(package_el_cnt <= m_max_package_el_cnt);
      return package_el_cnt;
    } else {
      return m_max_package_el_cnt;
    }
  }

  int ElementOffset(int package_id) const {
    return package_id * m_max_package_el_cnt;
  }

  const char* m_sendbuf;
  char* m_recvbuf;
  const int m_local_el_cnt;
  const MPI_Datatype m_datatype;
  const MPI_Op m_op;
  RBC::Comm const& m_comm;

  const int m_tag;
  const int m_datatype_byte_cnt;
  const int m_input_byte_cnt;
  const int m_rank;
  const int m_nprocs;

  const int m_max_package_el_cnt;
  const int m_package_cnt;
  const int m_top_package_cnt;
  const int m_bottom_package_cnt;

  std::unique_ptr<char[]> m_tmpbuf;

  // void PrintArray(std::string name, const int* arr) {
  //     std::cout << "PE: " << m_rank << " " << name << " ";
  //     for (auto it = arr; it != arr + m_local_el_cnt; ++it) {
  //         std::cout << *it << " ";
  //     }
  // }

  // void PrintArrays() {
  //     PrintArray("sendbuf", static_cast<const int*>(m_sendbuf));
  //     PrintArray("recvbuf", static_cast<const int*>(m_recvbuf));
  //     PrintArray("tmpbuf", static_cast<const int*>(m_tmpbuf.get()));
  // }

  const RBC::_internal::Twotree::Twotree m_tree;
};

int Allreduce(const void* sendbuf, void* recvbuf, int local_el_cnt, MPI_Datatype datatype,
              MPI_Op op, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, local_el_cnt, datatype, op, comm.get());
  }

  if (local_el_cnt == 0) {
    return 0;
  }

  int nprocs = 0;
  Comm_size(comm, &nprocs);
  if (nprocs <= 2) {
    RBC::Allreduce(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
    return 0;
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  size_t bytes = local_el_cnt * static_cast<size_t>(type_size);

  if (EstimatedTimeBinomialtree(nprocs, bytes) < EstimatedTimeTwotree(nprocs, bytes)) {
    RBC::Allreduce(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
    return 0;
  } else {
    auto executer = AllreduceExecuter::get(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
    executer.execute();
    return 0;
  }
}
}  // end namespace Twotree

namespace optimized {
int AllreduceTwotree(const void* sendbuf, void* recvbuf, int local_el_cnt,
                     MPI_Datatype datatype, MPI_Op op, RBC::Comm const& comm) {
  return _internal::Twotree::Allreduce(sendbuf, recvbuf, local_el_cnt, datatype, op, comm);
}
}  // namespace optimized
}  // end namespace _internal
}  // end namespace RBC
