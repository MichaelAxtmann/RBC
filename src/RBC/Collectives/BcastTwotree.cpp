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
#include "tlx/math.hpp"

#include "Twotree.hpp"

namespace RBC {
namespace _internal {
namespace Twotree {
class BcastExecuter {
 public:
  BcastExecuter() = delete;

  static BcastExecuter get(void* sendbuf, int local_el_cnt, MPI_Datatype datatype,
                           int root, RBC::Comm const& comm) {
    const int tag = Tag_Const::BCASTTWOTREE;
    const int root_tag = Tag_Const::BCASTROOTTWOTREE;
    int datatype_byte_cnt;
    size_t input_byte_cnt;
    const int nprocs = comm.getSize();
    int rank = comm.getRank();

    // The 'root_offset' is used to transform the rbc ranks to so
    // called rooted ranks. The transformation guarantees that the
    // PE with rbc rank 'root' will be the root of the top
    // tree. If 'nprocs' is odd, the last PE is the root of the
    // top tree. Otherwise, the root of the top tree is the PE
    // with rank 2^{\floor{\log_2(p)}} - 1
    const int top_root = nprocs % 2 == 0 ?
                         tlx::round_down_to_power_of_two(nprocs) - 1 :
                         nprocs - 1;
    // The root offset is a value between -'nprocs'/2 and 'nprocs'-1.
    const int root_offset = top_root - root;
    rank = (nprocs + rank + root_offset) % nprocs;

    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_byte_cnt = static_cast<int>(type_size);
    input_byte_cnt = static_cast<size_t>(local_el_cnt) * datatype_byte_cnt;

    const int package_cnt = MaxPackageElCount(nprocs, input_byte_cnt);
    const int max_package_el_cnt = (local_el_cnt + package_cnt - 1) / package_cnt;

    const int bottom_package_cnt = (package_cnt + 1) / 2;
    const int top_package_cnt = package_cnt - bottom_package_cnt;

    return BcastExecuter(sendbuf, local_el_cnt, datatype, root_offset, comm,
                         tag, root_tag, datatype_byte_cnt, input_byte_cnt, rank, nprocs,
                         max_package_el_cnt, package_cnt, top_package_cnt, bottom_package_cnt);
  }

  int PackageCnt() const {
    return m_package_cnt;
  }

  void execute() {
    /*
     * Bcast:
     * We send and receive data from sendbuf.
     *
     * Tree:
     * We send even packages over bottom tree.
     * We send odd packages over top tree.
     * We send red packages (0) in even steps.
     * We send black packages (1) in odd steps.
     */

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
        in_red = UnrootPeRank(m_tree.m_parent_top);
        in_red_delay = m_tree.m_delay_top;
        num_steps = std::max(num_steps, in_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(in_black == -1);
        assert(m_tree.m_incolor_top == Color::Black);
        in_black = UnrootPeRank(m_tree.m_parent_top);
        in_black_delay = m_tree.m_delay_top;
        num_steps = std::max(num_steps, in_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_parent_bottom != -1) {
      if (m_tree.m_incolor_top == Color::Black) {
        assert(in_red == -1);
        in_red = UnrootPeRank(m_tree.m_parent_bottom);
        in_red_delay = m_tree.m_delay_bottom;
        in_red_top_tree = 0;
        num_steps = std::max(num_steps, in_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(in_black == -1);
        assert(m_tree.m_incolor_top == Color::Red);
        in_black = UnrootPeRank(m_tree.m_parent_bottom);
        in_black_delay = m_tree.m_delay_bottom;
        in_black_top_tree = 0;
        num_steps = std::max(num_steps, in_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }

    if (m_tree.m_lchild_top != -1) {
      if (m_tree.m_loutcolor_top == Color::Red) {
        assert(out_red == -1);
        out_red = UnrootPeRank(m_tree.m_lchild_top);
        out_red_delay = lchild_top_delay;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_loutcolor_top == Color::Black);
        out_black = UnrootPeRank(m_tree.m_lchild_top);
        out_black_delay = lchild_top_delay;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_rchild_top != -1) {
      if (m_tree.m_routcolor_top == Color::Red) {
        assert(out_red == -1);
        out_red = UnrootPeRank(m_tree.m_rchild_top);
        out_red_delay = rchild_top_delay;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_top_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_routcolor_top == Color::Black);
        out_black = UnrootPeRank(m_tree.m_rchild_top);
        out_black_delay = rchild_top_delay;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_top_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_lchild_bottom != -1) {
      if (m_tree.m_loutcolor_bottom == Color::Red) {
        assert(out_red == -1);
        out_red = UnrootPeRank(m_tree.m_lchild_bottom);
        out_red_delay = lchild_bottom_delay;
        out_red_top_tree = 0;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_loutcolor_bottom == Color::Black);
        out_black = UnrootPeRank(m_tree.m_lchild_bottom);
        out_black_delay = lchild_bottom_delay;
        out_black_top_tree = 0;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }
    if (m_tree.m_rchild_bottom != -1) {
      if (m_tree.m_routcolor_bottom == Color::Red) {
        assert(out_red == -1);
        out_red = UnrootPeRank(m_tree.m_rchild_bottom);
        out_red_delay = rchild_bottom_delay;
        out_red_top_tree = 0;
        num_steps = std::max(num_steps, out_red_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      } else {
        assert(out_black == -1);
        assert(m_tree.m_routcolor_bottom == Color::Black);
        out_black = UnrootPeRank(m_tree.m_rchild_bottom);
        out_black_delay = rchild_bottom_delay;
        out_black_top_tree = 0;
        num_steps = std::max(num_steps, out_black_delay + 2 * (m_bottom_package_cnt - 1) + 1);
      }
    }

    //
    if ((m_comm.getSize() % 2 == 0) && m_tree.m_parent_top == -1) {
      const int bottom_root = UnrootPeRank(m_comm.getSize() - m_rank - 1);

      // Bcast. The top root node sends bottom root node
      // packages in steps -1 ... 2 * 'bottom_package_cnt' - 2
      for (int step = -1; step < num_steps; ++step) {
        if ((step + 2) % 2 == 0) {
          // Handle red edges
          BcastSendRecv(step - out_red_delay + out_red_top_tree, out_red,
                        step - in_red_delay + in_red_top_tree, in_red);
        } else {
          // Handle black edges
          BcastSendRecvTopRoot(step - out_black_delay + out_black_top_tree, out_black,
                               step + 1, bottom_root,
                               step - in_black_delay + in_black_top_tree, in_black);
        }
      }
    } else if ((m_comm.getSize() % 2 == 0) && m_tree.m_parent_bottom == -1) {
      const int top_root = UnrootPeRank(m_comm.getSize() - m_rank - 1);

      // Bcast. The bottom root node receives top root node
      // packages in steps -1 ... 2 * 'bottom_package_cnt' - 2.
      for (int step = -1; step < num_steps; ++step) {
        if ((step + 2) % 2 == 0) {
          // Handle red edges
          BcastSendRecv(step - out_red_delay + out_red_top_tree, out_red,
                        step - in_red_delay + in_red_top_tree, in_red);
        } else {
          // Handle black edges
          BcastSendRecvBottomRoot(step - out_black_delay + out_black_top_tree, out_black,
                                  step - in_black_delay + in_black_top_tree, in_black,
                                  step + 1, top_root);
        }
      }
    } else {
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
  }

 private:
  BcastExecuter(void* sendbuf, const int local_el_cnt,
                const MPI_Datatype datatype, const int root_offset, RBC::Comm const& comm,
                const int tag, const int root_tag, const int datatype_byte_cnt, const int input_byte_cnt,
                const int rank, const int nprocs,
                const int max_package_el_cnt, const int package_cnt,
                const int top_package_cnt, const int bottom_package_cnt) :
    m_sendbuf(static_cast<char*>(sendbuf)),
    m_local_el_cnt(local_el_cnt),
    m_datatype(datatype),
    m_root_offset(root_offset),
    m_comm(comm),
    m_tag(tag),
    m_root_tag(root_tag),
    m_datatype_byte_cnt(datatype_byte_cnt),
    m_input_byte_cnt(input_byte_cnt),
    m_rank(rank),
    m_nprocs(nprocs),
    m_max_package_el_cnt(max_package_el_cnt),
    m_package_cnt(package_cnt),
    m_top_package_cnt(top_package_cnt),
    m_bottom_package_cnt(bottom_package_cnt),
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

  void BcastSendRecvTopRoot(int send_package_id, int target,
                            int parent_root_send_package_id, int parent_root,
                            int recv_package_id, int source) {
    MPI_Request requests[3];
    int is_sending = 0;
    int is_receiving = 0;
    int is_parent_root_sending = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      int el_cnt = PackageElCnt(send_package_id);
      int send_offset = ElementOffset(send_package_id);
      void* send_ptr = SendbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                m_tag, m_comm.get(), requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      const int recv_el_cnt = PackageElCnt(recv_package_id);
      const int recv_offset = ElementOffset(recv_package_id);
      const auto recv_ptr = SendbufPtr(recv_offset);
      MPI_Irecv(recv_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source),
                m_tag, m_comm.get(), requests + is_sending);
    }
    if (parent_root_send_package_id >= 0 &&
        parent_root_send_package_id < m_package_cnt &&
        parent_root != -1) {
      is_parent_root_sending = 1;
      const int el_cnt = PackageElCnt(parent_root_send_package_id);
      const int send_offset = ElementOffset(parent_root_send_package_id);
      const auto send_ptr = SendbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(parent_root),
                m_root_tag, m_comm.get(), requests + is_receiving + is_sending);
    }
    MPI_Waitall(is_sending + is_receiving + is_parent_root_sending, requests, MPI_STATUSES_IGNORE);
  }

  void BcastSendRecvBottomRoot(int send_package_id, int target,
                               int recv_package_id, int source,
                               int parent_root_recv_package_id, int parent_root) {
    MPI_Request requests[3];
    int is_sending = 0;
    int is_receiving = 0;
    int is_parent_root_receiving = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      const int el_cnt = PackageElCnt(send_package_id);
      const int send_offset = ElementOffset(send_package_id);
      const void* send_ptr = SendbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                m_tag, m_comm.get(), requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      const int recv_el_cnt = PackageElCnt(recv_package_id);
      const int recv_offset = ElementOffset(recv_package_id);
      auto recv_ptr = SendbufPtr(recv_offset);
      MPI_Irecv(recv_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source),
                m_tag, m_comm.get(), requests + is_sending);
    }
    if (parent_root_recv_package_id >= 0 &&
        parent_root_recv_package_id < m_package_cnt &&
        parent_root != -1) {
      is_parent_root_receiving = 1;
      const int recv_el_cnt = PackageElCnt(parent_root_recv_package_id);
      const int recv_offset = ElementOffset(parent_root_recv_package_id);
      auto recv_ptr = SendbufPtr(recv_offset);
      MPI_Irecv(recv_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(parent_root),
                m_root_tag, m_comm.get(), requests + is_sending + is_receiving);
    }
    MPI_Waitall(is_sending + is_receiving + is_parent_root_receiving, requests, MPI_STATUSES_IGNORE);
  }

  void BcastSendRecv(int send_package_id, int target, int recv_package_id, int source) {
    MPI_Request requests[2];
    int is_sending = 0;
    int is_receiving = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      int el_cnt = PackageElCnt(send_package_id);
      int send_offset = ElementOffset(send_package_id);
      void* send_ptr = SendbufPtr(send_offset);
      MPI_Isend(send_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                m_tag, m_comm.get(), requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      int recv_el_cnt = PackageElCnt(recv_package_id);
      int recv_offset = ElementOffset(recv_package_id);
      void* recv_ptr = SendbufPtr(recv_offset);
      MPI_Irecv(recv_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source),
                m_tag, m_comm.get(), requests + is_sending);
    }
    MPI_Waitall(is_sending + is_receiving, requests, MPI_STATUSES_IGNORE);
  }

  void* SendbufPtr(size_t el_offset) {
    return m_sendbuf + el_offset * m_datatype_byte_cnt;
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

  int UnrootPeRank(int rank) const {
    return (rank + m_comm.getSize() - m_root_offset) % m_comm.getSize();
  }

  char* m_sendbuf;
  const int m_local_el_cnt;
  const MPI_Datatype m_datatype;
  const int m_root_offset;
  RBC::Comm const& m_comm;

  const int m_tag;
  const int m_root_tag;
  const int m_datatype_byte_cnt;
  const int m_input_byte_cnt;
  const int m_rank;
  const int m_nprocs;

  const int m_max_package_el_cnt;
  const int m_package_cnt;
  const int m_top_package_cnt;
  const int m_bottom_package_cnt;

  const RBC::_internal::Twotree::Twotree m_tree;
};

int Bcast(void* sendbuf, int local_el_cnt, MPI_Datatype datatype, int root, Comm const& comm) {
  if (comm.useMPICollectives()) {
    return MPI_Bcast(sendbuf, local_el_cnt, datatype, root, comm.get());
  }

  if (local_el_cnt == 0) {
    return 0;
  }

  int nprocs = 0;
  Comm_size(comm, &nprocs);
  if (nprocs <= 2) {
    RBC::Bcast(sendbuf, local_el_cnt, datatype, root, comm);
    return 0;
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  size_t bytes = local_el_cnt * static_cast<size_t>(type_size);

  if (EstimatedTimeBinomialtree(nprocs, bytes) < EstimatedTimeTwotree(nprocs, bytes)) {
    RBC::Bcast(sendbuf, local_el_cnt, datatype, root, comm);
    return 0;
  } else {
    auto executer = BcastExecuter::get(sendbuf, local_el_cnt, datatype, root, comm);
    executer.execute();
    return 0;
  }
}
}  // end namespace Twotree

namespace optimized {
int BcastTwotree(void* sendbuf, int local_el_cnt, MPI_Datatype datatype, int root, RBC::Comm const& comm) {
  return _internal::Twotree::Bcast(sendbuf, local_el_cnt, datatype, root, comm);
}
}  // namespace optimized
}  // end namespace _internal
}  // end namespace RBC
