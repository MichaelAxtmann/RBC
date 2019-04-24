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
class ScanAndBcastExecuter {
 public:
  ScanAndBcastExecuter() = delete;

  static ScanAndBcastExecuter get(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast,
                                  int local_el_cnt, MPI_Datatype datatype, MPI_Op op,
                                  RBC::Comm const& comm) {
    const int scan_tag = Tag_Const::SCANANDBCASTSCANTWOTREE;
    const int bcast_tag = Tag_Const::SCANANDBCASTBCASTTWOTREE;
    size_t datatype_byte_cnt;
    size_t input_byte_cnt;
    int rank, nprocs;

    Comm_rank(comm, &rank);
    Comm_size(comm, &nprocs);

    MPI_Aint lb, type_size;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    datatype_byte_cnt = static_cast<int>(type_size);
    input_byte_cnt = static_cast<size_t>(local_el_cnt) * datatype_byte_cnt;

    std::unique_ptr<char[]> mysendbuf(new char[input_byte_cnt]);
    std::unique_ptr<char[]> tmpbuf(new char[input_byte_cnt]);

    std::memcpy(mysendbuf.get(), sendbuf, input_byte_cnt);
    std::memcpy(recvbuf_scan, sendbuf, input_byte_cnt);

    const int package_cnt = MaxPackageElCount(nprocs, input_byte_cnt);
    const int max_package_el_cnt = (local_el_cnt + package_cnt - 1) / package_cnt;

    const int bottom_package_cnt = (package_cnt + 1) / 2;
    const int top_package_cnt = package_cnt - bottom_package_cnt;

    return ScanAndBcastExecuter(std::move(mysendbuf), recvbuf_scan, recvbuf_bcast,
                                local_el_cnt, datatype, op, comm,
                                scan_tag, bcast_tag, datatype_byte_cnt, input_byte_cnt, rank, nprocs,
                                max_package_el_cnt, package_cnt, top_package_cnt, bottom_package_cnt,
                                std::move(tmpbuf));
  }

  int PackageCnt() const {
    return m_package_cnt;
  }

  void execute() {
    /*
     * Input is stored in mysendbuf (copy of sendbuf), and recvbuf_scan (input buffer).
     * Additional buffer tmpbuf and recvbuf_bcast (input buffer).
     *
     * Reduce:
     * Send package from sendbuf to parent.
     * Receive package into tmpbuf.
     * Packages from left child are reduced with input elements into recvbuf_scan.
     * Packages from left child and right child are reduced with input elements into sendbuf.
     *
     * Invariant after reduce:
     * mysendbuf contains input elements.
     * tmpbuf contains elements from left child.
     * recvbuf_scan contains input elements combined with elements from left child.
     * sendbuf contains input elements combined with elements from left child and right child.
     * recvbuf_bcast is empty.
     *
     * Prepare bcast-stage:
     * Root copies sendbuf packages to recvbuf_bcast (own elements combined with child elements).
     *
     * Bcast:
     * - Receive scan:
     * Receive elements from parent into tmpbuf(=elements from left subtrees).
     * If we received elements from parent into tmpbuf (we have a subtree to our left)
     * we combine the received elements from tmpbuf with recvbuf_scan(=input elements, elements from left child) into recvbuf_scan(=input els, els from left child, els from left subtrees).
     * Afterwards, recvbuf_scan contains the scan output.
     * Otherwise, recvbuf_scan already contained the scan output.
     * - Receive bcast (allreduce operation):
     * Receive reduced elements from root into recvbuf_bcast.
     * - Send scan:
     * Send tmpbuf(=elements from left subtrees) to left child if we already received a package from our parent.
     * Otherwise, we are one of the leftmost children of the tree
     * and we did not receive any elements from our parent (as there are no subtrees to our left).
     * In this case, we send a message of size zero to our left child.
     * Send recvbuf_scan(=input_els, els from left child, els from left subtrees) to right child.
     * - Send bcast (allreduce operation):
     * Send recvbuf_bcast to both children.
     *
     * Output recvbuf_scan and recvbuf_bcast.
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
    int in_red_top_tree = 1;     // 0 for bottom tree, 1 from top tree

    // Black in edge
    int in_black_delay = -1;
    int in_black = -1;
    int in_black_top_tree = 1;     // 0 for bottom tree, 1 from top tree

    // Red out edge
    int out_red_delay = -1;
    int out_red = -1;
    int out_red_top_tree = 1;     // 0 for bottom tree, 1 from top tree

    // Black out edge
    int out_black_delay = -1;
    int out_black = -1;
    int out_black_top_tree = 1;     // 0 for bottom tree, 1 from top tree

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

    // Root node copies reduced packages to bcast result vector.
    if (m_tree.m_parent_bottom == -1) {
      for (int i = 0; i < m_package_cnt; i += 2) {
        const int el_cnt = PackageElCnt(i);
        const int offset = ElementOffset(i);
        void* from_ptr = SendbufPtr(offset);
        void* to_ptr = RecvbufBcastPtr(offset);
        std::memcpy(to_ptr, from_ptr, el_cnt * m_datatype_byte_cnt);
      }
    }
    if (m_tree.m_parent_top == -1) {
      for (int i = 1; i < m_package_cnt; i += 2) {
        const int el_cnt = PackageElCnt(i);
        const int offset = ElementOffset(i);
        void* from_ptr = SendbufPtr(offset);
        void* to_ptr = RecvbufBcastPtr(offset);
        std::memcpy(to_ptr, from_ptr, el_cnt * m_datatype_byte_cnt);
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
  ScanAndBcastExecuter(std::unique_ptr<char[]> sendbuf, void* recvbuf_scan,
                       void* recvbuf_bcast, const int local_el_cnt,
                       const MPI_Datatype datatype, const MPI_Op op, RBC::Comm const& comm,
                       const int scan_tag, const int bcast_tag, const int datatype_byte_cnt,
                       const size_t input_byte_cnt,
                       const int rank, const int nprocs,
                       const int max_package_el_cnt, const int package_cnt,
                       const int top_package_cnt, const int bottom_package_cnt,
                       std::unique_ptr<char[]> tmpbuf) :
    m_has_received_from_top_parent(false),
    m_has_received_from_bottom_parent(false),
    m_sendbuf(std::move(sendbuf)),
    m_recvbuf_scan(static_cast<char*>(recvbuf_scan)),
    m_recvbuf_bcast(static_cast<char*>(recvbuf_bcast)),
    m_local_el_cnt(local_el_cnt),
    m_datatype(datatype),
    m_op(op),
    m_comm(comm),
    m_scan_tag(scan_tag),
    m_bcast_tag(bcast_tag),
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
    RBC::Request requests[2];
    int is_sending = 0;
    int is_receiving = 0;
    bool is_receiving_from_left = 0;
    // Sending
    void* send_send_ptr;
    // Receiving
    void* recv_tmp_ptr;
    void* recv_recv_scan_ptr;
    void* recv_send_ptr;
    int recv_el_cnt = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      is_sending = 1;
      int el_cnt = PackageElCnt(send_package_id);
      int send_offset = ElementOffset(send_package_id);
      send_send_ptr = SendbufPtr(send_offset);
      RBC::Isend(send_send_ptr, el_cnt, m_datatype, target, m_scan_tag, m_comm, requests);
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      is_receiving = 1;
      recv_el_cnt = PackageElCnt(recv_package_id);
      int recv_offset = ElementOffset(recv_package_id);
      recv_tmp_ptr = TmpbufPtr(recv_offset);
      recv_recv_scan_ptr = RecvbufScanPtr(recv_offset);
      recv_send_ptr = SendbufPtr(recv_offset);
      is_receiving_from_left = source < m_rank;
      RBC::Irecv(recv_tmp_ptr, recv_el_cnt, m_datatype, source, m_scan_tag, m_comm, requests + is_sending);
    }
    RBC::Waitall(is_sending + is_receiving, requests, MPI_STATUSES_IGNORE);
    if (is_receiving) {
      MPI_Reduce_local(recv_tmp_ptr, recv_send_ptr, recv_el_cnt, m_datatype, m_op);
      if (is_receiving_from_left) {
        MPI_Reduce_local(recv_tmp_ptr, recv_recv_scan_ptr, recv_el_cnt, m_datatype, m_op);
      }
    }
  }

  void BcastSendRecv(int send_package_id, int target, int recv_package_id, int source) {
    MPI_Request requests[4];
    MPI_Status statuses[4];
    // int is_scan_sending = 0;
    size_t request_offset = 0;
    size_t scan_receive_offset = 0;
    bool is_scan_receiving = false;
    bool receive_from_top = recv_package_id % 2 == 1;
    bool sending_to_top = send_package_id % 2 == 1;
    void* tmp_ptr;
    void* recv_recv_scan_ptr;
    int recv_el_cnt = 0;
    if (send_package_id >= 0 && send_package_id < m_package_cnt && target != -1) {
      const int el_cnt = PackageElCnt(send_package_id);
      const int send_offset = ElementOffset(send_package_id);

      // Send reduced elements from root
      void* send_recv_bcast_ptr = RecvbufBcastPtr(send_offset);
      MPI_Isend(send_recv_bcast_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                m_bcast_tag, m_comm.get(), requests + request_offset);
      ++request_offset;

      // Send scan elements

      if (target < m_rank) {
        // Send elements to left child

        void* send_tmp_ptr = TmpbufPtr(send_offset);
        if (sending_to_top && !m_has_received_from_top_parent) {
          MPI_Isend(send_tmp_ptr, 0, m_datatype, m_comm.RangeRankToMpiRank(target), m_scan_tag,
                    m_comm.get(), requests + request_offset);
          ++request_offset;           // is_scan_sending = 1;
        } else if (!sending_to_top && !m_has_received_from_bottom_parent) {
          MPI_Isend(send_tmp_ptr, 0, m_datatype, m_comm.RangeRankToMpiRank(target), m_scan_tag,
                    m_comm.get(), requests + request_offset);
          ++request_offset;           // is_scan_sending = 1;
        } else {
          MPI_Isend(send_tmp_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target), m_scan_tag,
                    m_comm.get(), requests + request_offset);
          ++request_offset;           // is_scan_sending = 1;
        }
      } else {
        // Send elements to right child

        void* send_recv_ptr = RecvbufScanPtr(send_offset);
        MPI_Isend(send_recv_ptr, el_cnt, m_datatype, m_comm.RangeRankToMpiRank(target),
                  m_scan_tag, m_comm.get(), requests + request_offset);
        ++request_offset;         // is_scan_sending = 1;
      }
    }
    if (recv_package_id >= 0 && recv_package_id < m_package_cnt && source != -1) {
      recv_el_cnt = PackageElCnt(recv_package_id);
      const int recv_offset = ElementOffset(recv_package_id);

      // Receive reduced elements from root
      void* recv_recv_bcast_ptr = RecvbufBcastPtr(recv_offset);
      MPI_Irecv(recv_recv_bcast_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source),
                m_bcast_tag, m_comm.get(),
                requests + request_offset);
      ++request_offset;
      assert(recv_offset + recv_el_cnt <= m_local_el_cnt);
      assert(recv_offset + recv_el_cnt >= 0);

      // Receive scan elements
      recv_recv_scan_ptr = RecvbufScanPtr(recv_offset);
      tmp_ptr = TmpbufPtr(recv_offset);
      MPI_Irecv(tmp_ptr, recv_el_cnt, m_datatype, m_comm.RangeRankToMpiRank(source), m_scan_tag,
                m_comm.get(), requests + request_offset);
      scan_receive_offset = request_offset;
      is_scan_receiving = true;
      ++request_offset;       // is_scan_sending = 1;
    }

    MPI_Waitall(request_offset, requests, statuses);
    if (is_scan_receiving) {
      int count = 0;
      MPI_Get_count(statuses + scan_receive_offset, m_datatype, &count);
      if (count > 0) {
        MPI_Reduce_local(tmp_ptr, recv_recv_scan_ptr, recv_el_cnt, m_datatype, m_op);
        if (receive_from_top) {
          m_has_received_from_top_parent = true;
        } else {
          m_has_received_from_bottom_parent = true;
        }
      }
    }
  }

  void* SendbufPtr(size_t el_offset) {
    return m_sendbuf.get() + el_offset * m_datatype_byte_cnt;
  }

  void* TmpbufPtr(size_t el_offset) {
    return m_tmpbuf.get() + el_offset * m_datatype_byte_cnt;
  }

  void* RecvbufScanPtr(size_t el_offset) {
    return m_recvbuf_scan + el_offset * m_datatype_byte_cnt;
  }

  void* RecvbufBcastPtr(size_t el_offset) {
    return m_recvbuf_bcast + el_offset * m_datatype_byte_cnt;
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

  bool m_has_received_from_top_parent;
  bool m_has_received_from_bottom_parent;

  std::unique_ptr<char[]> m_sendbuf;
  char* m_recvbuf_scan;
  char* m_recvbuf_bcast;
  const int m_local_el_cnt;
  const MPI_Datatype m_datatype;
  const MPI_Op m_op;
  RBC::Comm const& m_comm;

  const int m_scan_tag;
  const int m_bcast_tag;
  const int m_datatype_byte_cnt;
  const size_t m_input_byte_cnt;
  const int m_rank;
  const int m_nprocs;

  const int m_max_package_el_cnt;
  const int m_package_cnt;
  const int m_top_package_cnt;
  const int m_bottom_package_cnt;

  std::unique_ptr<char[]> m_tmpbuf;

  // std::string GetArray(std::string name, const int* arr) {
  //     name += "\t";
  //     for (auto it = arr; it != arr + m_local_el_cnt; ++it) {
  //         name += std::to_string(*it) + " ";
  //     }
  //     return name;
  // }

  // void PrintArrays() {
  //     std::string str("PE: ");
  //     str += std::to_string(m_rank) + "\t";
  //     str += GetArray("sendbuf", (const int*)m_sendbuf.get());
  //     str += "\t";
  //     str += GetArray("recvbuf_scan", (const int*)m_recvbuf_scan);
  //     str +=  "\t";
  //     str += GetArray("recvbuf_bcast", (const int*)m_recvbuf_bcast);
  //     str += "\t";
  //     str += GetArray("tmpbuf", (const int*)m_tmpbuf.get());
  //     str += "\n";
  //     std::cout << str;
  // }

  const RBC::_internal::Twotree::Twotree m_tree;
};

int ScanAndBcast(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast, int local_el_cnt,
                 MPI_Datatype datatype, MPI_Op op, RBC::Comm const& comm) {
  if (comm.useMPICollectives()) {
    return RBC::ScanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, local_el_cnt, datatype, op, comm);
  }

  if (local_el_cnt == 0) {
    return 0;
  }

  int nprocs = 0;
  Comm_size(comm, &nprocs);
  if (nprocs <= 2) {
    RBC::ScanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, local_el_cnt, datatype, op, comm);
    return 0;
  }

  MPI_Aint lb, type_size;
  MPI_Type_get_extent(datatype, &lb, &type_size);
  size_t bytes = local_el_cnt * static_cast<size_t>(type_size);

  if (EstimatedTimeBinomialtree(nprocs, bytes) < EstimatedTimeTwotree(nprocs, bytes)) {
    RBC::ScanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast, local_el_cnt, datatype, op, comm);
    return 0;
  } else {
    auto executer = ScanAndBcastExecuter::get(sendbuf, recvbuf_scan, recvbuf_bcast, local_el_cnt,
                                              datatype, op, comm);
    executer.execute();
    return 0;
  }
}
}  // end namespace Twotree

namespace optimized {
int ScanAndBcastTwotree(const void* sendbuf, void* recvbuf_scan, void* recvbuf_bcast,
                        int local_el_cnt, MPI_Datatype datatype, MPI_Op op, RBC::Comm const& comm) {
  return _internal::Twotree::ScanAndBcast(sendbuf, recvbuf_scan, recvbuf_bcast,
                                          local_el_cnt, datatype, op, comm);
}
}  // namespace optimized
}  // end namespace _internal
}  // end namespace RBC
