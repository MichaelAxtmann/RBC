/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2018-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2018, Jochen Speck <speck@kit.edu>
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

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "RBC.hpp"
#include "tlx/math.hpp"

#include "Twotree.hpp"

namespace RBC {
namespace _internal {
namespace Twotree {
bool Twotree::IsOdd(int i) {
  assert(i >= 0);
  return i & 1;
}

// We are not a leaf node in the top(!) tree.
bool Twotree::IsNonLeaf(int pnr) {
  return IsOdd(pnr);
}

/* @brief Calculates parent of pnr of a tree with p PEs.
 *
 * p must be even. Parent of a root is -1. Just works for the top tree.
 * @height Returns height of node pnr.
 * @parentheight Returns height of parent.
 */
int Twotree::parent(int pnr, int p, int* height, int* parentheight) {
  int i = 0;
  int parent_idx;
  int parent_cnt;

  // First PE has index 1
  int pnrt = pnr + 1;

  // Calculate height (first one in bitvector)
  while ((pnrt & (1 << i)) == 0) i++;
  *height = i;

  // Calculate naive parent
  if (pnrt & (1 << (i + 1))) {
    // Parent is to the left
    parent_cnt = pnrt - (1 << i);
  } else {
    // Parent is to the right
    parent_cnt = pnrt + (1 << i);
  }

  // If naive parent does not exist, we have to go to the parent of the parent.
  if (parent_cnt > p) {
    parent_cnt = parent_cnt - (1 << (i + 1));
    // Handle root case
    if (parent_cnt < 1) {
      parent_cnt = parent_cnt + (1 << i);
    }
  }

  // Calculate height of parent
  // i contains the current height -> Number of iterations is defined by the difference in height.
  while ((parent_cnt & (1 << i)) == 0) i++;
  *parentheight = i;

  // First PE has index 0 again
  parent_idx = parent_cnt - 1;

  return parent_idx;
}

/* @brief Same function as the implementation above.
 *
 * Implementation above calculates parent in O(log p).
 * This implementation calculates the parent in O(1). However, the caller must pass the height of then
 * current node.
 */
int Twotree::parent_fast(int pnr, int p, int* height_out, int height_in) {
  // i is height of pnr
  int i = height_in;
  int parent_idx;
  int parent_cnt;

  // First PE has index 1
  // leafs have odd numbers
  int pnrt = pnr + 1;

  // Calculate naive parent
  if (pnrt & (1 << (i + 1))) {
    // Parent is to the left
    parent_cnt = pnrt - (1 << i);
  } else {
    // Parent is to the right
    parent_cnt = pnrt + (1 << i);
  }

  // If naive parent does not exist, use parent of the parent
  if (parent_cnt > p) {
    parent_cnt = parent_cnt - (1 << (i + 1));
    // Root case
    if (parent_cnt < 1) {
      parent_cnt = parent_cnt + (1 << i);
    }
  }

  // Calculate height of parent
  // i contains the current height -> Number of iterations is defined by the difference in height.
  while ((parent_cnt & (1 << i)) == 0) i++;
  *height_out = i;

  // First PE has index 0 again
  parent_idx = parent_cnt - 1;

  return parent_idx;
}

/* @brief Calculates child of pnr.
 *
 * Total number of PEs of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * Just works for the top tree.
 * @return Index of the leaf.
 */
int Twotree::LeftChild(int pnr) {
  int i = 0;
  int lchild;

  // First PE has index 1
  // leafs have odd numbers
  int pnrt = pnr + 1;

  // Calculate own height (first one in bitvector)
  while ((pnrt & (1 << i)) == 0) i++;

  // Left child exists always
  if (i == 0) {
    // We are a leaf
    lchild = 0;
  } else {
    // We are an inner node
    lchild = pnrt - (1 << (i - 1));
  }

  // First PE has index 0 again
  lchild = lchild - 1;

  return lchild;
}

/* @brief Calculates child of pnr.
 *
 * Total number of PEs p of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * Just works for the top tree.
 * @return Index of the leaf.
 */
int Twotree::RightChild(int pnr, int p) {
  int i = 0;
  int rchild;

  // First PE has index 1
  // leafs have odd numbers
  int pnrt = pnr + 1;

  // Calculate own height (first one in bitvector)
  while ((pnrt & (1 << i)) == 0) i++;

  // Right child does not exist always. It may happen that this child is at the wrong place.
  if (i == 0) {
    // We are a leaf
    rchild = 0;
  } else {
    // We are an inner node
    rchild = pnrt + (1 << (i - 1));
    if (rchild > p) {
      // Child does not exist -> Search for parent of the parent
      int j = 2;
      while (((i - j) >= 0) && (rchild > p)) {
        rchild = pnrt + (1 << (i - j));
        j++;
        // Search until we detect a direct neighbor
      }
      if (rchild > p) {
        // Direct neighbor does not exist
        rchild = 0;
      }
    }
  }

  // First PE has index 0 again
  rchild = rchild - 1;

  return rchild;
}

void Twotree::Children(int pnr, int p, int* lchild_top, int* rchild_top,
                       int* lchild_bottom, int* rchild_bottom) {
  const int p_for_children = p - (IsOdd(p) ? 1 : 0);
  if (IsOdd(p) && pnr == p - 1) {
    int log_p = std::log2(p);
    *lchild_top = (1 << log_p) - 1;
    *rchild_top = -1;
    *lchild_bottom = p_for_children - (*lchild_top + 1);
    *rchild_bottom = -1;
  } else {
    *rchild_top = RightChild(pnr, p_for_children);
    *lchild_top = LeftChild(pnr);
    // Inverse bottom child
    *rchild_bottom = LeftChild(p_for_children - (pnr + 1));
    if (*rchild_bottom != -1) {
      *rchild_bottom = p_for_children - (*rchild_bottom + 1);
    }
    *lchild_bottom = RightChild(p_for_children - (pnr + 1), p_for_children);
    if (*lchild_bottom != -1) {
      *lchild_bottom = p_for_children - (*lchild_bottom + 1);
    }
  }
}

int Twotree::Delay(const std::vector<int>& colors) {
  // Even a root node has a incolor.
  assert(colors.size() > 0);
  if (colors.size() == 1) {
    // I'm a root node.
    return -1;
  }

  int delay = colors[1];
  for (size_t idx = 2; idx < colors.size(); ++idx) {
    if (colors[idx - 1] == colors[idx]) {
      delay += 2;
    } else {
      delay += 1;
    }
  }
  return delay;
}

int Twotree::InEdgeColor(int p, int pnr, int parent_pnr, int parent_incolor) {
  return ((p / 2) & 1) ^ (parent_pnr > pnr) ^ parent_incolor;
}

/* @brief Calculates height, parent and colors of top-tree node.
 *
 * Just works for even p.
 */
void Twotree::InEdgeColorRec(int pnr, int p, int* height, int* parent_top,
                             int height_in, std::vector<int>& colors) {
  int parent_pnr;
  int dummy = 1;
  int neue_height;   // Height of the parent

  // Calculate index of parent
  // The first time, we calculate height of parent in O(log p)
  if (height_in == -1) {
    parent_pnr = parent(pnr, p, height, &neue_height);
    *parent_top = parent_pnr;
  } else {
    // If we have calculated the height of the parent once, we calculate the new parent in O(1)
    parent_pnr = parent_fast(pnr, p, &neue_height, height_in);
  }

  // We are the root. The incolor of the root is the last element of 'colors'.
  if (parent_pnr == pnr) { } else {
    InEdgeColorRec(parent_pnr, p, &dummy, &dummy, neue_height, colors);
    colors.push_back(InEdgeColor(p, pnr, parent_pnr, colors.back()));
  }
}

/* @brief Calculates height, parent and colors of top-tree node and the bottom-tree node.
 *
 * Just works for even p.
 */
void Twotree::InEdgeColor(int pnr, int p, int* height_top, int* parent_top,
                          int* incolor_top, int* delay_top, std::vector<int> ocolors,
                          int* height_bottom, int* parent_bottom, int* delay_bottom,
                          std::vector<int> ucolors) {
  if (IsNonLeaf(pnr)) {
    // We are inner node in top tree.

    InEdgeColorRec(pnr, p, height_top, parent_top, -1, ocolors);

    // Invert PE index
    const int pnr_mirr = p - (pnr + 1);
    InEdgeColorRec(pnr_mirr, p, height_bottom, parent_bottom, -1, ucolors);
    // Invert PE index again
    *parent_bottom = p - (*parent_bottom + 1);

    // Use last color of top tree
    ucolors.back() = 1 ^ ocolors.back();
  } else {
    // We are inner node in bottom tree

    // Mirror PE index
    const int pnr_mirr = p - (pnr + 1);
    InEdgeColorRec(pnr_mirr, p, height_bottom, parent_bottom, -1, ucolors);
    // Mirror PE index again
    *parent_bottom = p - (*parent_bottom + 1);

    // We are leaf in top tree
    InEdgeColorRec(pnr, p, height_top, parent_top, -1, ocolors);

    // Use last color of bottom tree
    ocolors.back() = 1 ^ ucolors.back();
  }

  *incolor_top = ocolors.back();

  // Calcuate delays
  *delay_top = Delay(ocolors);
  *delay_bottom = Delay(ucolors);
}

void Twotree::ChildrenColor(int pnr, int p,
                            int lchild_top, int rchild_top,
                            int incolor_top,
                            int lchild_bottom, int rchild_bottom,
                            const std::vector<int>& init_ocolors,
                            const std::vector<int>& init_ucolors,
                            int* loutcolor_top, int* routcolor_top,
                            int* loutcolor_bottom, int* routcolor_bottom) {
  // Init values
  *loutcolor_top = -1;
  *routcolor_top = -1;
  *loutcolor_bottom = -1;
  *routcolor_bottom = -1;

  // Non-leaf node in top tree
  if (IsNonLeaf(pnr)) {
    // Left child exists
    if (lchild_top != -1) {
      if (IsNonLeaf(lchild_top)) {
        // Left child is not a leaf in the top tree.
        *loutcolor_top = InEdgeColor(p, lchild_top, pnr, incolor_top);
      } else {
        // Left child is a leaf in the top tree.
        int dummy = 0;
        InEdgeColor(lchild_top, p, &dummy, &dummy, loutcolor_top, &dummy,
                    init_ocolors, &dummy, &dummy, &dummy, init_ucolors);
      }
    }

    if (rchild_top != -1) {
      // Right child exists

      if (IsNonLeaf(rchild_top)) {
        // Right child is not a leaf in the top tree.
        *routcolor_top = InEdgeColor(p, rchild_top, pnr, incolor_top);
      } else {
        int dummy = 0;
        InEdgeColor(rchild_top, p, &dummy, &dummy, routcolor_top, &dummy,
                    init_ocolors, &dummy, &dummy, &dummy, init_ucolors);
      }
    }
  } else {
    if (lchild_bottom != -1) {
      // Non-leaf node in bottom tree

      const int pnr_mirr = p - (pnr + 1);
      if (!IsNonLeaf(lchild_bottom)) {
        // Left child is not a leaf in the bottom tree.
        const int lchild_bottom_mirr = p - (lchild_bottom + 1);
        *loutcolor_bottom = InEdgeColor(p, lchild_bottom_mirr, pnr_mirr, 1 ^ incolor_top);
      } else {
        // Left child is a leaf in the bottom tree.
        int dummy = 0;
        int child_incolor_top = -1;
        InEdgeColor(lchild_bottom, p, &dummy, &dummy, &child_incolor_top, &dummy,
                    init_ocolors, &dummy, &dummy, &dummy, init_ucolors);
        *loutcolor_bottom = 1 ^ child_incolor_top;
      }
    }

    if (rchild_bottom != -1) {
      const int pnr_mirr = p - (pnr + 1);
      if (!IsNonLeaf(rchild_bottom)) {
        // Right child is not a leaf in the bottom tree.
        const int rchild_bottom_mirr = p - (rchild_bottom + 1);
        auto a = InEdgeColor(p, rchild_bottom_mirr, pnr_mirr, 1 ^ incolor_top);
        *routcolor_bottom = a;
      } else {
        // Right child is a leaf in the bottom tree.
        int dummy = 0;
        int child_incolor_top = -1;
        InEdgeColor(rchild_bottom, p, &dummy, &dummy, &child_incolor_top, &dummy, init_ocolors, &dummy,
                    &dummy, &dummy, init_ucolors);
        *routcolor_bottom = 1 ^ child_incolor_top;
      }
    }
  }
}

Twotree::Twotree(int pnr, int p) {
  // Calcuate children
  Children(pnr, p, &m_lchild_top, &m_rchild_top, &m_lchild_bottom, &m_rchild_bottom);

  // Calculate colors and heights
  if (IsOdd(p)) {
    if (pnr == p - 1) {
      m_height_top = static_cast<int>(std::log2(p - 1)) + 1;
      m_height_bottom = static_cast<int>(std::log2(p - 1)) + 1;
      m_parent_top = -1;
      m_parent_bottom = -1;
      m_delay_top = -1;
      m_delay_bottom = -1;

      const std::vector<int> ocolors = { 0 };
      const std::vector<int> ucolors = { 1 };
      m_incolor_top = ocolors.back();

      m_loutcolor_top = 1;
      m_routcolor_top = -1;
      m_loutcolor_bottom = 0;
      m_routcolor_bottom = -1;
    } else {
      const std::vector<int> ocolors = { 0, 1 };
      const std::vector<int> ucolors = { 1, 0 };
      InEdgeColor(pnr, p - 1, &m_height_top, &m_parent_top, &m_incolor_top, &m_delay_top,
                  ocolors, &m_height_bottom, &m_parent_bottom, &m_delay_bottom, ucolors);
      if (m_parent_top == pnr) {
        m_parent_top = p - 1;
      }
      if (m_parent_bottom == pnr) {
        m_parent_bottom = p - 1;
      }

      // Calculate color of children
      ChildrenColor(pnr, p - 1, m_lchild_top, m_rchild_top, m_incolor_top, m_lchild_bottom,
                    m_rchild_bottom, ocolors, ucolors, &m_loutcolor_top, &m_routcolor_top,
                    &m_loutcolor_bottom, &m_routcolor_bottom);
    }
  } else {
    const std::vector<int> ocolors = { 1 };
    const std::vector<int> ucolors = { 0 };
    InEdgeColor(pnr, p, &m_height_top, &m_parent_top, &m_incolor_top, &m_delay_top, ocolors,
                &m_height_bottom, &m_parent_bottom, &m_delay_bottom, ucolors);
    if (m_parent_top == pnr) {
      m_parent_top = -1;
    }
    if (m_parent_bottom == pnr) {
      m_parent_bottom = -1;
    }

    // Calculate color of children
    ChildrenColor(pnr, p, m_lchild_top, m_rchild_top, m_incolor_top, m_lchild_bottom,
                  m_rchild_bottom, ocolors, ucolors, &m_loutcolor_top, &m_routcolor_top,
                  &m_loutcolor_bottom, &m_routcolor_bottom);
  }
}

double EstimatedTimeTwotree(int p, size_t bytes) {
  return bytes * optimized::kBETA
         + optimized::kALPHA* 2 * tlx::integer_log2_ceil(p)
         + std::sqrt(2 * bytes * tlx::integer_log2_ceil(p) * optimized::kALPHA* optimized::kBETA);
}

double EstimatedTimeBinomialtree(int p, size_t bytes) {
  return tlx::integer_log2_ceil(p)
         * (bytes * optimized::kBETA + optimized::kALPHA);
}
int MaxPackageElCount(int nprocs, size_t bytes) {
  const auto d = tlx::integer_log2_ceil(nprocs + 1);
  return std::ceil(std::sqrt(bytes * (2 * d - 1.) * optimized::kBETA_OVER_ALPHA));
}
}  // end namespace Twotree
}  // end namespace _internal
}  // end namespace RBC
