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

#include "BinaryTree.hpp"

namespace RBC {
namespace _internal {
namespace BinaryTree {
bool IsOdd(int i) {
  assert(i >= 0);
  return static_cast<unsigned int>(i) & 1;
}

/* @brief Calculates parent of pnr of a tree with p PEs.
 *
 * p must be even. Parent of a root is -1. Just works for the top tree.
 */
int Parent(int pnr, int p) {
  int i = 0;
  int parent_cnt;

  // First PE has index 1
  int pnrt = pnr + 1;

  // Calculate height (first one in bitvector)
  while ((pnrt & (1 << i)) == 0) i++;

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

  // First PE has index 0 again
  return parent_cnt - 1;
}

/* @brief Calculates child of pnr.
 *
 * Total number of PEs of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * @return Index of the leaf.
 */
int LeftChild(int pnr) {
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
 * @return Index of the leaf.
 */
int RightChild(int pnr, int p) {
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

void Children(int pnr, int p, int* lchild_top, int* rchild_top) {
  const int p_for_children = p - (IsOdd(p) ? 1 : 0);
  if (IsOdd(p) && pnr == p - 1) {
    int log_p = std::log2(p);
    *lchild_top = (1 << log_p) - 1;
    *rchild_top = -1;
  } else {
    *rchild_top = RightChild(pnr, p_for_children);
    *lchild_top = LeftChild(pnr);
  }
}

void Create(int myrank, int nprocs, int* lchild_top, int* rchild_top, int* parent) {
  // Calcuate children
  Children(myrank, nprocs, lchild_top, rchild_top);

  if (IsOdd(nprocs)) {
    if (myrank == nprocs - 1) {
      *parent = -1;
    } else {
      *parent = Parent(myrank, nprocs - 1);
      if (*parent == myrank) {
        *parent = nprocs - 1;
      }
    }
  } else {
    *parent = Parent(myrank, nprocs);
    if (*parent == myrank) {
      *parent = -1;
    }
  }
}
}  // end namespace BinaryTree
}  // end namespace _internal
}  // end namespace RBC
