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

#pragma once

#include <cstddef>
#include <vector>

namespace RBC {
namespace _internal {
namespace Twotree {
enum Color { Red = 0, Black = 1 };

class Twotree {
 public:
  Twotree() { }
  Twotree(int pnr, int p);

  int m_incolor_top;
  int m_height_top;
  int m_parent_top;
  int m_delay_top;

  int m_lchild_top;
  int m_rchild_top;
  int m_loutcolor_top;
  int m_routcolor_top;

  int m_height_bottom;
  int m_parent_bottom;
  int m_delay_bottom;

  int m_lchild_bottom;
  int m_rchild_bottom;
  int m_loutcolor_bottom;
  int m_routcolor_bottom;

 private:
  static bool IsOdd(int i);

// We are not a leaf node in the top(!) tree.
  static bool IsNonLeaf(int pnr);

/* @brief Calculates parent of pnr of a tree with p PEs.
 *
 * p must be even. Parent of a root is -1. Just works for the top tree.
 * @height Returns height of node pnr.
 * @parentheight Returns height of parent.
 */
  int parent(int pnr, int p, int* height, int* parentheight);

/* @brief Same function as the implementation above.
 *
 * Implementation above calculates parent in O(log p).
 * This implementation calculates the parent in O(1). However, the caller must pass the height of then
 * current node.
 */
  int parent_fast(int pnr, int p, int* height_out, int height_in);

/* @brief Calculates child of pnr.
 *
 * Total number of PEs of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * Just works for the top tree.
 * @return Index of the leaf.
 */
  int LeftChild(int pnr);

/* @brief Calculates child of pnr.
 *
 * Total number of PEs p of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * Just works for the top tree.
 * @return Index of the leaf.
 */
  int RightChild(int pnr, int p);

  void Children(int pnr, int p, int* lchild_top, int* rchild_top, int* lchild_bottom, int* rchild_bottom);

  int Delay(const std::vector<int>& colors);

  int InEdgeColor(int p, int pnr, int parent_pnr, int parent_incolor);

/* @brief Calculates height, parent and colors of top-tree node.
 *
 * Just works for even p.
 */
  void InEdgeColorRec(int pnr, int p, int* height, int* parent_top, int height_in, std::vector<int>& colors);

/* @brief Calculates height, parent and colors of top-tree node and the bottom-tree node.
 *
 * Just works for even p.
 */
  void InEdgeColor(int pnr, int p,
                   int* height_top, int* parent_top, int* incolor_top, int* delay_top,
                   std::vector<int> ocolors, int* height_bottom, int* parent_bottom, int* delay_bottom,
                   std::vector<int> ucolors);

  void ChildrenColor(int pnr, int p,
                     int lchild_top, int rchild_top,
                     int incolor_top,
                     int lchild_bottom, int rchild_bottom,
                     const std::vector<int>& init_ocolors,
                     const std::vector<int>& init_ucolors,
                     int* loutcolor_top, int* routcolor_top,
                     int* loutcolor_bottom, int* routcolor_bottom);
};

double EstimatedTimeTwotree(int p, size_t bytes);
double EstimatedTimeBinomialtree(int p, size_t bytes);
int MaxPackageElCount(int nprocs, size_t bytes);
}  // end namespace Twotree
}  // end namespace _internal
}  // end namespace RBC
