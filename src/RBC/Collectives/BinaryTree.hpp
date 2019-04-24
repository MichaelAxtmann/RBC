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

namespace RBC {
namespace _internal {
namespace BinaryTree {
bool IsOdd(int i);

/* @brief Calculates parent of pnr of a tree with p PEs.
 *
 * p must be even. Parent of a root is -1. Just works for the top tree.
 */
int Parent(int pnr, int p);

/* @brief Calculates child of pnr.
 *
 * Total number of PEs of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * @return Index of the leaf.
 */
int LeftChild(int pnr);

/* @brief Calculates child of pnr.
 *
 * Total number of PEs p of the tree must be even.
 * Child of a leaf or children which does not exist is -1.
 * @return Index of the leaf.
 */
int RightChild(int pnr, int p);

void Children(int pnr, int p, int* lchild_top, int* rchild_top);

void Create(int myrank, int nprocs, int* lchild_top, int* rchild_top, int* parent);
}  // end namespace BinaryTree
}  // end namespace _internal
}  // end namespace RBC
