/*****************************************************************************
 * This file is part of the Project RBC
 *
 * Copyright (c) 2018-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2018, Jochen Speck <speck@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

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
