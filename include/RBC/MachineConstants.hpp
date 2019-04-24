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
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#pragma once

#ifndef RBC_ALPHA
// SuperMUC phase 1 constants as default
#define RBC_ALPHA 3.2e-06;
#endif

#ifndef RBC_BETA
// SuperMUC phase 1 constants as default
#define RBC_BETA 3.3e-10;
#endif

namespace RBC {
namespace _internal {
namespace optimized {
constexpr double kBETA = RBC_BETA;
constexpr double kALPHA = RBC_ALPHA;
constexpr double kALPHA_OVER_BETA = kALPHA / kBETA;
constexpr double kBETA_OVER_ALPHA = kBETA / kALPHA;
}   // namespace optimized
}  // end namespace _internal
}  // namespace RBC
