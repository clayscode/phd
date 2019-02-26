// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/array_kernel_arg_value.h"

#include "gpu/cldrive/kernel_arg_value.h"

#include "third_party/opencl/cl.hpp"

#include "phd/logging.h"
#include "phd/string.h"

namespace gpu {
namespace cldrive {

namespace {

// TODO(cldrive): Work in progress!

template <typename T>
bool Vec2Equality(const T& lhs, const T& rhs) {
  return true;
  // TODO: return (l[0] == r[0]) && (l[1] == r[1]);
}

}  // anonymous namespace

template <>
/*virtual*/ string ArrayKernelArgValue<cl_char2>::ToString() const {
  string s = "";
  return s;
}

template <>
/*virtual*/ bool ArrayKernelArgValue<cl_char2>::ElementEquality(
    const cl_char2& lhs, const cl_char2& rhs) const {
  return Vec2Equality(lhs, rhs);
}

}  // namespace cldrive
}  // namespace gpu
