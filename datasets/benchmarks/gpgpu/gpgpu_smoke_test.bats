#!/usr/bin/env bats
#
# Test that `gpgpu` runs the dummy benchmark suite without catching fire.
#
# Copyright 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
source labm8/sh/test.sh

@test "run gpgpu" {
  run datasets/benchmarks/gpgpu/gpgpu \
    --gpgpu_benchmark_suites=dummy_just_for_testing \
    --gpgpu_envs='Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' \
    --gpgpu_logdir="$BATS_TMPDIR"
  [ "$status" -eq 0 ]
}
