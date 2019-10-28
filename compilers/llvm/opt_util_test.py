# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //compilers/llvm/util.py."""
import typing

import pytest
from labm8 import app
from labm8 import test

from compilers.llvm import llvm
from compilers.llvm import opt
from compilers.llvm import opt_util

FLAGS = app.FLAGS

# Bytecode generated by clang using the following command:
# $ clang -emit-llvm -S -xc - < foo.c -o - > foo.ll
# Original C source code:
#
#     #include <stdio.h>
#     #include <math.h>
#
#     int DoSomething(int a, int b) {
#       if (a % 5) {
#         return a * 10;
#       }
#       return pow((float)a, 2.5);
#     }
#
#     int main(int argc, char **argv) {
#       for (int i = 0; i < argc; ++i) {
#         argc += DoSomething(argc, i);
#       }
#
#       printf("Computed value %d", argc);
#       return 0;
#     }
SIMPLE_C_BYTECODE = """
; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@.str = private unnamed_addr constant [18 x i8] c"Computed value %d\00", align 1

; Function Attrs: norecurse nounwind readnone ssp uwtable
define i32 @DoSomething(i32, i32) #0 {
  %3 = srem i32 %0, 5
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %7, label %5

; <label>:5                                       ; preds = %2
  %6 = mul nsw i32 %0, 10
  br label %12

; <label>:7                                       ; preds = %2
  %8 = sitofp i32 %0 to float
  %9 = fpext float %8 to double
  %10 = tail call double @llvm.pow.f64(double %9, double 2.500000e+00)
  %11 = fptosi double %10 to i32
  br label %12

; <label>:12                                      ; preds = %7, %5
  %13 = phi i32 [ %6, %5 ], [ %11, %7 ]
  ret i32 %13
}

; Function Attrs: nounwind readnone
declare double @llvm.pow.f64(double, double) #1

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32, i8** nocapture readnone) #2 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %7

; <label>:4                                       ; preds = %2
  br label %10

; <label>:5                                       ; preds = %22
  %6 = phi i32 [ %24, %22 ]
  br label %7

; <label>:7                                       ; preds = %5, %2
  %8 = phi i32 [ %0, %2 ], [ %6, %5 ]
  %9 = tail call i32 (i8*, ...) @printf(i8* nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str, i64 0, i64 0), i32 %8)
  ret i32 0

; <label>:10                                      ; preds = %4, %22
  %11 = phi i32 [ %25, %22 ], [ 0, %4 ]
  %12 = phi i32 [ %24, %22 ], [ %0, %4 ]
  %13 = srem i32 %12, 5
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %17, label %15

; <label>:15                                      ; preds = %10
  %16 = mul nsw i32 %12, 10
  br label %22

; <label>:17                                      ; preds = %10
  %18 = sitofp i32 %12 to float
  %19 = fpext float %18 to double
  %20 = tail call double @llvm.pow.f64(double %19, double 2.500000e+00) #4
  %21 = fptosi double %20 to i32
  br label %22

; <label>:22                                      ; preds = %15, %17
  %23 = phi i32 [ %16, %15 ], [ %21, %17 ]
  %24 = add nsw i32 %23, %12
  %25 = add nuw nsw i32 %11, 1
  %26 = icmp slt i32 %25, %24
  br i1 %26, label %10, label %5
}

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #3

attributes #0 = { norecurse nounwind readnone ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"Apple LLVM version 8.0.0 (clang-800.0.42.1)"}
"""

# LLVM-generated dot file for the DoSomething() function of a simple C program.
# Original C source code:
#
#     #include <stdio.h>
#     #include <math.h>
#
#     int DoSomething(int a, int b) {
#       if (a % 5) {
#         return a * 10;
#       }
#       return pow((float)a, 2.5);
#     }
#
#     int main(int argc, char **argv) {
#       for (int i = 0; i < argc; ++i) {
#         argc += DoSomething(argc, i);
#       }
#
#       printf("Computed value %d", argc);
#       return 0;
#     }
#
# I converted tabs to spaces in the following string.
SIMPLE_C_DOT = """
digraph "CFG for 'DoSomething' function" {
  label="CFG for 'DoSomething' function";

  Node0x7f86c670c590 [shape=record,label="{%2:\l  %3 = alloca i32, align 4\l  %4 = alloca i32, align 4\l  %5 = alloca i32, align 4\l  store i32 %0, i32* %4, align 4\l  store i32 %1, i32* %5, align 4\l  %6 = load i32, i32* %4, align 4\l  %7 = srem i32 %6, 5\l  %8 = icmp ne i32 %7, 0\l  br i1 %8, label %9, label %12\l|{<s0>T|<s1>F}}"];
  Node0x7f86c670c590:s0 -> Node0x7f86c65001a0;
  Node0x7f86c670c590:s1 -> Node0x7f86c65001f0;
  Node0x7f86c65001a0 [shape=record,label="{%9:\l\l  %10 = load i32, i32* %4, align 4\l  %11 = mul nsw i32 %10, 10\l  store i32 %11, i32* %3, align 4\l  br label %18\l}"];
  Node0x7f86c65001a0 -> Node0x7f86c65084b0;
  Node0x7f86c65001f0 [shape=record,label="{%12:\l\l  %13 = load i32, i32* %4, align 4\l  %14 = sitofp i32 %13 to float\l  %15 = fpext float %14 to double\l  %16 = call double @llvm.pow.f64(double %15, double 2.500000e+00)\l  %17 = fptosi double %16 to i32\l  store i32 %17, i32* %3, align 4\l  br label %18\l}"];
  Node0x7f86c65001f0 -> Node0x7f86c65084b0;
  Node0x7f86c65084b0 [shape=record,label="{%18:\l\l  %19 = load i32, i32* %3, align 4\l  ret i32 %19\l}"];
  }
"""


def test_DotCallGraphAndControlFlowGraphs_simple_c_program():
  cg, cfgs = opt_util.DotCallGraphAndControlFlowGraphsFromBytecode(
      SIMPLE_C_BYTECODE)
  assert len(cfgs) == 2
  assert cg
  assert 'Call graph' in cg
  assert "CFG for 'DoSomething' function" in '\n'.join(cfgs)
  assert "CFG for 'main' function" in '\n'.join(cfgs)


def test_DotCallGraphFromBytecode_simple_c_program():
  """Test that simple C program produces two Dot CFGs."""
  dot_cfgs = list(opt_util.DotControlFlowGraphsFromBytecode(SIMPLE_C_BYTECODE))
  assert len(dot_cfgs) == 2
  assert "CFG for 'DoSomething' function" in '\n'.join(dot_cfgs)
  assert "CFG for 'main' function" in '\n'.join(dot_cfgs)


def test_DotControlFlowGraphsFromBytecode_simple_c_program():
  """Test that simple C program produces two Dot CFGs."""
  dot_cfgs = list(opt_util.DotControlFlowGraphsFromBytecode(SIMPLE_C_BYTECODE))
  assert len(dot_cfgs) == 2
  assert "CFG for 'DoSomething' function" in '\n'.join(dot_cfgs)
  assert "CFG for 'main' function" in '\n'.join(dot_cfgs)


def test_DotControlFlowGraphsFromBytecode_invalid_bytecode():
  """Test that exception is raised if bytecode is invalid."""
  with pytest.raises(opt.OptException) as e_ctx:
    next(opt_util.DotControlFlowGraphsFromBytecode("invalid bytecode!"))
  assert e_ctx.value.returncode
  assert e_ctx.value.stderr


@pytest.mark.parametrize('cflags', [['-O0'], ['-O1'], ['-O2'], ['-O3']])
def test_GetOptArgs_black_box(cflags: typing.List[str]):
  """Black box opt args test."""
  args = opt_util.GetOptArgs(cflags)
  assert args
  for invocation in args:
    assert invocation


def test_GetOptArgs_bad_args():
  """Error is raised if invalid args are passed."""
  with pytest.raises(llvm.LlvmError):
    opt_util.GetOptArgs(['-not-a-real-arg!'])


def test_GetAliasSetsByFunction_no_alias_sets():
  """Sample bytecode contains no alias sets."""
  alias_sets = opt_util.GetAliasSetsByFunction(SIMPLE_C_BYTECODE)
  assert 'DoSomething' in alias_sets
  assert 'main' in alias_sets
  assert len(alias_sets) == 2
  assert alias_sets['DoSomething'] == []
  assert alias_sets['main'] == []


def test_GetAliasSetsByFunction_aliases():
  """Sample bytecode that contains alias sets."""
  alias_sets = opt_util.GetAliasSetsByFunction("""
%struct.foo = type { i32 }

define i32 @A() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca [2 x i8], align 1
  %4 = alloca [10 x i8], align 1
  %5 = alloca %struct.foo, align 4
  %6 = alloca i32*, align 8
  %7 = alloca %struct.foo*, align 8
  %8 = alloca i32*, align 8
  store i32 0, i32* %2, align 4
  br label %9

; <label>:9:                                      ; preds = %24, %0
  %10 = load i32, i32* %2, align 4
  %11 = icmp ne i32 %10, 10
  br i1 %11, label %12, label %27

; <label>:12:                                     ; preds = %9
  %13 = load i32, i32* %2, align 4
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds [10 x i8], [10 x i8]* %4, i64 0, i64 %14
  %16 = load i8, i8* %15, align 1
  %17 = getelementptr inbounds [2 x i8], [2 x i8]* %3, i64 0, i64 0
  store i8 %16, i8* %17, align 1
  %18 = load i32, i32* %2, align 4
  %19 = sub nsw i32 9, %18
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [10 x i8], [10 x i8]* %4, i64 0, i64 %20
  %22 = load i8, i8* %21, align 1
  %23 = getelementptr inbounds [2 x i8], [2 x i8]* %3, i64 0, i64 1
  store i8 %22, i8* %23, align 1
  br label %24

; <label>:24:                                     ; preds = %12
  %25 = load i32, i32* %2, align 4
  %26 = add nsw i32 %25, 1
  store i32 %26, i32* %2, align 4
  br label %9

; <label>:27:                                     ; preds = %9
  %28 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0
  store i32* %28, i32** %6, align 8
  store %struct.foo* %5, %struct.foo** %7, align 8
  store i32* null, i32** %8, align 8
  %29 = load i32, i32* %1, align 4
  ret i32 %29
}
""")
  assert 'A' in alias_sets
  assert len(alias_sets) == 1
  assert alias_sets['A'] == [
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod/Ref',
          pointers=[opt_util.Pointer(type='i32*', identifier='%2', size=4)]),
      opt_util.AliasSet(
          type='may alias',
          mod_ref='Ref',
          pointers=[
              opt_util.Pointer(type='i8*', identifier='%15', size=1),
              opt_util.Pointer(type='(i8*', identifier='%21', size=1)
          ]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod',
          pointers=[opt_util.Pointer(type='i8*', identifier='%17', size=1)]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod',
          pointers=[opt_util.Pointer(type='i8*', identifier='%23', size=1)]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod',
          pointers=[opt_util.Pointer(type='i32**', identifier='%6', size=8)]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod',
          pointers=[
              opt_util.Pointer(type='%struct.foo**', identifier='%7', size=8)
          ]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Mod',
          pointers=[opt_util.Pointer(type='i32**', identifier='%8', size=8)]),
      opt_util.AliasSet(
          type='must alias',
          mod_ref='Ref',
          pointers=[opt_util.Pointer(type='i32*', identifier='%1', size=4)])
  ]


def test_RunAnalysisPasses():
  analyses = list(
      opt_util.RunAnalysisPasses(
          """
%struct.foo = type { i32 }

define i32 @A() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca [2 x i8], align 1
  %4 = alloca [10 x i8], align 1
  %5 = alloca %struct.foo, align 4
  %6 = alloca i32*, align 8
  %7 = alloca %struct.foo*, align 8
  %8 = alloca i32*, align 8
  store i32 0, i32* %2, align 4
  br label %9

; <label>:9:                                      ; preds = %24, %0
  %10 = load i32, i32* %2, align 4
  %11 = icmp ne i32 %10, 10
  br i1 %11, label %12, label %27

; <label>:12:                                     ; preds = %9
  %13 = load i32, i32* %2, align 4
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds [10 x i8], [10 x i8]* %4, i64 0, i64 %14
  %16 = load i8, i8* %15, align 1
  %17 = getelementptr inbounds [2 x i8], [2 x i8]* %3, i64 0, i64 0
  store i8 %16, i8* %17, align 1
  %18 = load i32, i32* %2, align 4
  %19 = sub nsw i32 9, %18
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [10 x i8], [10 x i8]* %4, i64 0, i64 %20
  %22 = load i8, i8* %21, align 1
  %23 = getelementptr inbounds [2 x i8], [2 x i8]* %3, i64 0, i64 1
  store i8 %22, i8* %23, align 1
  br label %24

; <label>:24:                                     ; preds = %12
  %25 = load i32, i32* %2, align 4
  %26 = add nsw i32 %25, 1
  store i32 %26, i32* %2, align 4
  br label %9

; <label>:27:                                     ; preds = %9
  %28 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0
  store i32* %28, i32** %6, align 8
  store %struct.foo* %5, %struct.foo** %7, align 8
  store i32* null, i32** %8, align 8
  %29 = load i32, i32* %1, align 4
  ret i32 %29
}
""", ['-instcount', '-iv-users', '-loops']))
  analyses = sorted(analyses, key=lambda a: (a.analysis, a.function))

  assert len(analyses) == 2

  assert analyses[0].analysis == 'Counts the various types of Instructions'
  assert analyses[0].function == 'A'

  assert analyses[1].analysis == 'Induction Variable Users'
  assert analyses[1].lines


if __name__ == '__main__':
  test.Main()
