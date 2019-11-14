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
"""A collection of seven GPGPU benchmark suites.

These seven benchmark suites were responsible for 92% of GPU results published
in 25 top tier conference papers. For more details, see:

  ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
  Benchmarks for Predictive Modeling. In CGO. IEEE.

When executed as a binary, this file runs the selected benchmark suites and
dumps the execution logs generated by libcecl in a directory.

Usage:

  bazel run -c opt //datasets/benchmarks/gpgpu --
      --gpgpu_envs='Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2'
      --gpgpu_benchmarks_suites=amd,npb --gpgpu_logdir=/tmp/logs
"""
import multiprocessing
import os

import contextlib
import pathlib
import subprocess
import tempfile
import typing

from datasets.benchmarks.gpgpu import gpgpu_pb2
from gpu.cldrive.legacy import env as cldrive_env
from gpu.libcecl import libcecl_compile
from gpu.libcecl import libcecl_runtime
from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import labdate
from labm8 import labtypes
from labm8 import pbutil
from labm8 import system
from labm8 import text


FLAGS = app.FLAGS

# The list of all GPGPU benchmark suites.
_BENCHMARK_SUITE_NAMES = [
    'amd-app-sdk-3.0',
    'npb-3.3',
    'nvidia-4.2',
    'parboil-0.2',
    'polybench-gpu-1.0',
    'rodinia-3.1',
    'shoc-1.1.5',
    'dummy_just_for_testing',
]

app.DEFINE_list(
    'gpgpu_benchmark_suites', _BENCHMARK_SUITE_NAMES,
    'The names of benchmark suites to run. Defaults to all '
    'benchmark suites.')
app.DEFINE_list(
    'gpgpu_envs', ['Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2'],
    'The OpenCL environment names to execute benchmark suites '
    'on. To list the available environments, run '
    '`bazel run //gpu/clinfo`.')
app.DEFINE_string('gpgpu_logdir', '/tmp/phd/datasets/benchmarks/gpgpu',
                  'The directory to write log files to.')
app.DEFINE_integer(
    'gpgpu_build_process_count', multiprocessing.cpu_count(),
    'The number of parallel threads to use when building '
    'GPGPU benchmark suites. Defaults to the number of '
    'processors on your system.')
app.DEFINE_integer('gpgpu_benchmark_run_count', 1,
                   'The number of times to execute each benchmark suite.')
app.DEFINE_string('gpgpu_log_extension', '.pb',
                  'The file extension for generated log files.')
app.DEFINE_boolean(
    'gpgpu_record_outputs', True,
    "Record each benchmark's stdout and stderr. This "
    "information is not needed to get performance data, and "
    "can be quite large.")
app.DEFINE_boolean('gpgpu_fail_on_error', False,
                   'If a benchmark exits with a nonzero return code, fail.')

_RODINIA_DATA_ROOT = bazelutil.DataPath('rodinia_data')

_MKCECL = bazelutil.DataPath('phd/gpu/libcecl/mkcecl')


class BenchmarkInterrupt(OSError):
  """Early exit from benchmarking signal."""
  pass


def CheckCall(command: typing.Union[str, typing.List[str]],
              shell: bool = False,
              env: typing.Dict[str, str] = None):
  """Wrapper around subprocess.check_call() to log executed commands."""
  if shell:
    app.Log(3, '$ %s', command)
    subprocess.check_call(command, shell=True, env=env)
  else:
    command = [str(x) for x in command]
    app.Log(3, '$ %s', ' '.join(command))
    subprocess.check_call(command, env=env)


def RewriteClDeviceType(env: cldrive_env.OclgrindOpenCLEnvironment,
                        path: pathlib.Path):
  """Rewrite all instances of CL_DEVICE_TYPE_XXX in the given path."""
  cl_device_type = ('CL_DEVICE_TYPE_GPU' if env.device_type.lower() == 'gpu'
                    else 'CL_DEVICE_TYPE_CPU')
  CheckCall(
      f"""\
for f in $(find '{path}' -type f); do
  grep CL_DEVICE_TYPE_ $f &>/dev/null && {{
    sed -E -i 's/CL_DEVICE_TYPE_[A-Z]+/{cl_device_type}/g' $f
    echo Set {cl_device_type} in $f
  }} || true
done""",
      shell=True)


class BenchmarkRunObserver(object):
  """A class which provides a callback for processing / storing benchmark logs.
  """

  def OnBenchmarkRun(self, log: gpgpu_pb2.GpgpuBenchmarkRun) -> bool:
    """Notification callback that a new benchmark has been run.

    Args:
      log: The benchmark run log.

    Returns:
      True if should caryy on benchmarking, else False. If False, benchmarking
      will terminate once all observers have been notified.
    """
    raise NotImplementedError


class DumpLogProtoToFileObserver(BenchmarkRunObserver):
  """A benchmark observer that writes the log proto to file."""

  def __init__(self, logdir: pathlib.Path, file_extension: str = '.pb'):
    self._logdir = logdir
    self._logdir.mkdir(exist_ok=True, parents=True)
    self._log_paths: typing.List[pathlib.Path] = []
    self._file_extension = file_extension

  def OnBenchmarkRun(self, log: gpgpu_pb2.GpgpuBenchmarkRun) -> bool:
    """New log callback."""
    log_name = '.'.join([
        log.benchmark_suite, log.benchmark_name, log.dataset_name,
        log.run.device.name, log.hostname,
        str(labdate.MillisecondsTimestamp())
    ])

    if log.run.returncode:
      log_path = self._logdir / f'{log_name}.ERROR{self._file_extension}'
    else:
      log_path = self._logdir / f'{log_name}{self._file_extension}'

    pbutil.ToFile(log, log_path)

    app.Log(1, 'Wrote %s', log_path)
    self._log_paths.append(log_path)
    return True

  @property
  def logs(self) -> typing.Iterable[gpgpu_pb2.GpgpuBenchmarkRun]:
    """Return an iterator of log protos."""
    return (pbutil.FromFile(log, gpgpu_pb2.GpgpuBenchmarkRun())
            for log in self._log_paths)

  @property
  def log_count(self) -> int:
    return len(self._log_paths)


class FailOnErrorObserver(BenchmarkRunObserver):
  """A benchmark observer that exits on error."""

  def OnBenchmarkRun(self, log: gpgpu_pb2.GpgpuBenchmarkRun) -> bool:
    """New log callback."""
    if log.run.returncode:
      app.Error('Benchmark failed with stderr:\n%s', log.run.stderr)
      return False
    return True


@contextlib.contextmanager
def MakeEnv(make_dir: pathlib.Path,
            opencl_headers: bool = True) -> typing.Dict[str, str]:
  """Return a build environment for GPGPU benchmarks."""
  with fs.chdir(make_dir):
    with tempfile.TemporaryDirectory(prefix='phd_gpu_libcecl_header_') as d:
      d = pathlib.Path(d)
      # Many of the benchmarks include Linux-dependent headers. Spoof them here
      # so that we can build.
      if system.is_mac():
        with open(d / 'malloc.h', 'w') as f:
          f.write('#include <stdlib.h>')
        (d / 'linux').mkdir()
        with open(d / 'linux/limits.h', 'w') as f:
          f.write("""
#ifndef _LINUX_LIMITS_H
#define _LINUX_LIMITS_H

#define NR_OPEN	        1024

#define NGROUPS_MAX    65536	/* supplemental group IDs are available */
#define ARG_MAX       131072	/* # bytes of args + environ for exec() */
#define LINK_MAX         127	/* # links a file may have */
#define MAX_CANON        255	/* size of the canonical input queue */
#define MAX_INPUT        255	/* size of the type-ahead buffer */
#define NAME_MAX         255	/* # chars in a file name */
#define PATH_MAX        4096	/* # chars in a path name including nul */
#define PIPE_BUF        4096	/* # bytes in atomic write to a pipe */
#define XATTR_NAME_MAX   255	/* # chars in an extended attribute name */
#define XATTR_SIZE_MAX 65536	/* size of an extended attribute value (64k) */
#define XATTR_LIST_MAX 65536	/* size of extended attribute namelist (64k) */

#define RTSIG_MAX	  32

#endif
""")

      env = os.environ.copy()
      cflags, ldflags = libcecl_compile.LibCeclCompileAndLinkFlags(
          opencl_headers=opencl_headers)
      env['CFLAGS'] = ' '.join(cflags) + f' -isystem {d}'
      env['CXXFLAGS'] = ' '.join(cflags) + f' -isystem {d}'
      env['LDFLAGS'] = ' '.join(ldflags)

      for flag in ['CFLAGS', 'CXXFLAGS', 'LDFLAGS']:
        env[f'EXTRA_{flag}'] = env[flag]
      yield env


def Make(
    target: typing.Optional[str],
    make_dir: pathlib.Path,
    extra_make_args: typing.Optional[typing.List[str]] = None,
) -> None:
  """Run make target in the given path."""
  if not (make_dir / 'Makefile').is_file():
    raise EnvironmentError(f"Cannot find Makefile in {make_dir}")

  # Build relative to the path, rather than using `make -c <path>`. This is
  # because some of the source codes have hard-coded relative paths.
  with MakeEnv(make_dir) as env:
    app.Log(2, 'Running make %s in %s', target, make_dir)
    CheckCall(
        ['make', '-j', FLAGS.gpgpu_build_process_count] +
        ([target] if target else []) + (extra_make_args or []),
        env=env)


def FindExecutableInDir(path: pathlib.Path) -> pathlib.Path:
  """Find an executable file in a directory."""
  exes = [f for f in path.iterdir() if f.is_file() and os.access(f, os.X_OK)]
  if len(exes) != 1:
    raise EnvironmentError(f"Expected a single executable, found {len(exes)}")
  return exes[0]


class _BenchmarkSuite(object):
  """Abstract base class for a GPGPU benchmark suite.

  A benchmark suite provides two methods: ForceOpenCLEnvironment(), which forces
  the benchmarks within the suite to execute on a given device, and Run(), which
  executes the benchmarks and produces run logs. Example usage:

    gpu_log_observers = [SomeObserver()]
    cpu_log_observers = [SomeObserver()]
    with SomeBenchmarkSuite() as bs:
      bs.ForceOpenCLEnvironment('gpu')
      bs.Run([gpu_log_observers])
      bs.ForceOpenCLEnvironment('cpu')
      bs.Run([cpu_log_observers])
      bs.Run([cpu_log_observers])
  """

  def __init__(self):
    if self.name not in _BENCHMARK_SUITE_NAMES:
      raise ValueError(f"Unknown benchmark suite: {self.name}")

    self._env = None
    self._input_files = bazelutil.DataPath(
        f'phd/datasets/benchmarks/gpgpu/{self.name}')
    self._mutable_location = None
    self._observers = None

  def __enter__(self) -> pathlib.Path:
    prefix = f'phd_datasets_benchmarks_gpgpu_{self.name}'
    self._mutable_location = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    fs.cp(self._input_files, self._mutable_location)
    return self

  def __exit__(self, *args):
    fs.rm(self._mutable_location)
    self._mutable_location = None

  @property
  def path(self):
    """Return the path of the mutable copy of the benchmark sources."""
    if self._mutable_location is None:
      raise TypeError("Must be used as a context manager")
    return self._mutable_location

  def ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment) -> None:
    """Force benchmarks to execute with the given environment."""
    self._env = env
    return self._ForceOpenCLEnvironment(env)

  @property
  def env(self) -> cldrive_env.OpenCLEnvironment:
    return self._env

  def Run(self, observers: typing.List[BenchmarkRunObserver]) -> None:
    """Run benchmarks and log results to directory."""
    if self.env is None:
      raise TypeError("Must call ForceOpenCLEnvironment() before Run()")
    self._observers = observers
    ret = self._Run()
    self._observers = None
    return ret

  @contextlib.contextmanager
  def RunEnv(self, path: pathlib.Path) -> typing.Dict[str, str]:
    """Return an execution environment for a GPGPU benchmark."""
    with fs.chdir(path):
      yield libcecl_runtime.RunEnv(self.env)

  def _ExecToLogFile(
      self,
      executable: pathlib.Path,
      benchmark_name: str,
      command: typing.Optional[typing.List[str]] = None,
      dataset_name: str = 'default',
      env: typing.Optional[typing.Dict[str, str]] = None) -> None:
    """Run executable using runcecl script and log output."""
    app.Log(1, 'Executing %s:%s', self.name, benchmark_name)
    assert self._observers

    # Assemble the command to run.
    command = command or [str(executable)]
    if self.env.name == oclgrind.CLINFO_DESCRIPTION.name:
      command = [str(oclgrind.OCLGRIND_PATH)] + command

    extra_env = env or dict()
    with self.RunEnv(executable.parent) as os_env:
      # Add the additional environment variables.
      os_env.update(extra_env)

      libcecl_log = libcecl_runtime.RunLibceclExecutable(
          command, self.env, os_env, record_outputs=FLAGS.gpgpu_record_outputs)

    log = gpgpu_pb2.GpgpuBenchmarkRun(
        benchmark_suite=self.name,
        benchmark_name=benchmark_name,
        dataset_name=dataset_name,
        hostname=system.HOSTNAME,
        run=libcecl_log,
    )

    should_continue = True
    for observer in self._observers:
      should_continue &= observer.OnBenchmarkRun(log)

    if not should_continue:
      app.Log(1, 'Stopping benchmarking on request of observer(s)')
      raise BenchmarkInterrupt

  # Abstract attributes that must be provided by subclasses.

  @property
  def name(self) -> str:
    raise NotImplementedError("abstract property")

  @property
  def benchmarks(self) -> typing.List[str]:
    """Return a list of all benchmark names."""
    raise NotImplementedError("abstract property")

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment) -> None:
    """Set the given device type."""
    raise NotImplementedError("abstract method")

  def _Run(self) -> None:
    """Run the benchmarks."""
    raise NotImplementedError("abstract method")


class DummyJustForTesting(_BenchmarkSuite):
  """A dummy benchmark suite for testing purposes.

  It sill behaves like a real benchmark suite, but without running any expensive
  binaries.
  """

  @property
  def name(self) -> str:
    return "dummy_just_for_testing"

  @property
  def benchmarks(self) -> typing.List[str]:
    return ['hello']

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    app.Log(1, "Dummy benchmark running on %s", env.name)

    CheckCall([_MKCECL, self.path / 'hello.cc'])

    with MakeEnv(self.path) as env:
      CheckCall(
          f'gcc {self.path}/hello.cc -o {self.path}/hello {env["CFLAGS"]} '
          f'{env["LDFLAGS"]}',
          shell=True)

  def _Run(self):
    app.Log(1, "Executing dummy benchmarks!")
    self._ExecToLogFile(self.path / 'hello', 'hello')


class AmdAppSdkBenchmarkSuite(_BenchmarkSuite):
  """The AMD App SDK benchmarks.

  This is a subset of the App SDK example programs. They use a CMake build
  system. One caveat is that we have to use the OpenCL headers provided as
  part of the package, rather than the phd versions. The Ubunutu libglew-dev
  package is a build requirement.
  """

  @property
  def name(self):
    return 'amd-app-sdk-3.0'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'AdvancedConvolution',
        'BinomialOption',
        'BitonicSort',
        'BlackScholes',
        'FastWalshTransform',
        'FloydWarshall',
        'Histogram',
        'MatrixMultiplication',
        'MatrixTranspose',
        'MonteCarloAsian',
        'NBody',
        'PrefixSum',
        'Reduction',
        'ScanLargeArrays',
        'SimpleConvolution',
        'SobelFilter',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'samples/opencl/cl/1.x')

    for benchmark in self.benchmarks:
      # Clean any existing builds.
      if (self.path / f'samples/opencl/cl/1.x/{benchmark}/Makefile').is_file():
        Make('clean', self.path / 'samples/opencl/cl/1.x' / benchmark)

    # Delete all CMake generated files.
    CheckCall([
        'find', self.path / 'samples/opencl/cl/1.x', '-iwholename', '*cmake*',
        '-not', '-name', 'CMakeLists.txt', '-delete'
    ])

    for benchmark in self.benchmarks:
      with MakeEnv(
          self.path / f'samples/opencl/cl/1.x/{benchmark}',
          opencl_headers=False) as env:
        env['CFLAGS'] = f'{env["CFLAGS"]} -isystem {self.path}/include'
        env['CXXFLAGS'] = f'{env["CXXFLAGS"]} -isystem {self.path}/include'

        app.Log(2, 'Building %s:%s in %s', self.name, benchmark)
        CheckCall(['cmake', '.'], env=env)
        CheckCall(['make', '-j', FLAGS.gpgpu_build_process_count, 'VERBOSE=1'],
                  env=env)

  def _Run(self):
    for benchmark in self.benchmarks:
      executable = (self.path / 'samples/opencl/cl/1.x' / benchmark /
                    'bin/x86_64/Release' / benchmark)
      self._ExecToLogFile(executable, benchmark)


class NasParallelBenchmarkSuite(_BenchmarkSuite):
  """The NAS benchmark suite."""

  @property
  def name(self):
    return 'npb-3.3'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'BT',
        'CG',
        'EP',
        'FT',
        'IS',
        'LU',
        'MG',
        'SP',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    Make('clean', self.path)
    Make('suite', self.path)

  def _Run(self):
    for benchmark in self.benchmarks:
      for dataset in ['S', 'W', 'A', 'B', 'C']:
        executable = self.path / f'bin/{benchmark.lower()}.{dataset}.x'
        # TODO(cec): Fix a handful of build errors which prevent all
        # executables from being compiled.
        if not executable.is_file():
          continue
        self._ExecToLogFile(
            executable,
            f'{benchmark.lower()}.{dataset}',
            env={
                'OPENCL_DEVICE_TYPE':
                ('GPU' if self.env.device_type.lower() == 'gpu' else 'CPU')
            },
            dataset_name=dataset,
            command=[executable, f'../{benchmark}'])


class NvidiaBenchmarkSuite(_BenchmarkSuite):
  """NVIDIA GPU SDK."""

  @property
  def name(self):
    return 'nvidia-4.2'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'BlackScholes',
        'ConvolutionSeparable',
        'DCT8x8',
        'DXTCompression',
        'DotProduct',
        'FDTD3d',
        'HiddenMarkovModel',
        'MatVecMul',
        'MatrixMul',
        'MersenneTwister',
        'RadixSort',
        'Reduction',
        'Scan',
        'VectorAdd',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'OpenCL/src')
    Make('clean', self.path / 'OpenCL')
    Make(None, self.path / 'OpenCL')

  def _Run(self):
    for benchmark in self.benchmarks:
      executable = self.path / f'OpenCL/bin/linux/release/ocl{benchmark}'
      self._ExecToLogFile(executable, benchmark)


class ParboilBenchmarkSuite(_BenchmarkSuite):
  """Parboil benchmark suite."""

  @property
  def name(self):
    return 'parboil-0.2'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'bfs',
        'cutcp',
        'histo',
        'lbm',
        'mri-gridding',
        'mri-q',
        'sad',
        'sgemm',
        'spmv',
        'stencil',
        'tpacf',
    ]

  @property
  def benchmarks_and_datasets(self):
    return [
        ('bfs', '1M'),
        ('bfs', 'NY'),
        ('bfs', 'SF'),
        ('bfs', 'UT'),
        ('cutcp', 'large'),
        ('cutcp', 'small'),
        ('histo', 'default'),
        ('histo', 'large'),
        ('lbm', 'long'),
        ('lbm', 'short'),
        ('mri-gridding', 'small'),
        ('mri-q', 'large'),
        ('mri-q', 'small'),
        ('sad', 'default'),
        ('sad', 'large'),
        ('sgemm', 'medium'),
        ('sgemm', 'small'),
        ('spmv', 'large'),
        ('spmv', 'medium'),
        ('spmv', 'small'),
        ('stencil', 'default'),
        ('stencil', 'small'),
        ('tpacf', 'large'),
        ('tpacf', 'medium'),
        ('tpacf', 'small'),
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'benchmarks')

    # Due to the large size of parboil benchmarks (> 900 MB uncompressed), we
    # ship compressed archives with per-benchmark datasets. These must be
    # decompressed. This must be done prior to building. Once decompressed,
    # we remove the compressed archives so that the unpacked archives are
    # re-used for the lifetime of this object.
    with fs.chdir(self.path / f'datasets'):
      for benchmark in self.benchmarks:
        dataset_archive = self.path / f'datasets/{benchmark}.tar.bz2'
        if dataset_archive.is_file():
          app.Log(1, 'Unpacking datasets for %s:%s', self.name, benchmark)
          CheckCall(['tar', 'xjvf', dataset_archive])
        elif pathlib.Path(f'{dataset_archive}.part1').is_file():
          app.Log(1, 'Unpacking datasets for %s:%s', self.name, benchmark)
          CheckCall(
              f'cat {dataset_archive}.part1 {dataset_archive}.part2 '
              f'> {dataset_archive}',
              shell=True)
          pathlib.Path(f'{dataset_archive}.part1').unlink()
          pathlib.Path(f'{dataset_archive}.part2').unlink()
          CheckCall(['tar', 'xjvf', dataset_archive])

        if dataset_archive.is_file():
          dataset_archive.unlink()

    CheckCall(['find', self.path, '-name', '*.o', '-delete'])
    with MakeEnv(self.path) as env:
      for benchmark in self.benchmarks:
        CheckCall(['python2', './parboil', 'compile', benchmark, 'opencl_base'],
                  env=env)

  def _Run(self):
    for benchmark, dataset in self.benchmarks_and_datasets:
      self._ExecToLogFile(
          self.path / 'parboil',
          f'{benchmark}.{dataset}',
          command=[
              'python2', './parboil', 'run', benchmark, 'opencl_base', dataset
          ])


class PolybenchGpuBenchmarkSuite(_BenchmarkSuite):
  """PolyBench/GPU 1.0 Benchmarks."""

  @property
  def name(self):
    return 'polybench-gpu-1.0'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        '2DCONV',
        '2MM',
        '3DCONV',
        '3MM',
        'ATAX',
        'BICG',
        'CORR',
        'COVAR',
        # Bad: 'FDTD-2D',
        'GEMM',
        'GESUMMV',
        'GRAMSCHM',
        'MVT',
        'SYR2K',
        'SYRK',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'OpenCL')
    for benchmark in self.benchmarks:
      app.Log(1, 'Building benchmark %s', benchmark)
      Make('clean', self.path / 'OpenCL' / benchmark)
      Make('all', self.path / 'OpenCL' / benchmark)

  def _Run(self):
    for benchmark in self.benchmarks:
      executable = FindExecutableInDir(self.path / 'OpenCL' / benchmark)
      self._ExecToLogFile(executable, benchmark)


class RodiniaBenchmarkSuite(_BenchmarkSuite):
  """Rodinia Benchmark Suite 3.1.

  The University of Virginia Rodinia Benchmark Suite is a collection of parallel
  programs which targets heterogeneous computing platforms with both multicore
  CPUs and GPUs.

  Copyright (c)2008-2011 University of Virginia.

  For further details, see:

      S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, Sang-Ha Lee and
      K. Skadron. "Rodinia: A Benchmark Suite for Heterogeneous Computing".
      IISWC, 2009.
  """

  @property
  def name(self) -> str:
    return 'rodinia-3.1'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'b_tree',
        'backprop',
        'bfs',
        'cfd',
        'dwt2d',
        'gaussian',
        'heartwall',
        'hotspot',
        # 'hotspot3D',  # TODO(cec): Fails to build.
        'hybridsort',
        'kmeans',
        'lavaMD',
        'leukocyte',
        'lud',
        'myocyte',
        'nn',
        'nw',
        'particlefilter',
        'pathfinder',
        'srad',
        'streamcluster',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'opencl')

    app.Log(1, "Building Rodinia benchmarks")

    # This directory is not generated by the Makefile, but is needed by it.
    (self.path / 'bin/linux/opencl').mkdir(parents=True, exist_ok=True)

    # Copy and unpack the data sets, which come from a data-only file tree.
    app.Log(1, 'Unpacking compressed data archives.')
    fs.cp(_RODINIA_DATA_ROOT, self.path / 'data')
    Make('all', self.path / 'data')

    Make('OCL_clean', self.path)
    Make('OPENCL', self.path)
    # TODO(cec): Original script then deleted the opencl/hotspot3D/3D file. Is
    # it not working?

  def _Run(self):
    for benchmark in self.benchmarks:
      executable = self.path / 'opencl' / benchmark / 'run'
      self._ExecToLogFile(executable, benchmark, command=['bash', './run'])


class ShocBenchmarkSuite(_BenchmarkSuite):
  """SHOC Benchmarks."""

  @property
  def name(self):
    return 'shoc-1.1.5'

  @property
  def benchmarks(self) -> typing.List[str]:
    return [
        'BFS',
        'FFT',
        'GEMM',
        'MD',
        'MD5Hash',
        'Reduction',
        'Scan',
        'Sort',
        'Spmv',
        'Stencil2D',
        'Triad',
        'S3D',
    ]

  def _ForceOpenCLEnvironment(self, env: cldrive_env.OpenCLEnvironment):
    RewriteClDeviceType(env, self.path / 'src/opencl')
    if (self.path / 'Makefile').is_file():
      Make('distclean', self.path)

    with fs.chdir(self.path):
      CheckCall(['./configure'])

    Make(None, self.path / 'src/common')
    Make(None, self.path / 'src/opencl')

  def _Run(self):
    for benchmark in self.benchmarks:
      level1 = self.path / f'src/opencl/level1/{benchmark.lower()}/{benchmark}'
      level2 = self.path / f'src/opencl/level2/{benchmark.lower()}/{benchmark}'
      if level1.is_file():
        executable = level1
      else:
        executable = level2
      self._ExecToLogFile(executable, benchmark)


# A map of benchmark suite names to classes.
BENCHMARK_SUITES = {
    bs().name: bs for bs in labtypes.AllSubclassesOfClass(_BenchmarkSuite)
}


def ResolveBenchmarkSuiteClassesFromNames(names: typing.List[str]):
  trie = text.BuildPrefixTree(_BENCHMARK_SUITE_NAMES)
  benchmark_suite_classes = []

  for name in names:
    try:
      options = list(text.AutoCompletePrefix(name, trie))
    except KeyError:
      raise app.UsageError(f"Unknown benchmark suite: '{name}'. "
                           f"Legal values: {BENCHMARK_SUITES.keys()}")

    if len(options) > 1:
      raise app.UsageError(f"Ambiguous benchmark suite: '{name}'. "
                           f"Candidates: {options}")

    benchmark_suite_classes.append(BENCHMARK_SUITES[options[0]])

  return benchmark_suite_classes


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  # Get the OpenCL environments.
  envs = [
      cldrive_env.OpenCLEnvironment.FromName(env) for env in FLAGS.gpgpu_envs
  ]

  # Create the observers that will process the results..
  observers = [
      DumpLogProtoToFileObserver(
          pathlib.Path(FLAGS.gpgpu_logdir), FLAGS.gpgpu_log_extension)
  ]

  if FLAGS.gpgpu_fail_on_error:
    observers.append(FailOnErrorObserver())

  for benchmark_suite_class in ResolveBenchmarkSuiteClassesFromNames(
      FLAGS.gpgpu_benchmark_suites):
    with benchmark_suite_class() as benchmark_suite:
      for env in envs:
        app.Log(1, 'Building and running %s on %s', benchmark_suite.name,
                env.name)
        benchmark_suite.ForceOpenCLEnvironment(env)
        for i in range(FLAGS.gpgpu_benchmark_run_count):
          app.Log(1, 'Starting run %d of %s', i + 1, benchmark_suite.name)
          benchmark_suite.Run(observers)


if __name__ == '__main__':
  app.RunWithArgs(main)
