"""A collection of seven GPGPU benchmark suites.

These seven benchmark suites were responsible for 92% of GPU results published
in 25 top tier conference papers. For more details, see:

  ﻿Cummins, C., Petoumenos, P., Zang, W., & Leather, H. (2017). Synthesizing
  Benchmarks for Predictive Modeling. In CGO. IEEE.

When executed as a binary, this file runs the selected benchmark suites and
dumps the execution logs generated by libcecl in a directory.

Usage:

  bazel run //datasets/benchmarks/gpgpu -- --gpgpu_device_types=oclgrind
      --gpgpu_benchmarks_suites=npb-3.3 --gpgpu_logdir=/tmp/logs
"""
import contextlib
import functools
import multiprocessing
import networkx as nx
import os
import pathlib
import subprocess
import tempfile
import time
import typing

import humanize
from absl import app
from absl import flags
from absl import logging

from datasets.benchmarks.gpgpu import gpgpu_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import fs
from labm8 import labdate
from labm8 import labtypes
from labm8 import pbutil
from labm8 import system
from labm8 import text


FLAGS = flags.FLAGS

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

flags.DEFINE_list('gpgpu_benchmark_suites', _BENCHMARK_SUITE_NAMES,
                  'The names of benchmark suites to run. Defaults to all '
                  'benchmark suites.')
flags.DEFINE_list('gpgpu_device_types', ['oclgrind'],
                  'The device types to execute benchmark suites on. One or '
                  'more of {cpu,gpu,oclgrind}.')
flags.DEFINE_string('gpgpu_logdir', '/tmp/phd/datasets/benchmarks/gpgpu',
                    'The directory to write log files to.')
flags.DEFINE_integer('gpgpu_build_process_count', multiprocessing.cpu_count(),
                     'The number of parallel threads to use when building '
                     'GPGPU benchmark suites. Defaults to the number of '
                     'processors on your system.')

# The path of libcecl directory, containing the libcecl header, library, and
# run script.
_LIBCECL = bazelutil.DataPath('phd/gpu/libcecl/libcecl.so')
_LIBCECL_HEADER = bazelutil.DataPath('phd/gpu/libcecl/libcecl.h')

# Path to OpenCL headers and library.
_OPENCL_HEADERS_DIR = bazelutil.DataPath('opencl_120_headers')
if system.is_linux():
  _LIBOPENCL_DIR = bazelutil.DataPath('libopencl')

_DUMMY_BENCHMARK = bazelutil.DataPath(
    'phd/datasets/benchmarks/gpgpu/dummy_just_for_testing/dummy_benchmark')

_RODINIA_DATA_ROOT = bazelutil.DataPath('rodinia_data')


def CheckCall(command: typing.Union[str, typing.List[str]],
              shell: bool = False, env: typing.Dict[str, str] = None):
  """Wrapper around subprocess.check_call() to log executed commands."""
  if shell:
    logging.debug('$ %s', command)
    subprocess.check_call(command, shell=True, env=env)
  else:
    command = [str(x) for x in command]
    logging.debug('$ %s', ' '.join(command))
    subprocess.check_call(command, env=env)


def RewriteClDeviceType(device_type: str, path: pathlib.Path):
  """Rewrite all instances of CL_DEVICE_TYPE_XXX in the given path."""
  cl_device_type = {
    'cpu': 'CL_DEVICE_TYPE_CPU',
    'gpu': 'CL_DEVICE_TYPE_GPU',
    'oclgrind': 'CL_DEVICE_TYPE_CPU',
  }[device_type]
  CheckCall(f"""\
for f in $(find '{path}' -type f); do
  grep CL_DEVICE_TYPE_ $f &>/dev/null && {{
    sed -E -i 's/CL_DEVICE_TYPE_[A-Z]+/{cl_device_type}/g' $f
    echo Set {cl_device_type} in $f
  }} || true
done""", shell=True)


@functools.lru_cache(maxsize=1)
def OpenClCompileAndLinkFlags() -> typing.Tuple[str, str]:
  """Get device-specific OpenCL compile and link flags."""
  if system.is_linux():
    return (f'-isystem {_OPENCL_HEADERS_DIR}',
            f'-L{_LIBOPENCL_DIR} -Wl,-rpath,{_LIBOPENCL_DIR} -lOpenCL')
  else:
    return f'-isystem {_OPENCL_HEADERS_DIR}', '-framework OpenCL'


@contextlib.contextmanager
def MakeEnv(make_dir: pathlib.Path) -> typing.Dict[str, str]:
  """Return a build environment for GPGPU benchmarks."""
  cflags, ldflags = OpenClCompileAndLinkFlags()

  with fs.chdir(make_dir):
    with tempfile.TemporaryDirectory(prefix='phd_gpu_libcecl_header_') as d:
      d = pathlib.Path(d)
      fs.cp(_LIBCECL_HEADER, d / 'cecl.h')
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
      env['CFLAGS'] = f'-isystem {d} {cflags}'
      env['CXXFLAGS'] = f'-isystem {d} {cflags}'
      env['LDFLAGS'] = f'-lcecl -L{_LIBCECL.parent} {ldflags}'

      for flag in ['CFLAGS', 'CXXFLAGS', 'LDFLAGS']:
        env[f'EXTRA_{flag}'] = env[flag]
      yield env


def Make(target: typing.Optional[str], make_dir: pathlib.Path,
         extra_make_args: typing.Optional[typing.List[str]] = None,
         ) -> None:
  """Run make target in the given path."""
  if not (make_dir / 'Makefile').is_file():
    raise EnvironmentError(f"Cannot find Makefile in {make_dir}")

  # Build relative to the path, rather than using `make -c <path>`. This is
  # because some of the source codes have hard-coded relative paths.
  with MakeEnv(make_dir) as env:
    logging.debug('Running make %s in %s', target, make_dir)
    CheckCall(['make', '-j', FLAGS.gpgpu_build_process_count] +
              ([target] if target else []) + (extra_make_args or []), env=env)


def CMake(target: str, make_dir: pathlib.Path,
          extra_cflags: str = '') -> None:
  """Run make target in the given path."""
  if not (make_dir / 'CMakeLists.txt').is_file():
    raise EnvironmentError(f"Cannot find CMakeLists.txt in {make_dir}")

  # Build relative to the path, rather than using `make -c <path>`. This is
  # because some of the source codes have hard-coded relative paths.
  with MakeEnv(make_dir) as env:
    logging.debug('Running make %s in %s', target, make_dir)
    env['CFLAGS'] = f'{env["CFLAGS"]} {extra_cflags}'
    env['CXXFLAGS'] = f'{env["CXXFLAGS"]} {extra_cflags}'
    CheckCall(['cmake', '.'], env=env)
    CheckCall(['make', target, '-j', FLAGS.gpgpu_build_process_count,
               'VERBOSE=1'], env=env)


def FindExecutableInDir(path: pathlib.Path) -> pathlib.Path:
  """Find an executable file in a directory."""
  exes = [f for f in path.iterdir() if f.is_file() and os.access(f, os.X_OK)]
  if len(exes) != 1:
    raise EnvironmentError(f"Expected a single executable, found {len(exes)}")
  return exes[0]


@contextlib.contextmanager
def RunEnv(path: pathlib.Path) -> typing.Dict[str, str]:
  """Return an execution environment for a GPGPU benchmark."""
  with fs.chdir(path):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(_LIBCECL.parent)
    env['DYLD_LIBRARY_PATH'] = str(_LIBCECL.parent)
    yield env


def KernelInvocationsFromCeclLog(
    cecl_log: typing.List[str], device_type: str
) -> typing.List[gpgpu_pb2.OpenClKernelInvocation]:
  """Interpret and parse the output of a libcecl instrumented application.

  This is an updated and adapted implementation of
  kernel_invocations_from_cecl_log() from:
    //docs/2017_02_cgo/code/benchmarks:cecl2features
  """
  # Per-benchmark data transfer size and time.
  total_transferred_bytes = 0
  total_transfer_time = 0

  kernel_invocations = []

  expected_devtype = {
    'oclgrind': 'CPU'
  }.get(device_type, device_type.upper())

  # Iterate over each line in the cec log.
  logging.debug('Processing %d lines of libcecl logs', len(cecl_log))
  for line in cecl_log:
    # Split line based on ; delimiter into opcode and operands.
    components = [x.strip() for x in line.strip().split(';')]
    opcode, operands = components[0], components[1:]

    # Skip empty lines.
    if not opcode:
      continue

    if opcode == "clCreateCommandQueue":
      actual_devtype = {
        'UNKNOWN': expected_devtype,
      }.get(operands[0], operands[0])
      if expected_devtype != actual_devtype:
        raise ValueError(
            f"Expected device type {expected_devtype} does not match actual "
            f"device type {operands[0]}")
    elif opcode == "clEnqueueNDRangeKernel":
      kernel_name, global_size, local_size, elapsed = operands
      global_size = int(global_size)
      local_size = int(local_size)
      elapsed = float(elapsed)
      kernel_invocations.append(
          gpgpu_pb2.OpenClKernelInvocation(
              kernel_name=kernel_name,
              global_size=global_size,
              local_size=local_size,
              runtime_ms=elapsed))
      logging.debug('Extracted clEnqueueNDRangeKernel from log')
    elif opcode == "clEnqueueTask":
      kernel_name, elapsed = operands
      elapsed = float(elapsed)
      kernel_invocations.append(
          gpgpu_pb2.OpenClKernelInvocation(
              kernel_name=kernel_name,
              global_size=1, local_size=1,
              runtime_ms=elapsed))
      logging.debug('Extracted clEnqueueTask from log')
    elif opcode == "clCreateBuffer":
      size, _, flags = operands
      size = int(size)
      flags = flags.split("|")
      if "CL_MEM_COPY_HOST_PTR" in flags and "CL_MEM_READ_ONLY" not in flags:
        # Device <-> host.
        total_transferred_bytes += size * 2
      else:
        # Host -> Device, or Device -> host.
        total_transferred_bytes += size
      logging.debug('Extracted clCreateBuffer from log')
    elif (opcode == "clEnqueueReadBuffer" or
          opcode == "clEnqueueWriteBuffer" or
          opcode == "clEnqueueMapBuffer"):
      _, size, elapsed = operands
      elapsed = float(elapsed)
      total_transfer_time += elapsed
    else:
      # Not a line that we're interested in.
      pass

  # Defer transfer overhead until we have computed it.
  for ki in kernel_invocations:
    ki.transferred_bytes = total_transferred_bytes
    ki.runtime_ms += total_transfer_time

  return kernel_invocations


class _BenchmarkSuite(object):
  """Abstract base class for a GPGPU benchmark suite.

  A benchmark suite provides two methods: ForceDeviceType(), which forces all
  of the benchmarks within the suite to execute on a given device type (CPU or
  GPU), and Run(), which executes the benchmarks and logs output to a directory.
  Example usage:

    with SomeBenchmarkSuite() as bs:
      bs.ForceDeviceType('gpu')
      bs.Run('/tmp/logs/gpu')
      bs.ForceDeviceType('cpu')
      bs.Run('/tmp/logs/cpu/1')
      bs.Run('/tmp/logs/cpu/2')
  """

  def __init__(self):
    if self.name not in _BENCHMARK_SUITE_NAMES:
      raise ValueError(f"Unknown benchmark suite: {self.name}")

    self._device_type = None
    self._input_files = bazelutil.DataPath(
        f'phd/datasets/benchmarks/gpgpu/{self.name}')
    self._mutable_location = None
    self._logdir = None
    self._log_paths: typing.List[pathlib.Path] = []

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

  @property
  def logs(self) -> typing.Iterable[gpgpu_pb2.GpgpuBenchmarkRun]:
    """Return an iterator of log protos."""
    return (pbutil.FromFile(log, gpgpu_pb2.GpgpuBenchmarkRun())
            for log in self._log_paths)

  @property
  def log_count(self):
    return len(self._log_paths)

  def ForceDeviceType(self, device_type: str) -> None:
    """Force benchmarks to execute with the given device type."""
    if device_type not in {'cpu', 'gpu', 'oclgrind'}:
      raise ValueError(f"Unknown device type: {device_type}")
    self._device_type = device_type
    return self._ForceDeviceType(device_type)

  @property
  def device_type(self) -> str:
    return self._device_type

  def Run(self, logdir: pathlib.Path) -> None:
    """Run benchmarks and log results to directory."""
    logdir.mkdir(parents=True, exist_ok=True)
    if self.device_type is None:
      raise TypeError("Must call ForceDeviceType() before Run()")
    self._logdir = logdir
    ret = self._Run()
    self._logdir = None
    return ret

  def _ExecToLogFile(
      self, executable: pathlib.Path,
      benchmark_name: str,
      command: typing.Optional[typing.List[str]] = None,
      dataset_name: str = 'default',
      env: typing.Optional[typing.Dict[str, str]] = None
  ) -> None:
    """Run executable using runcecl script and log output."""
    logging.info('Executing benchmark %s', benchmark_name)
    self._logdir.mkdir(exist_ok=True, parents=True)

    # Create the name of the logfile now, so that is timestamped to the start of
    # execution.
    timestamp = labdate.MillisecondsTimestamp()
    log_name = '.'.join([
      self.name,
      benchmark_name,
      self.device_type,
      system.HOSTNAME,
      str(timestamp)
    ])

    # Assemble the command to run.
    command = command or [executable]
    if self.device_type == 'oclgrind':
      command = [str(oclgrind.OCLGRIND_PATH)] + command

    extra_env = env or dict()
    with RunEnv(executable.parent) as env:
      # Add the additional environment variables.
      env.update(extra_env)

      start_time = time.time()
      process = subprocess.Popen(
          command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          env=env, universal_newlines=True)
      stdout, stderr = process.communicate()
      elapsed = time.time() - start_time

      # Split libcecl logs out from stderr.
      cecl_lines, stderr_lines = [], []
      for line in stderr.split('\n'):
        if line.startswith('[CECL] '):
          stripped_line = line[len('[CECL] '):].strip()
          if stripped_line:
            cecl_lines.append(stripped_line)
        elif line.strip():
          stderr_lines.append(line.strip())

      if process.returncode:
        log_produced = self._logdir / f'{log_name}.ERROR.pbtxt'
      else:
        log_produced = self._logdir / f'{log_name}.pbtxt'

      pbutil.ToFile(gpgpu_pb2.GpgpuBenchmarkRun(
          ms_since_unix_epoch=timestamp,
          benchmark_suite=self.name,
          benchmark_name=benchmark_name,
          dataset_name=dataset_name,
          returncode=process.returncode,
          stdout=stdout,
          stderr='\n'.join(stderr_lines),
          cecl_log='\n'.join(cecl_lines),
          device_type=self.device_type,
          hostname=system.HOSTNAME,
          kernel_invocation=KernelInvocationsFromCeclLog(
              cecl_lines, self.device_type),
          elapsed_time_ms=int(elapsed * 1000),
      ), log_produced)

    logging.info('Wrote %s', log_produced)
    self._log_paths.append(log_produced)

  # Abstract attributes that must be provided by subclasses.

  @property
  def name(self) -> str:
    raise NotImplementedError("abstract property")

  @property
  def benchmarks(self) -> typing.List[str]:
    """Return a list of all benchmark names."""
    raise NotImplementedError("abstract property")

  def _ForceDeviceType(self, device_type: str) -> None:
    """Set the given device type."""
    raise NotImplementedError("abstract method")

  def _Run(self) -> None:
    """Run the benchmarks and produce output log files."""
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
    return ["dummy_benchmark"]

  def _ForceDeviceType(self, device_type: str):
    logging.info("Dummy benchmarks switching to %s", device_type)

  def _Run(self):
    logging.info("Executing dummy benchmarks!")
    self._ExecToLogFile(_DUMMY_BENCHMARK, 'dummy_benchmark')


class AmdAppSdkBenchmarkSuite(_BenchmarkSuite):
  """The AMD App SDK benchmarks."""

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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'samples/opencl/cl/1.x')

    for benchmark in self.benchmarks:
      # Clean any existing builds.
      if (self.path / f'samples/opencl/cl/1.x/{benchmark}/Makefile').is_file():
        Make('clean', self.path / 'samples/opencl/cl/1.x' / benchmark)

    # Delete all CMake generated files.
    CheckCall(['find', self.path / 'samples/opencl/cl/1.x', '-iwholename',
               '*cmake*', '-not', '-name', 'CMakeLists.txt', '-delete'])

    for benchmark in self.benchmarks:
      CMake('all', self.path / 'samples/opencl/cl/1.x' / benchmark,
            extra_cflags=' '.join([
              f'-isystem {self.path}/include',
              f'-include {self.path}/include/CL/cl_ext.h',
              f'-include {self.path}/include/CL/cl_gl.h'
            ]))

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

  def _ForceDeviceType(self, device_type: str):
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
            executable, f'{benchmark.lower()}.{dataset}',
            env={
              'OPENCL_DEVICE_TYPE': ('GPU' if self.device_type == 'gpu'
                                     else 'CPU')
            },
            dataset_name=dataset, command=[executable, f'../{benchmark}'])


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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'OpenCL/src')
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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'benchmarks')

    CheckCall(['find', self.path, '-name', '*.o', '-delete'])
    with MakeEnv(self.path) as env:
      for benchmark in self.benchmarks:
        CheckCall(['python2', './parboil', 'compile', benchmark,
                   'opencl_base'], env=env)

  def _Run(self):
    for benchmark, dataset in self.benchmarks_and_datasets:
      self._ExecToLogFile(
          self.path / 'parboil', f'{benchmark}.{dataset}',
          command=['python2', './parboil', 'run', benchmark, 'opencl_base',
                   dataset])


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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'OpenCL')
    for benchmark in self.benchmarks:
      logging.info('Building benchmark %s', benchmark)
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
      'hotspot3D',
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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'opencl')

    logging.info("Building Rodinia benchmarks")

    # This directory is not generated by the Makefile, but is needed by it.
    (self.path / 'bin/linux/opencl').mkdir(parents=True, exist_ok=True)

    # Copy and unpack the data sets, which come from a data-only file tree.
    logging.info('Unpacking compressed data archives.')
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

  def _ForceDeviceType(self, device_type: str):
    RewriteClDeviceType(device_type, self.path / 'src/opencl')
    if (self.path / 'Makefile').is_file():
      Make('distclean', self.path)

    with fs.chdir(self.path):
      CheckCall(['./configure'])

    Make('all', self.path)

  def _Run(self):
    CheckCall(f'find {self.path} -type f -executable | grep -v level0 | sort', shell=True)


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

  # Run the requested benchmark suites on the requested devices.
  outdir = pathlib.Path(FLAGS.gpgpu_logdir)
  for benchmark_suite_class in ResolveBenchmarkSuiteClassesFromNames(
        FLAGS.gpgpu_benchmark_suites):
    with benchmark_suite_class() as benchmark_suite:
      for device_type in FLAGS.gpgpu_device_types:
        logging.info('Building and running %s on %s', benchmark_suite.name,
                     device_type)
        benchmark_suite.ForceDeviceType(device_type)
        benchmark_suite.Run(outdir)

      kernel_invocation_count = sum(
          len(log.kernel_invocation) for log in benchmark_suite.logs)
      logging.info('Extracted %s kernel invocations from %s logs',
                   humanize.intcomma(kernel_invocation_count),
                   humanize.intcomma(benchmark_suite.log_count))


if __name__ == '__main__':
  app.run(main)
