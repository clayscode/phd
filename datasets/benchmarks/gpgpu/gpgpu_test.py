"""Unit tests for //datasets/benchmarks/gpgpu:gpgpu.py."""
import pathlib
import typing

import pytest
from absl import flags

from datasets.benchmarks.gpgpu import gpgpu
from labm8 import test


FLAGS = flags.FLAGS

BENCHMARK_SUITES_TO_TEST = [
  gpgpu.PolybenchGpuBenchmarkSuite
]


def test_RewriteClDeviceType_rewrites_file(tempdir: pathlib.Path):
  """Test that CL_DEVICE_TYPE is rewritten in file."""
  with open(tempdir / 'foo', 'w') as f:
    f.write("Hello world! The device type is: CL_DEVICE_TYPE_GPU.")
  gpgpu.RewriteClDeviceType('oclgrind', tempdir)
  with open(tempdir / 'foo') as f:
    assert f.read() == "Hello world! The device type is: CL_DEVICE_TYPE_CPU."


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_path_contains_files(benchmark_suite: typing.Callable):
  """Test that benchmark suite contains files."""
  with benchmark_suite() as bs:
    assert bs.path.is_dir()
    assert list(bs.path.iterdir())


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_invalid_path_access(benchmark_suite: typing.Callable):
  """Path cannot be accessed except when used as a context manager."""
  bs = benchmark_suite()
  with pytest.raises(TypeError):
    bs.path


@pytest.mark.parametrize('benchmark_suite', BENCHMARK_SUITES_TO_TEST)
def test_BenchmarkSuite_integration_test(benchmark_suite: typing.Callable,
                                         tempdir: pathlib.Path):
  """Test compilation and execution of benchmark suite using oclgrind."""
  with benchmark_suite() as bs:
    bs.ForceDeviceType('oclgrind')
    bs.Run(tempdir)
    logfiles = list(tempdir.listdir())

    for benchmark in bs.benchmarks:
      # Check that a file starting with the expected name was created.
      assert any([f.name.startswith(f'{benchmark}.oclgrind.')
                 for f in logfiles])

    assert len(logfiles) == len(bs.benchmarks)


if __name__ == '__main__':
  test.Main()
