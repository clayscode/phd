# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines a formatter for Python sources."""
import os
import reorder_python_imports
import sys

import black

from tools.format.formatters.base import batched_file_formatter


class FormatPython(batched_file_formatter.BatchedFileFormatter):
  """Format Python sources."""

  def __init__(self, *args, **kwargs):
    super(FormatPython, self).__init__(*args, **kwargs)

    # Reorder-python-imports provides unstable ordering of packages
    # when PYTHONPATH is set, but the bazel-generated py_binary target
    # uses PYTHONPATH to add required third party dependencies. The
    # compromise is to create a subset of the full PYTHONPATH
    # generated by bazel, keeping only those dependencies which are
    # required by //third_party/py/reorder_python_imports..
    self.reorder_python_imports_env = os.environ.copy()
    full_pythonpaths = self.reorder_python_imports_env["PYTHONPATH"].split(
      os.pathsep
    )
    essential_pythonpaths = [
      path
      for path in full_pythonpaths
      if "aspy_refactor_imports" in path or "cached_property" in path
    ]
    self.reorder_python_imports_env["PYTHONPATH"] = os.pathsep.join(
      essential_pythonpaths
    )

  def RunMany(self, paths):
    str_paths = [str(x) for x in paths]

    # Run black as a subprocess. Although it should be possible to use
    # click.testing.CliRunner() to rain black.main(), I found that this raised
    # errors with operations on closed I/O files.
    try:
      self._Exec(
        [
          sys.executable,
          black.__file__,
          "--line-length=80",
          "--target-version=py37",
        ]
        + str_paths
      )
    except self.FormatError as e:
      # black has a verbose error message style with the format:
      #
      #   error: <useful_info>
      #   Oh no! <emojis>
      #   <number_of_files_modified>
      #
      # We reshape the error message to discard those final two lines and the
      # "error: " prefix from the first line.
      new_message = "\n".join(str(e).split("\n")[:-3])[len("error: ") :]
      raise self.FormatError(new_message)

    self._Exec(
      [
        sys.executable,
        reorder_python_imports.__file__,
        "--exit-zero-even-if-changed",
        "--py3-plus",
      ]
      + str_paths,
      env=self.reorder_python_imports_env,
    )
