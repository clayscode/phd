# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //labm8:make."""

import pytest

from labm8 import app
from labm8 import fs
from labm8 import make
from labm8 import test

FLAGS = app.FLAGS


# make()
def test_make():
  ret, out, err = make.make(dir="labm8/data/test/makeproj")
  assert not ret
  assert out
  assert fs.isfile("labm8/data/test/makeproj/foo")
  assert fs.isfile("labm8/data/test/makeproj/foo.o")


def test_make_bad_target():
  with pytest.raises(make.NoTargetError):
    make.make(target="bad-target", dir="labm8/data/test/makeproj")


def test_make_bad_target():
  with pytest.raises(make.NoMakefileError):
    make.make(dir="/bad/path")
  with pytest.raises(make.NoMakefileError):
    make.make(target="foo", dir="labm8/data/test")


def test_make_fail():
  with pytest.raises(make.MakeError):
    make.make(target="fail", dir="labm8/data/test/makeproj")


# clean()
def test_make_clean():
  fs.cd("labm8/data/test/makeproj")
  make.make()
  assert fs.isfile("foo")
  assert fs.isfile("foo.o")
  make.clean()
  assert not fs.isfile("foo")
  assert not fs.isfile("foo.o")
  fs.cdpop()


if __name__ == '__main__':
  test.Main()
