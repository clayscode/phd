"""Pytest fixtures for //tools/git."""
import pathlib

import git

from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def repo_with_history(tempdir: pathlib.Path) -> git.Repo:
  """Test fixture that returns a git repo with history."""
  repo = git.Repo.init(tempdir)
  repo_dir = pathlib.Path(repo.working_tree_dir)

  # Create the first commit with three files.
  readme = fs.Write(repo_dir / "README.txt", "Hello, world!\n".encode("utf-8"))
  (repo_dir / "src").mkdir()
  main = fs.Write(
    repo_dir / "src" / "main.c", "int main() { return 5; }".encode("utf-8")
  )
  makefile = fs.Write(
    repo_dir / "src" / "Makefile", "# An empty makefile".encode("utf-8")
  )

  repo.index.add([str(readme), str(main), str(makefile)])
  repo.index.commit("First commit, add some files")

  # Change the source file and Makefile in the second commit.
  main = fs.Write(
    repo_dir / "src" / "main.c", "int main() { return 0; }".encode("utf-8")
  )
  makefile = fs.Write(
    repo_dir / "src" / "Makefile", "# A modified makefile".encode("utf-8")
  )
  repo.index.add([str(main), str(makefile)])
  repo.index.commit("Change return value of program")

  # Remove the empty Makefile in the third commit.
  (repo_dir / "src" / "Makefile").unlink()
  repo.index.remove([str(makefile)])
  repo.index.commit("Remove the makefile")
  yield repo


@test.Fixture(scope="function")
def empty_repo(tempdir2: pathlib.Path) -> git.Repo:
  """Test fixture that returns an empty git repo."""
  repo = git.Repo.init(tempdir2)
  yield repo


@test.Fixture(scope="function")
def empty_repo2(tempdir3: pathlib.Path) -> git.Repo:
  """Test fixture that returns an empty git repo."""
  repo = git.Repo.init(tempdir3)
  yield repo
