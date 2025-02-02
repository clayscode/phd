"""A script which exports the subset of this repository required for target(s).

This project is getting large. This has two major downsides:
  * Fresh checkouts of the git repository take longer and consume more space.
  * The large number of packages is confusing to newcomers.

I feel like there's a 90-10 rule that applies to this repo: 90% of people who
checkout this repo only need 10% of the code contained within it.
This script provides a way to export that 10%.
"""
import pathlib
import sys
import tempfile
from typing import Dict
from typing import List

import git
import github as github_lib

from datasets.github import api
from labm8.py import app
from labm8.py.internal import workspace_status
from tools.source_tree import phd_workspace

FLAGS = app.FLAGS

app.DEFINE_list("targets", [], "The bazel target(s) to export.")
app.DEFINE_list(
  "excluded_targets", [], "A list of bazel targets to exclude from export."
)
app.DEFINE_list(
  "extra_files",
  [],
  "A list of additional files to export. Each element in "
  "the list is a relative path to export. E.g. `bar/baz.txt`.",
)
app.DEFINE_list(
  "move_file_mapping",
  [],
  "Each element in the list is a mapping of relative paths in the form "
  "<src>:<dst>. E.g. `foo.py:bar/baz.txt` will move file `foo.py` to "
  "destination `bar/baz.txt`.",
)
app.DEFINE_string("github_repo", None, "Name of a GitHub repo to export to.")
app.DEFINE_boolean(
  "github_create_repo",
  False,
  "Whether to create the repo if it does not exist.",
)
app.DEFINE_boolean(
  "github_repo_create_private",
  True,
  "Whether to create new GitHub repos as private.",
)
app.DEFINE_boolean(
  "export_source_tree_print_files",
  False,
  "Print the files that will be exported and terminate.",
)
app.DEFINE_boolean(
  "ignore_last_export",
  False,
  "If true, run through the entire git history. Otherwise, "
  "continue from the last commit exported. Use this flag if "
  "the set of exported files changes.",
)


def GetOrCreateRepoOrDie(
  github: github_lib.Github, repo_name: str
) -> github_lib.Repository:
  """Get the github repository to export to. Create it if it doesn't exist."""
  try:
    if FLAGS.github_create_repo:
      return api.GetOrCreateUserRepo(
        github,
        repo_name,
        description="PhD repo subtree export",
        homepage="https://github.com/ChrisCummins/phd",
        has_wiki=False,
        has_issues=False,
        private=FLAGS.github_repo_create_private,
      )
    else:
      return api.GetUserRepo(github, repo_name)
  except (api.RepoNotFoundError, OSError) as e:
    app.FatalWithoutStackTrace(str(e))


def Export(
  workspace_root: pathlib.Path,
  github_repo: str,
  targets: List[str],
  excluded_targets: List[str] = None,
  extra_files: List[str] = None,
  move_file_mapping: Dict[str, str] = None,
) -> None:
  """Custom entry-point to export source-tree.

  This should be called from a bare python script, before flags parsing.

  Args:
    workspace_root: The root path of the bazel workspace.
    github_repo: The name of the GitHub repo to export to.
    targets: A list of bazel targets to export. These targets, and their
      dependencies, will be exported. These arguments are passed unmodified to
      bazel query, so `/...` and `:all` labels are expanded, e.g.
      `//some/package/to/export/...`. All targets should be absolute, and
      prefixed with '//'.
    excluded_targets: A list of targets to exlude.
    extra_files: A list of additional files to export.
    move_file_mapping: A dictionary of <src,dst> relative paths listing files
      which should be moved from their respective source location to the
      destination.
  """
  excluded_targets = excluded_targets or []
  extra_files = extra_files or []
  move_file_mapping = move_file_mapping or {}

  source_workspace = phd_workspace.PhdWorkspace(workspace_root)

  with tempfile.TemporaryDirectory(prefix=f"phd_export_{github_repo}_") as d:
    destination = pathlib.Path(d)
    connection = api.GetDefaultGithubConnectionOrDie(
      extra_access_token_paths=[
        "~/.github/access_tokens/export_source_tree.txt"
      ]
    )
    repo = GetOrCreateRepoOrDie(connection, github_repo)
    api.CloneRepo(repo, destination)
    destination_repo = git.Repo(destination)

    src_files = source_workspace.GetAllSourceTreeFiles(
      targets, excluded_targets, extra_files, move_file_mapping
    )
    if FLAGS.export_source_tree_print_files:
      print("\n".join(str(x) for x in src_files))
      sys.exit(0)

    exported_commit_count = source_workspace.ExportToRepo(
      repo=destination_repo,
      targets=targets,
      src_files=src_files,
      extra_files=extra_files,
      file_move_mapping=move_file_mapping,
    )
    if not exported_commit_count:
      return
    app.Log(1, "Pushing changes to remote")

    # Force push since we may have rewritten history.
    destination_repo.git.push("origin", force=True)


def main():
  if not FLAGS.targets:
    raise app.UsageError("--targets must be one-or-more bazel targets")
  targets = list(sorted(set(FLAGS.targets)))

  def _GetFileMapping(f: str):
    if len(f.split(":")) == 2:
      return f.split(":")
    else:
      return f, f

  extra_files = list(sorted(set(FLAGS.extra_files)))

  move_file_tuples = [
    _GetFileMapping(f) for f in list(sorted(set(FLAGS.move_file_mapping)))
  ]
  move_file_mapping = {x[0]: x[1] for x in move_file_tuples}

  workspace_root = pathlib.Path(workspace_status.STABLE_UNSAFE_WORKSPACE)

  Export(
    workspace_root=workspace_root,
    github_repo=FLAGS.github_repo,
    targets=targets,
    excluded_targets=FLAGS.excluded_targets,
    extra_files=extra_files,
    move_file_mapping=move_file_mapping,
  )


if __name__ == "__main__":
  app.Run(main)
