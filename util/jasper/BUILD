# Jasper is a small command line utility to run long-running MySQL queries.

load("@subpar//:subpar.bzl", "par_binary")
load("//tools/bzl:exports.bzl", "exports_repo")

exports_repo(
    name = "export",
    github_repo = "jasper",
    move_file_mapping = {
        "util/jasper/README.md": "README.md",
        "util/jasper/LICENSE": "LICENSE",
    },
    targets = [
        "//util/jasper/...",
    ],
)

par_binary(
    name = "jasper",
    srcs = ["jasper.py"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:text",
        "//third_party/py/sqlparse",
        "//util/lmk",
    ],
)

py_test(
    name = "jasper_test",
    srcs = ["jasper_test.py"],
    deps = [
        ":jasper",
        "//labm8/py:test",
        "//third_party/py/pytest",
    ],
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [":jasper.par"],
)
