#!/usr/bin/env bash
#
# Test that `export_source_tree` can export itself, and the result can be built.
#
set -eux

TMPDIR="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

tools/source_tree/export_source_tree \
    --target=//tools/source_tree:export_source_tree \
    --destination="$TMPDIR"

ls "$TMPDIR"

test -f "$TMPDIR/tools/source_tree/export_source_tree.py"

cd "$TMPDIR"
# Build the exported source tree.
./configure --noninteractive
test -f bootstrap.sh
./bazel_wrapper.py build --incompatible_remove_native_http_archive=false \
  //tools/source_tree:export_source_tree

# Tidy up.
./bazel_wrapper.py clean --incompatible_remove_native_http_archive=false \
  --expunge
