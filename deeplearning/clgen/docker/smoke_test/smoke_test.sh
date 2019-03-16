#!/usr/bin/env bash
#
# Test that CLgen docker can train and sample a model on the tiny corpus.
#
set -eux
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT

cp deeplearning/clgen/tests/data/tiny/corpus.tar.bz2 "$workdir"
cp deeplearning/clgen/tests/data/tiny/config.pbtxt "$workdir"
# Set working directory in config file.
sed -i 's,working_dir: ".*",working_dir: "/clgen",' "$workdir"/config.pbtxt

sed -i 's/num_epochs: .+/num_epochs: 2/' "$workdir"/config.pbtxt

# FIXME(cec): Presently the clang_rewriter doesn't work when testing
# a host-compiled docker image. For this to work, you must build the docker
# image from within a docker phd_build container.
sed -i '/preprocessor: "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers"/d' "$workdir"/config.pbtxt

docker load -i deeplearning/clgen/docker/clgen.tar
docker run -v"$workdir":/clgen bazel/deeplearning/clgen/docker:clgen \
  --min_samples=10
