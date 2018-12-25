#!/usr/bin/env bash
#
# A smoke test to ensure that target runs without crashing. This doesn't test
# the results of execution, other than the return code.

set -eux

# Runs the classification task using the published datasets used in the paper,
# but with all other parameters reduced to small values to minimize execution
# time.
deeplearning/ncc/train_inst2vec \
    --v=1 \
    --data_folder=/tmp/deeplearning/ncc/inst2vec/data \
    --use_default_dataset
    # TODO(cec): Add flags from //deeplearning/ncc/inst2vec:inst2vec_appflags.
