# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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
"""Generate labelled program graphs using data flow analysis.

This program reads a ProgramGraph protocol buffer from stdin, creates labelled
graphs by running a specified analysis, then prints the results as a
ProgramGraphs protocol buffer to stdout. Use --stdin_fmt and --stdout_fmt to
support binary or text formats.

List the available analyses using:

    $ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:annotate -- --list

For analyses that can produce multiple labelled graphs (e.g. by picking
different root nodes), use the --n argument to limit the number of generated
graphs.

For example, to produce up to 5 labelled graphs using reachability analysis
and text format protocol buffers:

    $ bazel run //deeplearning/ml4pl/graphs/labelled/dataflow:annotate -- \
        --analysis=reachability \
        --stdin_fmt=pbtxt \
        --stdout_fmt=pbtxt \
        --n=5 \
        < /tmp/program_graph.pbtxt
"""
import signal
import subprocess
import sys
from typing import List
from typing import Union

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled.dataflow.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.dataflow.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.dataflow.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.dataflow.polyhedra import polyhedra
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.graphs.labelled.dataflow.subexpressions import (
  subexpressions,
)
from deeplearning.ml4pl.testing import test_annotators
from labm8.py import app
from labm8.py import bazelutil


app.DEFINE_boolean(
  "list", False, "If true, list the available analyses and exit."
)
app.DEFINE_string("analysis", "", "The name of the analysis to run.")
app.DEFINE_integer(
  "n",
  10,
  "The maximum number of labelled program graphs to produce. "
  "For a graph with `n` root statements, `n` instances can be produced by "
  "changing the root statement. If --n=0, enumerate all possible labelled "
  "graphs.",
)

FLAGS = app.FLAGS

# A map from analysis name to a callback which instantiates a
# DataFlowGraphAnnotator object for this anlysis. To add a new analysis, create
# a new entry in this table.
ANALYSES = {
  "reachability": reachability.ReachabilityAnnotator,
  "domtree": dominator_tree.DominatorTreeAnnotator,
  "liveness": liveness.LivenessAnnotator,
  "datadep": data_dependence.DataDependencyAnnotator,
  "subexpressions": subexpressions.CommonSubexpressionAnnotator,
  # Annotators which are used for testing this script. These should, for obvious
  # reasons, not be used in prod. However, they must remain here so that we can
  # test the behaviour of the annotator under various conditions.
  "test_pass_thru": test_annotators.PassThruAnnotator,
  "test_flaky": test_annotators.FlakyAnnotator,
  "test_timeout": test_annotators.TimeoutAnnotator,
  "test_error": test_annotators.ErrorAnnotator,
  "test_empty": test_annotators.EmptyAnnotator,
}

# A list of the available analyses. We filter out the test_xxx named annotators
# for clarity.
AVAILABLE_ANALYSES = sorted(
  analysis for analysis in ANALYSES if not analysis.startswith("test_")
)

# The path of this script. Because a target cannot depend on itself, all calling
# code must add this script to its `data` dependencies.
SELF = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/labelled/dataflow/annotate"
)


# Return codes for error conditions.
#
# Error reading stdin.
E_INVALID_INPUT = 11
# The analysis failed.
E_ANALYSIS_FAILED = 12
# Error writing stdout.
E_INVALID_STDOUT = 13


def _AnnotateInSubprocess(
  analysis: str,
  graph: Union[programl_pb2.ProgramGraph, bytes],
  n: int = 0,
  timeout: int = 120,
  binary_graph: bool = False,
) -> programl_pb2.ProgramGraphs:
  """Run this script in a subprocess.

  This is the most robust method for enforcing the timeout, but has a huge
  overhead in starting up a new python interpreter for every invocation.

  DISCLAIMER: Because a target cannot depend on itself, all calling code must
  add //deeplearning/ml4pl/graphs/labelled/dataflow:annotate to its list of
  data dependencies.

  Args:
    analysis: The name of the analysis to run.
    graph: The unlabelled ProgramGraph protocol buffer to to annotate, either
      as a proto instance or as binary-encoded byte array.
    n: The maximum number of labelled graphs to produce.
    timeout: The maximum number of seconds to run the analysis for.
    binary_graph: If true, treat the graph argument as a binary byte array.

  Returns:
    A ProgramGraphs protocol buffer.

  Raises:
    IOError: If serializing the input or output protos fails.
    ValueError: If an invalid analysis is requested.
    data_flow_graphs.AnalysisFailed: If the analysis raised an error.
    data_flow_graphs.AnalysisTimeout: If the analysis did not complete within
      the requested timeout.
  """
  process = subprocess.Popen(
    [
      "timeout",
      "-s9",
      str(timeout),
      str(SELF),
      "--analysis",
      analysis,
      "--n",
      str(n),
      "--stdin_fmt",
      "pb",
      "--stdout_fmt",
      "pb",
    ],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )

  # Encode the input if required.
  if binary_graph:
    stdin = graph
  else:
    stdin = programl.ToBytes(graph, fmt=programl.StdoutGraphFormat.PB)

  # Run this analysis script.
  stdout, stderr = process.communicate(stdin)

  if process.returncode == 9 or process.returncode == -9:
    # Process was killed. We assume this is because of timeout, though it could
    # be the user.
    raise data_flow_graphs.AnalysisTimeout(timeout)
  elif process.returncode == E_INVALID_INPUT:
    raise IOError("Failed to serialize input graph")
  elif process.returncode == E_INVALID_STDOUT:
    raise IOError("Analysis failed to write stdout")
  elif process.returncode:
    raise data_flow_graphs.AnalysisFailed(
      f"Analysis failed with returncode {process.returncode}: "
      f"{stderr.decode('utf-8')}"
    )

  # Construct the protocol buffer from stdout.
  output = programl.FromBytes(
    stdout,
    programl.StdinGraphFormat.PB,
    proto=programl_pb2.ProgramGraphs(),
    empty_okay=True,
  )

  return output


def Annotate(
  analysis: str,
  graph: Union[programl_pb2.ProgramGraph, bytes],
  n: int = 0,
  timeout: int = 120,
) -> data_flow_graphs.DataFlowGraphs:
  """Run an analysis

  Args:
    analysis: The name of the analysis to run.
    graph: The unlabelled ProgramGraph protocol buffer to to annotate, either
      as a proto instance or as binary-encoded byte array.
    n: The maximum number of labelled graphs to produce.
    timeout: The maximum number of seconds to run the analysis for.
    binary_graph: If true, treat the graph argument as a binary byte array.

  Returns:
    A ProgramGraphs protocol buffer.

  Raises:
    ValueError: If an invalid analysis is requested.
    IOError: If serializing the input or output protos fails.
    data_flow_graphs.AnalysisFailed: If the analysis raised an error.
    data_flow_graphs.AnalysisTimeout: If the analysis did not complete within
      the requested timeout.
  """
  if analysis not in ANALYSES:
    raise ValueError(
      f"Unknown analysis: {analysis}. "
      f"Available analyses: {AVAILABLE_ANALYSES}",
    )

  def TimeoutHandler(signum, frame):
    """Callback to raise a timeout error."""
    del signum
    del frame
    raise data_flow_graphs.AnalysisTimeout(timeout)

  signal.signal(signal.SIGALRM, TimeoutHandler)
  signal.alarm(timeout)
  annotator = ANALYSES[analysis](graph)

  try:
    annotated_graphs = annotator.MakeAnnotated(n)
  finally:
    signal.alarm(0)

  return annotated_graphs


def Main():
  """Main entry point."""
  if FLAGS.list:
    print(f"Available analyses: {AVAILABLE_ANALYSES}")
    return

  n = FLAGS.n

  try:
    input_graph = programl.ReadStdin()
  except Exception as e:
    print(f"Error parsing stdin: {e}", file=sys.stderr)
    sys.exit(E_INVALID_INPUT)

  annotator = ANALYSES[FLAGS.analysis](input_graph)

  annotated_graphs: List[programl_pb2.ProgramGraph] = []
  try:
    for annotated_graph in annotator.MakeAnnotated(n):
      annotated_graphs.append(annotated_graph)
  except Exception as e:
    print(f"{e}", file=sys.stderr)
    sys.exit(E_ANALYSIS_FAILED)

  try:
    programl.WriteStdout(programl_pb2.ProgramGraphs(graph=annotated_graphs))
  except Exception as e:
    print(f"Error writing stdout: {e}", file=sys.stderr)
    sys.exit(E_INVALID_STDOUT)


if __name__ == "__main__":
  app.Run(Main)
