"""This module prepares datasets for data flow analyses."""
import multiprocessing
import os
import pathlib
import signal
import sys
import threading
import time
import traceback
from typing import Iterable
from typing import List
from typing import Tuple

import networkx as nx
import sqlalchemy as sql
import tqdm

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.labelled import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.polyhedra import polyhedra
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.labelled.subexpressions import subexpressions
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import ppar
from labm8.py import prof

app.DEFINE_string("analysis", None, "The name of the analysis to run.")
app.DEFINE_database(
  "input_db",
  unlabelled_graph_database.Database,
  None,
  "Database to read unlabelled graph protos from.",
  must_exist=True,
)
app.DEFINE_database(
  "output_db",
  graph_tuple_database.Database,
  None,
  "Database to write annotated graph tuples to.",
)
# TODO(github.com/ChrisCummins/ProGraML/issues/22): Port all graph annotators
# to new graph tuple representation.
# app.DEFINE_database(
#   "bytecode_db",
#   bytecode_database.Database,
#   None,
#   "URL of database to read bytecode from. Only required when "
#   "analysis requires bytecode.",
#   must_exist=True,
# )
app.DEFINE_integer(
  "max_instances_per_graph",
  10,
  "The maximum number of instances to produce from a single input graph. "
  "For a graph with `n` root statements, `n` instances can be produced by "
  "changing the root statement.",
)
app.DEFINE_integer(
  "annotator_timeout",
  120,
  "The maximum number of seconds to allow an annotator to process a single "
  "graph.",
)
app.DEFINE_integer(
  "nproc",
  multiprocessing.cpu_count(),
  "Tuning parameter. The number of processes to spawn.",
)
app.DEFINE_integer(
  "proto_batch_mb",
  4,
  "Tuning parameter. The number of megabytes of protocol buffers to read in "
  "a batch.",
)
app.DEFINE_integer(
  "max_reader_queue_size", 1, "Tuning parameter. The maximum number of proto."
)
app.DEFINE_integer(
  "chunk_size", 32, "Tuning parameter. The number of processes to spawn."
)

app.DEFINE_boolean("error", False, "If true, crash on export error.")

FLAGS = app.FLAGS


def GetDataFlowGraphAnnotator(
  name: str,
) -> data_flow_graphs.DataFlowGraphAnnotator:
  """Get the data flow annotator."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/22): Port all graph annotators
  # to new graph tuple representation.
  if name == "reachability":
    return reachability.ReachabilityAnnotator()
  else:
    raise app.UsageError(f"Unknown analyses '{name}'")


class ProgressContext(object):
  """The context for logging and profiling within a progress bar."""

  def __init__(self, print_context):
    self.print_context = print_context

  def Log(self, *args, **kwargs):
    app.Log(*args, **kwargs, print_context=self.print_context)

  def Profile(self, level: int, msg):
    return prof.Profile(
      msg,
      print_to=lambda x: app.Log(level, x, print_context=self.print_context),
    )

  def Warning(self, *args, **kwargs):
    app.Warning(*args, **kwargs, print_context=self.print_context)

  def Error(self, *args, **kwargs):
    app.Error(*args, **kwargs, print_context=self.print_context)


def BatchedGraphReader(
  graph_db: unlabelled_graph_database.Database,
  ids_and_sizes_to_do: List[Tuple[int, int]],
  batch_size_in_bytes: int,
  ctx: ProgressContext,
) -> Iterable[List[unlabelled_graph_database.ProgramGraph]]:
  """Read from the given list of graph IDs in batches."""
  ids_and_sizes_to_do = sorted(ids_and_sizes_to_do, key=lambda x: x[0])
  i = 0
  while i < len(ids_and_sizes_to_do):
    end_i = i
    batch_size = 0
    while batch_size < batch_size_in_bytes:
      batch_size += ids_and_sizes_to_do[end_i][1]
      end_i += 1
      if end_i >= len(ids_and_sizes_to_do):
        # We have run out of graphs to read.
        break

    with graph_db.Session() as session:
      with ctx.Profile(
        2,
        f"[reader {os.getpid()}] Read {humanize.BinaryPrefix(batch_size, 'B')} "
        f"batch of {end_i - i} unlabelled graphs",
      ):
        start_id = ids_and_sizes_to_do[i][0]
        end_id = ids_and_sizes_to_do[end_i - 1][0]
        graphs = (
          session.query(unlabelled_graph_database.ProgramGraph)
          .filter(unlabelled_graph_database.ProgramGraph.ir_id >= start_id)
          .filter(unlabelled_graph_database.ProgramGraph.ir_id <= end_id)
          .options(
            sql.orm.joinedload(unlabelled_graph_database.ProgramGraph.data)
          )
          .all()
        )
      yield graphs

    i = end_i


def AnnotateWithTimeout(
  annotator: data_flow_graphs.DataFlowGraphAnnotator,
  g: nx.MultiDiGraph,
  n: int,
  seconds: int,
) -> List[data_flow_graphs.DataFlowAnnotatedGraph]:
  """Run the given annotator with a timeout."""

  def _RaiseTimoutError(signum, frame):
    del signum
    del frame
    raise TimeoutError(f"Function failed to complete within {seconds} seconds")

  # Register a function to raise a TimeoutError on the signal.
  signal.signal(signal.SIGALRM, _RaiseTimoutError)
  signal.alarm(seconds)
  try:
    return annotator.MakeAnnotated(g, n)
  except TimeoutError as e:
    raise e
  finally:
    # Unregister the signal so it won't be triggered
    # if the timeout is not reached.
    signal.signal(signal.SIGALRM, signal.SIG_IGN)


def MakeAnnotatedGraphs(
  packed_args,
) -> Tuple[int, List[graph_tuple_database.GraphTuple]]:
  """Multiprocess worker."""
  analysis: str = packed_args[0]
  graphs: List[unlabelled_graph_database.ProgramGraph] = packed_args[1]
  ctx: ProgressContext = packed_args[2]

  ctx.Log(3, "[worker %s] Processing %s graphs", os.getpid(), len(graphs))

  annotator = GetDataFlowGraphAnnotator(analysis)

  graph_tuples: List[graph_tuple_database.GraphTuple] = []
  for i, program_graph in enumerate(graphs):
    with ctx.Profile(
      3, f"[worker {os.getpid()}] Processed graph [{i+1}/{len(graphs)}]"
    ):
      graph = programl.ProgramGraphToNetworkX(program_graph.proto)
      try:
        # Create annotated graphs.
        annotated_graphs = AnnotateWithTimeout(
          annotator,
          graph,
          FLAGS.max_instances_per_graph,
          FLAGS.annotator_timeout,
        )
        # Create GraphTuple database instances from annotated graphs.
        graph_tuples += [
          graph_tuple_database.GraphTuple.CreateFromDataFlowAnnotatedGraph(
            a, program_graph.ir_id
          )
          for a in annotated_graphs
        ]
      except Exception as e:
        graph_tuples.append(
          graph_tuple_database.GraphTuple.CreateEmpty(ir_id=program_graph.ir_id)
        )
        _, _, tb = sys.exc_info()
        tb = traceback.extract_tb(tb, 2)
        filename, line_number, function_name, *_ = tb[-1]
        filename = pathlib.Path(filename).name
        ctx.Error(
          "Failed to annotate graph for ProgramGraph.ir_id=%d: %s "
          "(%s:%s:%s() -> %s)",
          program_graph.ir_id,
          e,
          filename,
          line_number,
          function_name,
          type(e).__name__,
        )
        if FLAGS.error:
          raise e
  return len(graphs), graph_tuples


class DatasetGenerator(threading.Thread):
  """Worker thread for dataset."""

  def __init__(
    self,
    graph_reader: ppar.ThreadedIterator,
    analysis: str,
    output_db: graph_tuple_database.Database,
    ctx: ProgressContext,
    exported_count: int = 0,
  ):
    self.graph_reader = graph_reader
    self.analysis = analysis
    self.output_db = output_db
    self.ctx = ctx
    self.exported_count = exported_count

    super(DatasetGenerator, self).__init__()
    self.start()

  def run(self):
    """Run the dataset generation."""
    pool = multiprocessing.Pool(FLAGS.nproc)

    self.graph_reader.Start()

    def MakeAnnotatedGraphsArgsGenerator(graph_reader):
      """Generate packed arguments for a multiprocessing worker."""
      for graph_batch in graph_reader:
        yield (self.analysis, graph_batch, self.ctx)

    # Have a thread generating inputs, and a multiprocessing pool consuming
    # them.
    worker_args = MakeAnnotatedGraphsArgsGenerator(self.graph_reader)
    workers = pool.imap_unordered(MakeAnnotatedGraphs, worker_args)
    start_time = time.time()
    for graph_count, graph_tuples in workers:
      self.exported_count += graph_count
      # Record the generated annotated graphs.
      tuples_size = sum(t.pickled_graph_tuple_size for t in graph_tuples)
      with self.output_db.Session(commit=True) as out_session:
        out_session.add_all(graph_tuples)
      end_time = time.time()
      self.ctx.Log(
        2,
        "[writer %s] Processed %s graphs (%s graph tuples, %s) in %s",
        os.getpid(),
        graph_count,
        len(graph_tuples),
        humanize.BinaryPrefix(tuples_size, "B"),
        humanize.Duration(end_time - start_time),
      )
      start_time = end_time


def MakeDataFlowAnalysisDataset(
  analysis: str,
  input_db: unlabelled_graph_database.Database,
  output_db: graph_tuple_database.Database,
):
  """Run the given analysis."""
  # Create an annotator now to crash-early if the analysis name is invalid.
  GetDataFlowGraphAnnotator(analysis)

  # Get the graphs that have already been processed.
  with output_db.Session() as out_session:
    already_done_max, already_done_count = out_session.query(
      sql.func.max(graph_tuple_database.GraphTuple.ir_id),
      sql.func.count(sql.func.distinct(graph_tuple_database.GraphTuple.ir_id)),
    ).one()
    already_done_max = already_done_max or -1

  # Get the total number of graphs to process, and the IDs of the graphs to
  # process.
  with input_db.Session() as in_session:
    total_graph_count = in_session.query(
      sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
    ).scalar()
    ids_and_sizes_to_do = [
      (row.ir_id, row.serialized_proto_size)
      for row in in_session.query(
        unlabelled_graph_database.ProgramGraph.ir_id,
        unlabelled_graph_database.ProgramGraph.serialized_proto_size,
      )
      .filter(unlabelled_graph_database.ProgramGraph.ir_id > already_done_max)
      .order_by(unlabelled_graph_database.ProgramGraph.ir_id)
    ]

  # Sanity check.
  if len(ids_and_sizes_to_do) + already_done_count != total_graph_count:
    app.FatalWithoutStackTrace(
      "ids_to_do(%s) + already_done(%s) != total_rows(%s)",
      len(ids_and_sizes_to_do),
      already_done_count,
      total_graph_count,
    )

  with output_db.Session(commit=True) as out_session:
    out_session.add(
      unlabelled_graph_database.Meta.Create(
        key="Graph counts", value=(already_done_count, total_graph_count)
      )
    )
  app.Log(
    1,
    "Selected %s of %s to process",
    humanize.Commas(len(ids_and_sizes_to_do)),
    humanize.Plural(total_graph_count, "unlabelled graph"),
  )

  bar = tqdm.tqdm(
    initial=already_done_count,
    total=total_graph_count,
    desc=analysis,
    unit=" protos",
    position=0,
  )
  ctx = ProgressContext(bar.external_write_mode)

  graph_reader = ppar.ThreadedIterator(
    BatchedGraphReader(
      input_db, ids_and_sizes_to_do, FLAGS.proto_batch_mb * 1024 * 1024, ctx
    ),
    max_queue_size=FLAGS.max_reader_queue_size,
    start=False,
  )

  # Run migration asynchronously so that we can keep the progress bar updating.
  generator = DatasetGenerator(
    graph_reader=graph_reader,
    analysis=analysis,
    output_db=output_db,
    ctx=ctx,
    exported_count=already_done_count,
  )
  while generator.is_alive():
    bar.n = generator.exported_count
    bar.refresh()
    generator.join(0.25)

  # Sanity check the number of generated program graphs.
  if generator.exported_count != total_graph_count:
    app.FatalWithoutStackTrace(
      "unlabelled_graph_count(%s) != exported_count(%s)",
      total_graph_count,
      generator.exported_count,
    )
  with output_db.Session() as out_session:
    annotated_graph_count = out_session.query(
      sql.func.count(sql.func.distinct(graph_tuple_database.GraphTuple.ir_id))
    ).scalar()
  if annotated_graph_count != total_graph_count:
    app.FatalWithoutStackTrace(
      "unlabelled_graph_count(%s) != annotated_graph_count(%s)",
      total_graph_count,
      annotated_graph_count,
    )

  ctx.Log(1, "Done!")


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()

  MakeDataFlowAnalysisDataset(FLAGS.analysis, input_db, output_db)


if __name__ == "__main__":
  app.Run(main)
