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
"""Module for conversion from unlabelled graphs to encoded sequences."""
import json
import subprocess
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lru
import numpy as np
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.seq import graph2seq_pb2
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_integer(
  "graph2seq_cache_entries",
  10000,
  "The number of ID -> encoded sequence entries to cache.",
)
app.DEFINE_integer(
  "graph_encoder_timeout",
  120,
  "The number of seconds to permit the graph encoder to run before terminating.",
)

GRAPH_ENCODER_WORKER = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/seq/graph_encoder_worker"
)

# The vocabulary to use for LLVM encoders. Use
# //deeplearning/ml4pl/seq:derive_vocab to generate a vocabulary.
LLVM_VOCAB = bazelutil.DataPath("phd/deeplearning/ml4pl/seq/llvm_vocab.json")


class EncoderBase(object):
  """Base class for performing graph-to-encoded sequence translation."""

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    cache_size: Optional[int] = None,
  ):
    self.graph_db = graph_db

    # Maintain a mapping from IR IDs to encoded sequences to amortize the
    # cost of encoding.
    cache_size = cache_size or FLAGS.graph2seq_cache_entries
    self.ir_id_to_encoded: Dict[int, np.array] = lru.LRU(cache_size)

  def Encode(
    self,
    graphs: List[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[Union[np.array, graph2seq_pb2.ProgramGraphSeq]]:
    """Translate a list of graphs to encoded sequences."""
    unique_ids = {graph.ir_id for graph in graphs}
    id_to_encoded = {
      ir_id: self.ir_id_to_encoded[ir_id]
      for ir_id in unique_ids
      if ir_id in self.ir_id_to_encoded
    }

    ctx.Log(
      5,
      "%.2f%% encoded graph cache hit rate",
      (len(id_to_encoded) / len(unique_ids)) * 100,
    )

    if len(id_to_encoded) != len(unique_ids):
      unknown_ir_ids = {
        ir_id for ir_id in unique_ids if ir_id not in id_to_encoded
      }

      # Encode the unknown IRs.
      sorted_ir_ids_to_encode = sorted(unknown_ir_ids)
      sorted_encoded_sequences = self.EncodeIds(sorted_ir_ids_to_encode, ctx)

      # Cache the recently encoded sequences. We must do this *after* fetching
      # from the cache to prevent the cached items from being evicted.
      for ir_id, encoded in zip(
        sorted_ir_ids_to_encode, sorted_encoded_sequences
      ):
        id_to_encoded[ir_id] = encoded
        self.ir_id_to_encoded[ir_id] = encoded

    # Assemble the list of encoded graphs.
    encoded = [id_to_encoded[graph.ir_id] for graph in graphs]

    return encoded

  def EncodeIds(
    self, ir_ids: List[int], ctx: progress.ProgressContext
  ) -> List[Union[np.array, graph2seq_pb2.ProgramGraphSeq]]:
    """Encode a list of graph IDs and return the sequences in the same order."""
    raise NotImplementedError("abstract class")

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    raise NotImplementedError("abstract class")

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary, including the unknown-vocab element."""
    raise NotImplementedError("abstract class")


class GraphEncoder(EncoderBase):
  """Encode a graph to a single encoded sequence.

  Uses the original intermediate representation to produce the tokenized
  sequence, entirely discarding the graph structure.
  """

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    ir2seq_encoder: ir2seq.EncoderBase,
    cache_size: Optional[int] = None,
  ):
    super(GraphEncoder, self).__init__(graph_db, cache_size)
    self.ir2seq_encoder = ir2seq_encoder

  @property
  def max_encoded_length(self) -> int:
    """Return an upper bound on the length of the encoded sequences."""
    return self.ir2seq_encoder.max_encoded_length

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary, including the unknown-vocab element."""
    return self.ir2seq_encoder.vocabulary_size

  def EncodeIds(
    self, ir_ids: List[int], ctx: progress.ProgressContext
  ) -> List[np.array]:
    """Return encoded sequences for the given graph IDs.

    This adapts the methodology used in the PACT'17 "DeepTune" paper to LLVM
    IR. It provides a tokenized list of vocabulary indices from bytecodes, which
    can then be processed by sequential models.

    Args:
      graphs: A list of unlabelled graphs to encode.
      ctx: A logging context.

    Returns:
      A list of encoded sequences.
    """
    return self.ir2seq_encoder.Encode(ir_ids, ctx=ctx)


class StatementEncoder(EncoderBase):
  """Encode graphs to per-node sub-sequences.

  This uses the graph structure to produce a tokenized sequence ordered by
  depth first traversal, and allows mapping from graph nodes to sub-sequences
  within the encoded output.
  """

  def __init__(
    self,
    graph_db: graph_tuple_database.Database,
    proto_db: unlabelled_graph_database.Database,
    max_encoded_length: int,
    max_nodes: int,
    cache_size: Optional[int] = None,
  ):
    super(StatementEncoder, self).__init__(graph_db, cache_size)
    self.proto_db = proto_db

    with open(LLVM_VOCAB) as f:
      data_to_load = json.load(f)
    self.vocabulary = data_to_load["vocab"]
    self._max_encoded_length = max_encoded_length
    self.max_nodes = max_nodes

  @property
  def max_encoded_length(self) -> int:
    return self._max_encoded_length

  def EncodeIds(
    self, ir_ids: List[int], ctx: progress.ProgressContext
  ) -> List[graph2seq_pb2.ProgramGraphSeq]:
    """Serialize a graph into an encoded sequence.

    This method is used to provide a serialized sequence of encoded tokens
    of the statements in a program graph that can be processed sequentially,
    and a grouping of encoded tokens to statements.

    For example, the graph of this function:

      define i32 @B() #0 {
        %1 = alloca i32, align 4
        store i32 0, i32* %1, align 4
        ret i32 15
      }

    would comprise three statement nodes, and additional data nodes for the
    statements' operands (e.g. %1). A pre-order depth first traversal of the
    graph produces the linear ordering of statement nodes, which are then
    encoded and concatenated to produce a list of vocabulary indices such as:

      [
         0, 1, 2, 1, 4, 5, 6,  # encoded `%1 = alloca i32, align 4`
         12, 9, 3,             # encoded `store i32 0, i32* %1, align 4`
         9, 8,                 # encoded `ret i32 15`
      ]

    This encoded sequence can then be grouped into the three individual
    statements by assigning each statement a unique ID, such as:

      [
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1,
        2, 2,
      ]

    This method computes and returns these two arrays, along with a third array
    which contains a masking of nodes from the input program graph, marking the
    non-statement nodes as inactive. E.g. for a graph with 5 statement nodes
    and 3 data nodes, the mask will consist of 8 boolean values, 5 True, 3
    False. Use this array to mask a 'node_y' label list to exclude the labels
    for non-statement nodes.

    Args:
      graphs: A list of graphs to encode.
      ctx: A logging context.

    Returns:
      A list of EncodedSubsequence tuples, where each tuple maps a graph to
      encoded sequences, subsequence groupings, and node_mask arrays which list
      the nodes which are selected from each graph.
    """
    # Fetch the protos for the graphs that we need to encode.
    with self.proto_db.Session() as session:
      protos_to_encode = [
        row.proto
        for row in session.query(unlabelled_graph_database.ProgramGraph)
        .options(
          sql.orm.joinedload(unlabelled_graph_database.ProgramGraph.data)
        )
        .filter(unlabelled_graph_database.ProgramGraph.ir_id.in_(ir_ids))
        .order_by(unlabelled_graph_database.ProgramGraph.ir_id)
      ]
      if len(protos_to_encode) != len(ir_ids):
        raise OSError(
          f"Requested {len(ir_ids)} protos "
          "from database but received "
          f"{len(protos_to_encode)}"
        )

    # Encode the unknown graphs. If the encoder fails, propagate the error as a
    # ValueError with the IDs of the graphs that failed.
    try:
      encoded = self.EncodeGraphs(protos_to_encode, ctx=ctx)
    except subprocess.CalledProcessError as e:
      raise ValueError(
        f"Graph encoder failed to encode IRs: {ir_ids} with error: {e}"
      )

    # Squeeze the encoded representations down to the maximum lengths allowed.
    for seq in encoded:
      seq.encoded[:] = seq.encoded[: self.max_encoded_length]
      seq.encoded_node_length[:] = seq.encoded_node_length[: self.max_nodes]
      seq.node[:] = seq.node[: self.max_nodes]

    return encoded

  @property
  def vocabulary_size(self) -> int:
    """Get the size of the vocabulary, including the unknown-vocab element."""
    return len(self.vocabulary)

  def EncodeGraphs(
    self,
    graphs: List[programl_pb2.ProgramGraph],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> List[graph2seq_pb2.ProgramGraphSeq]:
    """Encode a list of graphs and return them in order.

    Args:
      A list of zero or more strings.

    Returns:
      A pair of <encoded_sequences, statement_indices> arrays.

    Raises:
      CalledProcessError: If the graph encoder fails.
      ProtoWorkerTimeoutError: If the encoder fails to complete within 60
        seconds.
    """
    with ctx.Profile(
      3,
      lambda t: (
        f"Encoded {len(graphs)} graphs "
        f"({humanize.DecimalPrefix(token_count / t, ' tokens/sec')})"
      ),
    ):
      message = graph2seq_pb2.GraphEncoderJob(
        vocabulary=self.vocabulary, graph=graphs,
      )
      pbutil.RunProcessMessageInPlace(
        [str(GRAPH_ENCODER_WORKER)],
        message,
        timeout_seconds=FLAGS.graph_encoder_timeout,
      )
      encoded_graphs = [encoded for encoded in message.seq]
      token_count = sum(len(encoded.encoded) for encoded in encoded_graphs)
      if len(encoded_graphs) != len(graphs):
        raise ValueError(
          f"Requested {len(graphs)} graphs to be encoded but "
          f"received {len(encoded_graphs)}"
        )

    return encoded_graphs
