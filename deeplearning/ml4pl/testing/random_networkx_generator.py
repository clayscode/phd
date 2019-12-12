"""This module defines a generator for random graph tuples."""
from typing import Iterable
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


def CreateRandomGraph(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  node_count: int = None,
) -> nx.MultiDiGraph:
  """Generate a random graph.

  This generates a random graph which has sensible values for fields, but does
  not have meaningful semantics, e.g. there may be data flow edges between
  identifiers, etc. For speed, this generator guarantees only that:

    1. There is a 'root' node with outgoing call edges.
    2. Nodes are either statements, identifiers, or immediates.
    3. Nodes have text, preprocessed_text, and a single node_x value.
    4. Edges are either control, data, or call.
    5. Edges have positions.
    6. The graph is strongly connected.
  """
  proto = random_programl_generator.CreateRandomProto(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    node_count=node_count,
  )
  return programl.ProgramGraphToNetworkX(proto)


def EnumerateGraphTestSet(n: Optional[int] = None) -> Iterable[nx.MultiDiGraph]:
  """Enumerate a test set of "real" graphs."""
  for proto in random_programl_generator.EnumerateProtoTestSet(n=n):
    yield programl.ProgramGraphToNetworkX(proto)
