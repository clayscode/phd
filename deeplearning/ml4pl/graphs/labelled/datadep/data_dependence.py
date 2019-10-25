"""Module for labelling program graphs with data depedencies."""
import collections
import networkx as nx
import typing

from deeplearning.ml4pl.graphs import graph_query as query
from labm8 import app
from labm8 import decorators

FLAGS = app.FLAGS


@decorators.timeout(seconds=60)
def AnnotateDataDependencies(
    g: nx.MultiDiGraph,
    root_node: str,
    x_label: str = 'x',
    y_label: str = 'y',
    true: typing.Any = True,
    false: typing.Any = False) -> typing.Tuple[int, int]:
  # Initialize all nodes as not dependent and not the root node, except the root
  # node.
  for node, data in g.nodes(data=True):
    data[x_label] = false
    data[y_label] = false
  g.nodes[root_node][x_label] = true

  # Breadth-first traversal to mark node dependencies.
  data_flow_steps = 0
  dependency_node_count = 0
  visited = set()
  q = collections.deque([(root_node, 1)])
  while q:
    next, data_flow_steps = q.popleft()
    dependency_node_count += 1
    visited.add(next)

    data_predecessors = query.StatementNeighbors(g,
                                                 next,
                                                 flow='data',
                                                 direction=lambda src, dst: src)
    print("DEPS", next, data_predecessors)
    for neighbor in data_predecessors:
      if neighbor not in visited:
        q.append((neighbor, data_flow_steps + 1))

    # Mark the node as dependent.
    g.nodes[next][y_label] = true
    print("N", next)

  return dependency_node_count, data_flow_steps


def MakeDataDependencyGraphs(
    g: nx.MultiDiGraph,
    n: typing.Optional[int] = None,
    false=False,
    true=True,
) -> typing.Iterable[nx.MultiDiGraph]:
  """Produce up to `n` dependency graphs from the given unlabelled graph.

  Args:
    g: The unlabelled input graph.
    n: The maximum number of graphs to produce. Multiple graphs are produced by
      selecting different root nodes for creating labels. If `n` is provided,
      the number of graphs generated will be in the range
      1 <= x <= min(num_statements, n). Else, the number of graphs will be equal
      to num_statements (i.e. one graph for each statement in the input graph).
    false: The value to set for binary false values on X and Y labels.
    true: The value to set for binary true values on X and Y labels.

  Returns:
    A generator of annotated graphs, where each graph has 'x' and 'y' labels on
    the statement nodes, and additionally 'dominated_node_count' and
    'data_flow_max_steps_required' attributes.
  """
  root_statements = query.SelectRandomNStatements(g, n)

  for root_node in root_statements[:n]:
    labelled = g.copy()
    labelled.dependency_node_count, labelled.data_flow_max_steps_required = (
        AnnotateDataDependencies(labelled, root_node, false=false, true=true))
    yield labelled
