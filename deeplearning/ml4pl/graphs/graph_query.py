"""Functions for iterating and maneuvering around graphs."""
import collections
import random
import typing

import networkx as nx
from labm8 import app

from deeplearning.ml4pl.graphs import graph_iterators as iterators
from deeplearning.ncc.inst2vec import inst2vec_preprocess

FLAGS = app.FLAGS


def StatementNeighbors(
    g: nx.Graph,
    node: str,
    flow='control',
    direction: typing.Optional[
        typing.Callable[[typing.Any, typing.Any], typing.Any]] = None
) -> typing.Set[str]:
  """Return the neighboring statements connected by the given flow type."""
  direction = direction or (lambda src, dst: dst)
  neighbors = set()
  neighbor_edges = direction(g.in_edges, g.out_edges)
  for src, dst, edge_flow in neighbor_edges(node,
                                            data='flow',
                                            default='control'):
    neighbor = direction(src, dst)
    if edge_flow == flow:
      if g.nodes[neighbor].get('type', 'statement') == 'statement':
        neighbors.add(neighbor)
      else:
        neighbors = neighbors.union(StatementNeighbors(g, neighbor, flow=flow))
  return neighbors


def SuccessorNodes(
    g: nx.DiGraph,
    node: str,
    direction: typing.Optional[
        typing.Callable[[typing.Any, typing.Any], typing.Any]] = None,
    ignored_nodes: typing.Optional[typing.Iterable[str]] = None
) -> typing.List[str]:
  """Find the successor nodes of a node."""
  direction = direction or (lambda src, dst: dst)
  ignored_nodes = ignored_nodes or set()
  real = []
  for src, dst in direction(g.in_edges, g.out_edges)(node):
    if direction(src, dst) not in ignored_nodes:
      real.append(direction(src, dst))
    else:
      # The node is ignored, so skip over it and look for the next
      real += SuccessorNodes(g, direction(src, dst), ignored_nodes, direction)
  return real


def StatementIsSuccessor(g: nx.MultiDiGraph,
                         src: str,
                         dst: str,
                         flow: str = 'control') -> bool:
  """Return True if `dst` is successor to `src`."""
  visited = set()
  q = collections.deque([src])
  while q:
    current = q.popleft()
    if current == dst:
      return True
    visited.add(current)
    for next_node in g.neighbors(current):
      edge_flow = g.edges[current, next_node, 0].get('flow', 'control')
      if edge_flow != flow:
        continue
      node_type = g.nodes[next_node].get('type', 'statement')
      if node_type != 'statement':
        continue
      if next_node in visited:
        continue
      q.append(next_node)
  return False


def SelectRandomNStatements(g: nx.Graph, n: int):
  root_statements = [node for node, _ in iterators.StatementNodeIterator(g)]
  n = n or len(root_statements)

  # If we're taking a sample of nodes to produce graphs (i.e. not all of them),
  # process the nodes in a random order.
  if n < len(root_statements):
    random.shuffle(root_statements)

  return root_statements


def GetCallStatementSuccessor(graph: nx.MultiDiGraph, call_site: str) -> str:
  """Find the successor statement for a call statement.

  Returns:
    The successor statement node.

  Raises:
    ValueError: If call site does not have exactly one successor.
  """
  call_site_successors = []
  for src, dst, flow in graph.out_edges(call_site,
                                        data='flow',
                                        default='control'):
    if (flow == 'control' and
        graph.nodes[dst].get('type', 'statement') == 'statement'):
      call_site_successors.append(dst)
  if len(call_site_successors) != 1:
    raise ValueError(
        f"Call statement `{call_site}` should have exactly one successor "
        "statement but found "
        f"{humanize.Plural(len(call_size_successors), 'successor')}: "
        f"`{call_site_successors}`")
  return list(call_site_successors)[0]


def GetCalledFunctionName(statement) -> typing.Optional[str]:
  """Get the name of a function called in the statement."""
  if 'call ' not in statement:
    return None
  # Try and resolve the call destination.
  _, m_glob, _, _ = inst2vec_preprocess.get_identifiers_from_line(statement)
  if not m_glob:
    return None
  return m_glob[0][1:]  # strip the leading '@' character


def FindCallSites(graph, src, dst):
  """Find the statements in `src` function that call `dst` function."""
  call_sites = []
  for node, data in iterators.StatementNodeIterator(graph):
    if data['function'] != src:
      continue
    statement = data.get('original_text', data['text'])
    called_function = GetCalledFunctionName(statement)
    if not called_function:
      continue
    if called_function == dst:
      call_sites.append(node)
  return call_sites


def LoopConnetedness(graph) -> int:
  """Return the loop connectedness of a graph.

  Args:
    graph: The graph to compute the loop connectedness of.

  Returns:
    A non-negative loop connectedness value.
  """
  # TODO(github.com/ChrisCummins/ml4pl/issues/5): This overestimates the loop
  # connectedness by counting *all* back edges, not just the ones on the longest
  # acyclic path through the graph.
  visited = set()
  back_edge_count = 0

  for node in graph.nodes():
    if graph.in_degree(node) == 1:
      root = node
      break
  else:
    raise ValueError("No entry block found in graph")

  stack = [root]
  while stack:
    node = stack[-1]
    stack.pop()

    if node in visited:
      back_edge_count += 1
    else:
      visited.add(node)
      for _, dst, flow in graph.out_edges(node, data='flow', default='control'):
        if flow == 'control':
          stack.append(dst)

  return back_edge_count
