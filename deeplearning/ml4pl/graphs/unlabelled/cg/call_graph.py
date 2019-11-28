"""LLVM Call Graphs."""
import typing

import networkx as nx
import pydot
import pyparsing

from labm8.py import app
from labm8.py import labtypes

FLAGS = app.FLAGS


def CallGraphFromDotSource(dot_source: str) -> nx.MultiDiGraph:
  """Create a call graph from an LLVM-generated dot file.

  Args:
    dot_source: The dot source generated by the LLVM -dot-callgraph pass.

  Returns:
    A directed multigraph, where each node is a function (or the special
    "external node"), and edges indicate calls between functions.

  Raises:
    pyparsing.ParseException: If dotfile could not be parsed.
    ValueError: If dotfile could not be interpretted / is malformed.
  """
  try:
    parsed_dots = pydot.graph_from_dot_data(dot_source)
  except TypeError as e:
    raise pyparsing.ParseException('Failed to parse dot source') from e

  if len(parsed_dots) != 1:
    raise ValueError(f'Expected 1 Dot in source, found {len(parsed_dots)}')

  dot = parsed_dots[0]

  graph = nx.drawing.nx_pydot.from_pydot(dot)

  # Nodes are given a fairly arbitrary name by pydot, instead, we want to name
  # the nodes by their label, which, for all except the magic "external node"
  # node, is the name of a function.
  node_name_to_label = {}

  nodes_to_delete = []

  for node, data in graph.nodes(data=True):
    if 'label' not in data:
      nodes_to_delete.append(node)
      continue
    label = data['label']
    if label and not (label.startswith('"{') and label.endswith('}"')):
      raise ValueError(f"Invalid label: `{label}`")
    label = label[2:-2]
    node_name_to_label[node] = label
    # Remove unneeded data attributes.
    labtypes.DeleteKeys(data, {'shape', 'label'})

  # Remove unlabelled nodes.
  for node in nodes_to_delete:
    graph.remove_node(node)

  nx.relabel_nodes(graph, node_name_to_label, copy=False)
  return graph


def CallGraphToFunctionCallCounts(
    call_graph: nx.MultiDiGraph) -> typing.Dict[str, int]:
  """Build a table of call counts for each function.

  Args:
    call_graph: A call graph, such as produced by LLVM's -dot-callgraph pass.
      See CallGraphFromDotSource().

  Returns:
    A dictionary where each function in the graph is a key, and the value is the
    number of unique call sites for that function. Note this may be zero, in the
    case of library functions.
  """
  # Initialize the call count table with an entry for each function, except
  # the magic "external node" entry produced by LLVM's -dot-callgraph pass.
  function_names = [n for n in call_graph.nodes if n != 'external node']
  call_counts = {n: 0 for n in function_names}
  for src, dst, _ in call_graph.edges:
    if src != 'external node':
      call_counts[dst] += 1
  return call_counts
