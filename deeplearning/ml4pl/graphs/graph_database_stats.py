"""A module for obtaining stats from graph databases."""
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import decorators
from labm8 import humanize
from labm8 import prof


FLAGS = app.FLAGS


class GraphDatabaseStats(object):
  """Efficient aggregation of graph stats."""

  def __init__(
      self,
      db: graph_database.Database,
      filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None):
    self.db = db
    self._filters = filters or []
    self._edge_type_count = 0
    self._node_features_dimensionality = 0
    self._data_flow_max_steps_required = 0

  @decorators.memoized_property
  def graph_count(self) -> int:
    self._ComputeStats()
    return self._stats.graph_count

  @decorators.memoized_property
  def edge_type_count(self) -> int:
    self._ComputeStats()
    return self._stats.edge_type_count

  @decorators.memoized_property
  def max_node_count(self) -> int:
    self._ComputeStats()
    return self._stats.max_node_count

  @decorators.memoized_property
  def max_edge_count(self) -> int:
    self._ComputeStats()
    return self._stats.max_edge_count

  @decorators.memoized_property
  def node_features_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.node_features_dimensionality

  @decorators.memoized_property
  def edge_features_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.edge_features_dimensionality

  @decorators.memoized_property
  def graph_features_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.graph_features_dimensionality

  @decorators.memoized_property
  def node_labels_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.node_labels_dimensionality

  @decorators.memoized_property
  def edge_labels_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.edge_labels_dimensionality

  @decorators.memoized_property
  def graph_labels_dimensionality(self) -> int:
    self._ComputeStats()
    return self._stats.graph_labels_dimensionality

  @decorators.memoized_property
  def data_flow_max_steps_required(self) -> int:
    self._ComputeStats()
    return self._stats.data_flow_max_steps_required

  def __repr__(self):
    summaries = [
      f"Graphs database: {humanize.Plural(self.graph_count, 'instance')}",
      humanize.Plural(self.edge_type_count, 'edge type'),
      f"max {humanize.Plural(self.max_node_count, 'node')}",
      f"max {humanize.Plural(self.max_edge_count, 'edge')}",
    ]
    if self.node_features_dimensionality:
      summaries.append(humanize.Plural(self.node_features_dimensionality,
                                       'node feature dimension'))
    if self.edge_features_dimensionality:
      summaries.append(humanize.Plural(self.edge_features_dimensionality,
                                       'edge feature dimension'))
    if self.graph_features_dimensionality:
      summaries.append(humanize.Plural(self.graph_features_dimensionality,
                                       'graph feature dimension'))
    if self.node_labels_dimensionality:
      summaries.append(humanize.Plural(self.node_labels_dimensionality,
                                       'node label dimension'))
    if self.edge_labels_dimensionality:
      summaries.append(humanize.Plural(self.edge_labels_dimensionality,
                                       'edge label dimension'))
    if self.graph_labels_dimensionality:
      summaries.append(humanize.Plural(self.graph_labels_dimensionality,
                                       'graph label dimension'))
    if self.data_flow_max_steps_required:
      summaries.append(humanize.Plural(self.data_flow_max_steps_required,
                                       'data flow step'))
    return ", ".join(summaries)

  def _ComputeStats(self) -> None:
    with prof.Profile("Computed database stats"), self.db.Session() as s:
      q = s.query(
          sql.func.count(graph_database.GraphMeta.id).label("graph_count"),
          sql.func.max(graph_database.GraphMeta.edge_type_count).label(
              "edge_type_count"),
          sql.func.max(
              graph_database.GraphMeta.node_count).label(
              "max_node_count"),
          sql.func.max(
              graph_database.GraphMeta.edge_count).label(
              "max_edge_count"),
          sql.func.max(
              graph_database.GraphMeta.node_features_dimensionality).label(
                  "node_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.edge_features_dimensionality).label(
              "edge_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.graph_features_dimensionality).label(
              "graph_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.node_labels_dimensionality).label(
                  "node_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.edge_labels_dimensionality).label(
              "edge_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.graph_labels_dimensionality).label(
              "graph_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.data_flow_max_steps_required).label(
                  "data_flow_max_steps_required")
      )

      for filter_cb in self._filters:
        q = q.filter(filter_cb())
      self._stats = q.one()
