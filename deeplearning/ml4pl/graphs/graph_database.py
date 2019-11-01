"""Database backend for GGNN graphs."""
import datetime
import pickle
import typing

import networkx as nx
import numpy as np
import sqlalchemy as sql
from labm8 import app
from labm8 import bazelutil
from labm8 import decorators
from labm8 import labdate
from labm8 import sqlutil
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl.graphs import graph_query as query
from deeplearning.ml4pl.graphs.labelled.graph_tuple import \
  graph_tuple as graph_tuples

FLAGS = app.FLAGS

EMBEDDINGS = bazelutil.DataPath(
    'phd/deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings/inst2vec_augmented_embeddings.pickle'
)

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                          nullable=False)


class GraphMeta(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A table of graph metadata.

  For every GraphMeta, there should be a corresponding Graph row containing the
  actual data as pickled blob.
  """
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string name to group graphs into discrete buckets, e.g. "train", "test",
  # or "1", "2", ... k for k-fold classification.
  group: str = sql.Column(sql.String(32), nullable=False, index=True)

  # No foreign key relationship because they are separate databases.
  bytecode_id: int = sql.Column(sql.Integer, nullable=False)

  # The source of the graph. This is duplicates of information stored in the
  # LlvmBytecode table.
  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False)

  node_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)
  node_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)
  edge_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)

  # The maximum value of the 'position' attribute of edges.
  edge_position_max: int = sql.Column(sql.Integer, nullable=False)

  node_labels_dimensionality: int = sql.Column(sql.Integer,
                                               default=0,
                                               nullable=False)
  graph_features_dimensionality: int = sql.Column(sql.Integer,
                                                  default=0,
                                                  nullable=False)
  graph_labels_dimensionality: int = sql.Column(sql.Integer,
                                                default=0,
                                                nullable=False)

  # The loop connectedness (loop depth) of the graph. This is the largest number
  # of back edges found in any cycle-free path of the full flow graph.
  loop_connectedness: int = sql.Column(sql.Integer, nullable=False)

  # The minimum number of message passing steps that are be required to produce
  # the labels from the features. E.g. for graph flooding problems, this value
  # will be the diameter of the graph.
  data_flow_max_steps_required: int = sql.Column(sql.Integer,
                                                 default=0,
                                                 nullable=False)

  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)

  graph: 'Graph' = sql.orm.relationship('Graph',
                                        uselist=False,
                                        back_populates="meta",
                                        cascade="all")

  @property
  def data(self) -> typing.Any:
    """Load the pickled data."""
    return self.graph.data

  @classmethod
  def CreateFromNetworkX(cls, g: nx.MultiDiGraph, **graph_tuple_opts):
    """Create a GraphMeta with a corresponding Graph containing a graph tuple.

    Args:
      g: The graph to convert to a GraphMeta. Must have the following attributes
       set: bytecode_id, source_name, relpath, language.
      graph_tuple_opts: Keyword argument to be passed to CreateFromNetworkX().

    Returns:
      A fully-populated GraphMeta instance.
    """
    graph_tuple = graph_tuples.GraphTuple.CreateFromNetworkX(
        g, **graph_tuple_opts)
    node_labels_dimensionality = (graph_tuple.node_y[0]
                                  if graph_tuple.has_node_y else 0)
    graph_features_dimensionality = (len(graph_tuple.graph_x)
                                     if graph_tuple.has_graph_x else 0)
    graph_labels_dimensionality = (len(graph_tuple.graph_y)
                                   if graph_tuple.has_graph_y else 0)

    data_flow_max_steps_required = getattr(g, 'data_flow_max_steps_required', 0)

    edge_position_max = 0
    for src, dst, position in g.edges(data='position', default=0):
      edge_position_max = max(edge_position_max, position)

    return GraphMeta(
        group=getattr(g, 'group', None),
        bytecode_id=g.bytecode_id,
        source_name=g.source_name,
        relpath=g.relpath,
        language=g.language,
        node_count=g.number_of_nodes(),
        # Get the edge stats *after* graph_tuple has inserted the
        # backward edges.
        edge_count=sum([len(a) for a in graph_tuple.adjacency_lists]),
        edge_type_count=len(graph_tuple.adjacency_lists),
        edge_position_max=edge_position_max,
        node_labels_dimensionality=node_labels_dimensionality,
        graph_features_dimensionality=graph_features_dimensionality,
        graph_labels_dimensionality=graph_labels_dimensionality,
        data_flow_max_steps_required=data_flow_max_steps_required,
        loop_connectedness=query.LoopConnetedness(g),
        graph=Graph.CreateFromPickled(graph_tuple))

  @classmethod
  def CreateFromGraphMetaAndGraphTuple(cls, graph_meta: 'GraphMeta',
                                       graph_tuple: graph_tuples.GraphTuple):
    """Create a GraphMeta with a corresponding Graph containing a graph tuple.

    Args:
      g: The graph to convert to a GraphMeta. Must have the following attributes
       set: bytecode_id, source_name, relpath, language.
      edge_types: The set of edge flow types, e.g. {"control", "flow"}, etc.
      graph_tuple_opts: Keyword argument to be passed to CreateFromNetworkX().

    Returns:
      A fully-populated GraphMeta instance.
    """
    return GraphMeta(
        group=graph_meta.group,
        bytecode_id=graph_meta.bytecode_id,
        source_name=graph_meta.source_name,
        relpath=graph_meta.relpath,
        language=graph_meta.language,
        node_count=graph_meta.node_count,
        edge_count=graph_meta.edge_count,
        edge_type_count=graph_meta.edge_type_count,
        edge_position_max=graph_meta.edge_position_max,
        node_labels_dimensionality=graph_meta.node_labels_dimensionality,
        graph_features_dimensionality=graph_meta.graph_features_dimensionality,
        graph_labels_dimensionality=graph_meta.graph_labels_dimensionality,
        data_flow_max_steps_required=graph_meta.data_flow_max_steps_required,
        loop_connectedness=graph_meta.loop_connectedness,
        graph=Graph.CreateFromPickled(graph_tuple))


class Graph(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for a graph.

  This is an opaque byte array that can be used as needed, e.g. for pickled
  graph tuples, networkx graphs, etc.
  """
  id: int = sql.Column(sql.Integer,
                       sql.ForeignKey('graph_metas.id'),
                       primary_key=True)
  pickled_data: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                   nullable=False)
  meta: GraphMeta = sql.orm.relationship('GraphMeta',
                                         back_populates="graph",
                                         uselist=False,
                                         cascade="all")

  @property
  def data(self) -> typing.Any:
    return pickle.loads(self.pickled_data)

  @classmethod
  def CreateFromPickled(cls, data: typing.Any) -> 'Graph':
    return Graph(pickled_data=pickle.dumps(data))


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)

  @decorators.memoized_property
  def embeddings_table(self) -> np.array:
    with open(EMBEDDINGS, 'rb') as f:
      return pickle.load(f)
