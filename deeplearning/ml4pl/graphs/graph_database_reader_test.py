"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database_reader."""
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as reader
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.TEST_DB_URLS)
def empty_graph_db(request) -> graph_database.Database:
  yield from testing_databases.YieldDatabase(
    graph_database.Database, request.param
  )


@test.Fixture(scope="function")
def graph_db_512(
  empty_graph_db: graph_database.Database,
) -> graph_database.Database:
  """Fixture which returns a database with 512 graphs, indexed by node_count."""

  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
      group="train",
      bytecode_id=1,
      source_name="foo",
      relpath="bar",
      language="c",
      node_count=i,
      edge_count=2,
      edge_position_max=0,
      loop_connectedness=0,
      undirected_diameter=0,
      data_flow_max_steps_required=i,
      graph=graph_database.Graph(
        pickled_data=pickle.dumps(np.ones(200000 // 4) * i)  # ~200KB of data
      ),
    )

  with empty_graph_db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return empty_graph_db


@test.Parametrize("buffer_size", [1, 25, 10000])
def test_BufferedGraphReader_values_in_order(
  graph_db_512: graph_database.Database, buffer_size: int
):
  """Test that the expected number of graphs are returned"""
  graphs = list(
    reader.BufferedGraphReader(graph_db_512, buffer_size=buffer_size)
  )
  assert len(graphs) == 510
  assert all([g.bytecode_id == 1 for g in graphs])
  # Check the graph node counts, offset by the first two which are ignored
  # (because graphs with zero or one nodes are filtered out).
  assert all([g.node_count == i + 2 for i, g in enumerate(graphs)])
  # Check that eager graph loading enables access to the graph data.
  assert all([g.data.shape == (200000 // 4,) for g in graphs])
  assert [all(g.data[0] == i + 2 for i, g in enumerate(graphs))]


@test.Parametrize("buffer_size", [1, 25, 10000])
@test.Parametrize(
  "order",
  [
    reader.BufferedGraphReaderOrder.IN_ORDER,
    reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
  ],
)
def test_BufferedGraphReader_filter(
  graph_db_512: graph_database.Database,
  buffer_size: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test using a filter callback."""
  filter_cb = lambda: graph_database.GraphMeta.node_count % 2 == 0
  graphs = list(
    reader.BufferedGraphReader(
      graph_db_512, filters=[filter_cb], buffer_size=buffer_size, order=order
    )
  )
  assert len(graphs) == 255


@test.Parametrize("buffer_size", [1, 25, 10000])
@test.Parametrize(
  "order",
  [
    reader.BufferedGraphReaderOrder.IN_ORDER,
    reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
  ],
)
def test_BufferedGraphReader_filters(
  graph_db_512: graph_database.Database,
  buffer_size: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test using multiple filters in combination."""
  filters = [
    lambda: graph_database.GraphMeta.node_count % 2 == 0,
    lambda: graph_database.GraphMeta.id < 256,
  ]
  graphs = list(
    reader.BufferedGraphReader(
      graph_db_512, filters=filters, buffer_size=buffer_size, order=order
    )
  )
  assert len(graphs) == 127


@test.Flaky(
  reason="There is a possibility that random order returns all rows in order"
)
@test.Parametrize("buffer_size", [25, 10000])
@test.Parametrize(
  "order",
  [
    reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    reader.BufferedGraphReaderOrder.BATCH_RANDOM,
  ],
)
def test_BufferedGraphReader_random_orders(
  graph_db_512: graph_database.Database,
  buffer_size: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test using `order_by_random` arg to randomize row order."""
  graphs = list(
    reader.BufferedGraphReader(
      graph_db_512, buffer_size=buffer_size, order=order
    )
  )
  node_counts = [g.node_count for g in graphs]
  assert sorted(node_counts) != node_counts


@test.Parametrize("buffer_size", [1, 25, 10000])
@test.Parametrize(
  "order",
  [
    reader.BufferedGraphReaderOrder.IN_ORDER,
    reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
  ],
)
@test.Parametrize("limit", [1, 25, 10000])
def test_BufferedGraphReader_limit(
  graph_db_512: graph_database.Database,
  buffer_size: int,
  order: reader.BufferedGraphReaderOrder,
  limit: int,
):
  """Test using `limit` arg to limit number of returned rows."""
  graphs = list(
    reader.BufferedGraphReader(
      graph_db_512, limit=limit, buffer_size=buffer_size, order=order
    )
  )
  assert len(graphs) == min(limit, 510)


@test.Parametrize("buffer_size", [1, 25, 10000])
@test.Parametrize(
  "order",
  [
    reader.BufferedGraphReaderOrder.IN_ORDER,
    reader.BufferedGraphReaderOrder.GLOBAL_RANDOM,
    reader.BufferedGraphReaderOrder.BATCH_RANDOM,
    reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
  ],
)
def test_BufferedGraphReader_next(
  graph_db_512: graph_database.Database,
  buffer_size: int,
  order: reader.BufferedGraphReaderOrder,
):
  """Test using next() to read from BufferedGraphReader()."""
  db_reader = reader.BufferedGraphReader(
    graph_db_512, buffer_size=buffer_size, order=order
  )
  for _ in range(510):
    next(db_reader)
  with test.Raises(StopIteration):
    next(db_reader)


@test.Parametrize("buffer_size", [1, 25, 10000])
def test_BufferedGraphReader_data_flow_max_steps_order(
  graph_db_512: graph_database.Database, buffer_size: int
):
  """Test that data flow max steps increases monotonically."""
  db_reader = reader.BufferedGraphReader(
    graph_db_512,
    buffer_size=buffer_size,
    order=reader.BufferedGraphReaderOrder.DATA_FLOW_MAX_STEPS_REQUIRED,
  )
  current_steps = -1
  i = 0
  for i, graph in enumerate(db_reader):
    # Sanity check that test fixture set expected values for data flow steps.
    assert graph.data_flow_max_steps_required == graph.node_count
    # Assert that data flow max steps is monotonically increasing.
    assert graph.data_flow_max_steps_required >= current_steps
    current_steps = graph.data_flow_max_steps_required
  # Sanity check that database contains graphs.
  assert i == 509


if __name__ == "__main__":
  test.Main()
