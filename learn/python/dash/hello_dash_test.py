"""Smoke test for //learn/python/dash:hello_dash."""
import dash
import portpicker

from labm8 import app
from labm8 import decorators
from labm8 import test
from learn.python.dash import hello_dash

FLAGS = app.FLAGS


@decorators.timeout_without_exception(seconds=5)
def _RunDashApp(app: dash.Dash) -> None:
  """Run dash app server for 10 seconds."""
  app.RunWithArgs_server(debug=True, port=portpicker.pick_unused_port())


def test_CreateApp_server_smoke_test():
  """Test that dash server runs without blowing up."""
  _RunDashApp(hello_dash.CreateApp())


if __name__ == '__main__':
  test.Main()
