import pytest
from typer.testing import CliRunner

# Add the devtools debug() function globally in tests
try:
    import builtins
    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)
