import os
from pathlib import Path

import pytest
from typer.testing import CliRunner
from servo.cli import ServoCLI

# Force the test connectors to load early
from tests.test_helpers import MeasureConnector, AdjustConnector, LoadgenConnector

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

@pytest.fixture()
def servo_cli() -> ServoCLI:
    return ServoCLI()

@pytest.fixture()
def optimizer_env() -> None:
    os.environ.update({ "OPSANI_OPTIMIZER": "dev.opsani.com/servox", "OPSANI_TOKEN": "123456789" })

@pytest.fixture()
def servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    config_path.touch()
    return config_path


# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)
