import json
import os
import random
import string
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from servo.configuration import Optimizer
from servo.cli import ServoCLI
# Force the test connectors to load early
from tests.test_helpers import StubBaseConfiguration

# Add the devtools debug() function globally in tests
try:
    import builtins

    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug


# Set asyncio as a default marker across the suite
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker('asyncio')


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def servo_cli() -> ServoCLI:
    return ServoCLI()


@pytest.fixture()
def optimizer_env() -> None:
    os.environ.update(
        {"OPSANI_OPTIMIZER": "dev.opsani.com/servox", "OPSANI_TOKEN": "123456789"}
    )
    yield
    os.environ.pop("OPSANI_OPTIMIZER", None)
    os.environ.pop("OPSANI_TOKEN", None)


@pytest.fixture()
def optimizer() -> Optimizer:
    return Optimizer(id="dev.opsani.com/servox", token="123456789")

@pytest.fixture()
def servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    config_path.touch()
    return config_path


@pytest.fixture()
def stub_servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    settings = StubBaseConfiguration(name="stub")
    measure_config_json = json.loads(json.dumps(settings.dict(by_alias=True,)))
    config = {"connectors": ["measure"], "measure": measure_config_json}
    config = yaml.dump(config)
    config_path.write_text(config)
    return config_path


# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)


# Ensure that we don't have configuration bleeding into tests
@pytest.fixture(autouse=True)
def run_in_clean_environment() -> None:
    for key, value in os.environ.items():
        if key.startswith("SERVO_") or key.startswith("OPSANI_"):
            os.environ.pop(key)


@pytest.fixture()
def random_string() -> str:
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(32))
