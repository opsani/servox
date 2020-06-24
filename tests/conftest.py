import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

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


@contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env


@pytest.fixture()
def servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    config_path.touch()
    return config_path
