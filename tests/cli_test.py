import pytest
import os
import time
from pathlib import Path
from typer import Typer
from typer.testing import CliRunner
from servo import cli
from pydantic import ValidationError

# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)

@pytest.fixture()
def servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / 'servo.yaml'
    config_path.touch()
    return config_path

@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)

@pytest.fixture()
def cli_app() -> Typer:
    return cli.app

def test_help() -> None:
    result = cli_runner.invoke(cli_app, "--help")
    assert result.exit_code == 0
    assert "servox db [OPTIONS] COMMAND [ARGS]" in result.stdout

def test_new() -> None:
    """Creates a new servo assembly at [PATH]"""
    pass

def test_run() -> None:
    """Run the servo"""
    pass

def test_console() -> None:
    """Open an interactive console"""
    pass

def test_info() -> None:
    pass

def test_check() -> None:
    pass

def test_version() -> None:
    pass

## TODO: moves to config.py
def test_config() -> None:
    """Display servo configuration"""
    pass

def test_config_schema() -> None:
    pass

def test_config_validate() -> None:
    """Validate servo configuration file"""
    pass

def test_config_generate() -> None:
    """Generate servo configuration"""
    pass

def test_connectors() -> None:
    pass

def test_connectors_add() -> None:
    pass

def test_connectors_remove() -> None:
    pass

## TODO: Moves to developer.py
def test_developer_test() -> None:
    pass

def test_developer_lint() -> None:
    pass

def test_developer_format() -> None:
    pass


## TODO: Dynamic command namespacing and registration
