import pytest
import os
import time
import json
from pathlib import Path
from typer import Typer
from typer.testing import CliRunner
from servo import cli
from pydantic import ValidationError
from servo.connector import Connector, Optimizer, ServoSettings, Servo

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

def test_help(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "--help")
    assert result.exit_code == 0
    assert "servox [OPTIONS] COMMAND [ARGS]" in result.stdout

def test_new(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Creates a new servo assembly at [PATH]"""
    pass

def test_run(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Run the servo"""
    pass

def test_console(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Open an interactive console"""
    pass

def test_info(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "info")
    assert result.exit_code == 0
    assert "NAME              VERSION    DESCRIPTION\n" in result.stdout

def test_info_verbose(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "info -v")
    assert result.exit_code == 0
    assert "NAME              VERSION    DESCRIPTION                           HOMEPAGE                                    MATURI" in result.stdout

def test_check(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass

def test_version(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "version")
    assert result.exit_code == 0
    assert "Servo v0.0.0" in result.stdout

def test_settings(cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path) -> None:
    servo_yaml.write_text("connectors: []")
    result = cli_runner.invoke(cli_app, "settings")
    assert result.exit_code == 0
    assert "connectors:" in result.stdout

def test_schema(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "schema")
    assert result.exit_code == 0    
    schema = json.loads(result.stdout)
    assert schema['title'] == 'Servo'

def test_schema_all(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ['schema', '--all'])
    assert result.exit_code == 0    
    schema = json.loads(result.stdout)
    assert schema['title'] == 'Servo'

def test_schema_top_level(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ['schema', '--top-level'])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema['title'] == 'Servo Schema'

def test_schema_all_top_level(cli_runner: CliRunner, cli_app: Typer) -> None:    
    result = cli_runner.invoke(cli_app, ['schema', '--top-level', '--all'])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema['title'] == 'Servo Schema'

def test_validate(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Validate servo configuration file"""
    pass

def test_generate(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Generate servo configuration"""
    pass

def test_developer_test(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass

def test_developer_lint(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass

def test_developer_format(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass
