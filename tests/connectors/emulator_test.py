import asyncio
import pathlib
from typing import List, Optional

import fastapi
import httpx
import pytest
import pydantic
import typer
import typer.testing


import servo
import servo.connectors.emulator
import voracious.voracious
    
@pytest.fixture
def fastapi_app() -> fastapi.FastAPI:
    return voracious.voracious.app

async def test_list(fakeapi_client: httpx.AsyncClient) -> None:
    response = await fakeapi_client.get("/")
    response.raise_for_status()
    
    servo.logger.info(f"Loaded data: {response.text}\n")
    assert response.status_code == 200

async def test_describe() -> None:
    connector = servo.connectors.emulator.EmulatorConnector(config=servo.BaseConfiguration())
    debug(connector)
    response = await connector.describe()
    debug(response)

# cli = servo.cli.ConnectorCLI(servo.Servo.EmulatorConnector, help="Emulate servos for testing optimizers")

#################################################################

@pytest.fixture
def cli() -> typer.Typer:
    return servo.connectors.emulator.cli

@pytest.fixture
def token_path(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "token"
    path.write_text("123456789")
    return path

def test_help(cli: typer.Typer, cli_runner: typer.testing.CliRunner) -> None:
    result = typer.testing.CliRunner().invoke(cli, "--help", catch_exceptions=False)
    assert result.exit_code == 0
    assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout

# def test_list_optimizers(cli: typer.Typer, token_path: pathlib.Path, cli_runner: typer.testing.CliRunner) -> None:
#     result = typer.testing.CliRunner().invoke(cli, f"-f {token_path} list-optimizers", catch_exceptions=False)
#     debug(result.stdout, result.stderr)
#     assert result.exit_code == 0
#     assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout

# def test_create_optimizers(cli: typer.Typer, token_path: pathlib.Path, cli_runner: typer.testing.CliRunner) -> None:
#     result = typer.testing.CliRunner().invoke(cli, f"-f {token_path} create-optimizers", catch_exceptions=False)
#     debug(result.stdout, result.stderr)
#     assert result.exit_code == 0
#     assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout

# def test_delete_optimizers(cli: typer.Typer, token_path: pathlib.Path, cli_runner: typer.testing.CliRunner) -> None:
#     result = typer.testing.CliRunner().invoke(cli, f"-f {token_path} delete-optimizers", catch_exceptions=False)
#     debug(result.stdout, result.stderr)
#     assert result.exit_code == 0
#     assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout

# def test_scale_optimizers(cli: typer.Typer, token_path: pathlib.Path, cli_runner: typer.testing.CliRunner) -> None:
#     result = typer.testing.CliRunner().invoke(cli, f"-f {token_path} scale-optimizers", catch_exceptions=False)
#     debug(result.stdout)
#     assert result.exit_code == 0
#     assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout

# def test_get_template(cli: typer.Typer, token_path: pathlib.Path, cli_runner: typer.testing.CliRunner) -> None:
#     result = typer.testing.CliRunner().invoke(cli, f"-f {token_path} get-template", catch_exceptions=False)
#     debug(result.stdout, result.stderr)
#     assert result.exit_code == 0
#     assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout
    