import json
import os
import re
from pathlib import Path

import pytest
import yaml
from typer import Typer
from typer.testing import CliRunner

from servo import cli


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def cli_app() -> Typer:
    return cli.cli


@pytest.fixture()
def vegeta_config_file(servo_yaml: Path) -> Path:
    config = {
        "connectors": ["vegeta"],
        "vegeta": {"duration": 0, "rate": 0, "target": "https://opsani.com/"},
    }
    servo_yaml.write_text(yaml.dump(config))
    return servo_yaml


def test_help(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "--help")
    assert result.exit_code == 0
    assert "servox [OPTIONS] COMMAND [ARGS]" in result.stdout


def test_new(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Creates a new servo assembly at [PATH]"""


def test_run(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Run the servo"""


def test_console(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Open an interactive console"""


def test_connectors(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "connectors", catch_exceptions=False)
    assert result.exit_code == 0
    assert re.match("NAME\\s+VERSION\\s+DESCRIPTION\n", result.stdout)


def test_connectors_all(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "connectors --all")
    assert result.exit_code == 0
    assert re.match("^NAME\\s+VERSION\\s+DESCRIPTION\n", result.stdout)


def test_connectors_verbose(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path
) -> None:
    result = cli_runner.invoke(cli_app, "connectors -v")
    assert result.exit_code == 0
    assert re.match(
        "NAME\\s+VERSION\\s+DESCRIPTION\\s+HOMEPAGE\\s+MATURITY\\s+LICENSE",
        result.stdout,
    )


def test_connectors_all_verbose(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "connectors --all -v")
    assert result.exit_code == 0
    assert re.match(
        "NAME\\s+VERSION\\s+DESCRIPTION\\s+HOMEPAGE\\s+MATUR", result.stdout
    )


def test_check(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass


def test_version(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "version")
    assert result.exit_code == 0
    assert "Servo v0.0.0" in result.stdout


def test_settings(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path
) -> None:
    result = cli_runner.invoke(cli_app, "settings")
    assert result.exit_code == 0
    assert "connectors:" in result.stdout


def test_run_with_empty_config_file(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
) -> None:
    result = cli_runner.invoke(cli_app, "settings", catch_exceptions=False)
    assert result.exit_code == 0
    assert "{}" in result.stdout


def test_run_with_malformed_config_file(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
) -> None:
    servo_yaml.write_text("</\n\n..:989890j\n___*")
    with pytest.raises(ValueError) as e:
        cli_runner.invoke(cli_app, "settings", catch_exceptions=False)
    assert "parsed to an unexpected value of type" in str(e)


def test_settings_yaml(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path
) -> None:
    result = cli_runner.invoke(cli_app, "settings -f yaml", catch_exceptions=False)
    assert result.exit_code == 0
    assert "connectors:" in result.stdout


def test_settings_yaml_file(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path, tmp_path: Path
) -> None:
    path = tmp_path / "settings.yaml"
    result = cli_runner.invoke(cli_app, f"settings -f yaml -o {path}")
    assert result.exit_code == 0
    assert "connectors:" in path.read_text()


def test_settings_json(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path
) -> None:
    result = cli_runner.invoke(cli_app, "settings -f json")
    assert result.exit_code == 0
    settings = json.loads(result.stdout)
    assert settings["connectors"] is not None


def test_settings_json_file(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path, tmp_path: Path
) -> None:
    path = tmp_path / "settings.json"
    result = cli_runner.invoke(cli_app, f"settings -f json -o {path}")
    assert result.exit_code == 0
    settings = json.loads(path.read_text())
    assert settings["connectors"] is not None


def test_settings_dict(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path
) -> None:
    result = cli_runner.invoke(cli_app, "settings -f dict")
    assert result.exit_code == 0
    settings = eval(result.stdout)
    assert settings["connectors"] is not None


def test_settings_dict_file(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path, tmp_path: Path
) -> None:    
    path = tmp_path / "settings.py"
    result = cli_runner.invoke(cli_app, f"settings -f dict -o {path}")
    assert result.exit_code == 0
    settings = eval(path.read_text())
    assert settings["connectors"] is not None


def test_schema(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, "schema", catch_exceptions=False)
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "Servo"


def test_schema_output_to_file(
    cli_runner: CliRunner, cli_app: Typer, tmp_path: Path
) -> None:
    output_path = tmp_path / "schema.json"
    result = cli_runner.invoke(cli_app, f"schema -f json -o {output_path}")
    assert result.exit_code == 0
    schema = json.loads(output_path.read_text())
    assert schema["title"] == "Servo"


def test_schema_all(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ["schema", "--all"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "Servo"


def test_schema_top_level(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ["schema", "--top-level"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "Servo Schema"


def test_schema_all_top_level(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ["schema", "--top-level", "--all"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "Servo Schema"


def test_schema_top_level_dict(cli_app: Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli_app, "schema -f dict --top-level")
    assert result.exit_code == 0
    schema = eval(result.stdout)
    assert schema["title"] == "Servo Schema"


def test_schema_top_level_dict_file_output(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
) -> None:
    path = tmp_path / "output.dict"
    result = cli_runner.invoke(cli_app, f"schema -f dict --top-level -o {path}")
    assert result.exit_code == 0
    schema = eval(path.read_text())
    assert schema["title"] == "Servo Schema"


@pytest.fixture(autouse=True)
def test_set_defaults_via_env() -> None:
    os.environ["OPSANI_OPTIMIZER"] = "dev.opsani.com/test-app"
    os.environ["OPSANI_TOKEN"] = "123456789"


def test_schema_text(cli_app: Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli_app, "schema -f text")
    assert result.exit_code == 1
    assert "not yet implemented" in result.stderr


def test_schema_html(cli_app: Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli_app, "schema -f html", catch_exceptions=False)
    assert result.exit_code == 1
    assert "not yet implemented" in result.stderr


def test_schema_dict(cli_app: Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli_app, "schema -f dict")
    assert result.exit_code == 0
    dict = eval(result.stdout)
    assert dict["title"] == "Servo"


def test_schema_dict_file_output(
    cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
) -> None:
    path = tmp_path / "output.dict"
    result = cli_runner.invoke(cli_app, f"schema -f dict -o {path}")
    assert result.exit_code == 0
    dict = eval(path.read_text())
    assert dict["title"] == "Servo"


def test_validate(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Validate servo configuration file"""


def test_generate(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Generate servo configuration"""
    # TODO: Generate this thing in test dir


def test_developer_test(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass


def test_developer_lint(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass


def test_developer_format(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass


## CLI Lifecycle tests


def test_loading_cli_without_specific_connectors_activates_all_optionally(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
) -> None:
    # temp config file, no connectors key
    pass


def test_loading_cli_with_specific_connectors_only_activates_required(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
) -> None:
    pass


def test_loading_cli_with_empty_connectors_list_disables_all(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
) -> None:
    servo_yaml.write_text(yaml.dump({"connectors": []}))
    cli_runner.invoke(cli_app, "info")
