import json
import os
import re
from pathlib import Path

import pytest
import yaml
from typer import Typer
from typer.testing import CliRunner

from servo import cli
from servo.cli import CLI, ServoCLI
from servo.servo import BaseServoSettings
from servo.connector import ConnectorSettings, Optimizer

@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)

@pytest.fixture()
def optimizer() -> Optimizer:
    return Optimizer("dev.opsani.com/servox", token="123456789")

@pytest.fixture()
def cli_app() -> ServoCLI:
    return ServoCLI()


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
    assert "servo [OPTIONS] COMMAND [ARGS]" in result.stdout


def test_new(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Creates a new servo assembly at [PATH]"""


def test_run(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Run the servo"""


def test_console(cli_runner: CliRunner, cli_app: Typer) -> None:
    """Open an interactive console"""


def test_connectors(cli_runner: CliRunner, cli_app: Typer, optimizer_env: None) -> None:
    result = cli_runner.invoke(cli_app, "connectors", catch_exceptions=False)
    assert result.exit_code == 0
    assert re.match("NAME\\s+VERSION\\s+DESCRIPTION\n", result.stdout)


def test_connectors_all(cli_runner: CliRunner, cli_app: Typer, optimizer_env: None) -> None:
    result = cli_runner.invoke(cli_app, "connectors --all")
    assert result.exit_code == 0
    assert re.match("^NAME\\s+VERSION\\s+DESCRIPTION\n", result.stdout)


def test_connectors_verbose(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path, optimizer_env: None
) -> None:
    result = cli_runner.invoke(cli_app, "connectors -v")
    assert result.exit_code == 0
    assert re.match(
        "NAME\\s+VERSION\\s+DESCRIPTION\\s+HOMEPAGE\\s+MATURITY\\s+LICENSE",
        result.stdout,
    )


def test_connectors_all_verbose(cli_runner: CliRunner, cli_app: Typer, optimizer_env: None) -> None:
    result = cli_runner.invoke(cli_app, "connectors --all -v")
    assert result.exit_code == 0
    assert re.match(
        "NAME\\s+VERSION\\s+DESCRIPTION\\s+HOMEPAGE\\s+MATUR", result.stdout
    )


def test_check(cli_runner: CliRunner, cli_app: Typer) -> None:
    pass


def test_version(cli_runner: CliRunner, cli_app: Typer, optimizer_env: None) -> None:
    result = cli_runner.invoke(cli_app, "version")
    assert result.exit_code == 0
    assert "Servo v0.0.0" in result.stdout


def test_settings(
    cli_runner: CliRunner, cli_app: Typer, vegeta_config_file: Path, optimizer_env: None,
) -> None:
    result = cli_runner.invoke(cli_app, "settings")
    assert result.exit_code == 0
    assert "connectors:" in result.stdout


def test_run_with_empty_config_file(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path, optimizer_env: None,
) -> None:
    result = cli_runner.invoke(cli_app, "settings", catch_exceptions=False)
    assert result.exit_code == 0, f"RESULT: {result.stderr}"
    assert "{}" in result.stdout


def test_run_with_malformed_config_file(
    cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path, optimizer_env: None,
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

def table_test_command_options(cli_runner: CliRunner, cli_app: Typer) -> None:
    commands = [
        version,
        schema,
        settings,
        generate,
        validate,
        events,
        describe,
        check,
        measure,
        adjust,
        promote
    ]

    settings = ConnectorSettings.construct()
    connector = MeasureConnector.construct(settings)
    for value in (True, False):
        kwargs = dict.fromkeys(commands, value)
        cli = ConnectorCLI(name="tester", **kwargs)
        for command in commands:
            result = cli_runner.invoke(cli, command)
            if value:
                # TODO: Ask Click for the command?
                assert result.exit_code == 0, f"Expected {command} to return a zero exit code but got {result.exit_code}"
            else:
                assert result.exit_code == 1, f"Expected {command} to return a non-zero exit code but got {result.exit_code}"
            

# Test name and help

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
    assert schema["title"] == "Servo Configuration Schema"


def test_schema_output_to_file(
    cli_runner: CliRunner, cli_app: Typer, tmp_path: Path
) -> None:
    output_path = tmp_path / "schema.json"
    result = cli_runner.invoke(cli_app, f"schema -f json -o {output_path}")
    assert result.exit_code == 0
    schema = json.loads(output_path.read_text())
    assert schema["title"] == "Servo Configuration Schema"


def test_schema_all(cli_runner: CliRunner, cli_app: Typer) -> None:
    result = cli_runner.invoke(cli_app, ["schema", "--all"])
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["title"] == "Servo Configuration Schema"


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

# TODO: This needs to be scoped to a class to not bother all other tests
class TestCommands:
    @pytest.fixture(autouse=True)
    def test_set_defaults_via_env(self) -> None:
        os.environ["OPSANI_OPTIMIZER"] = "dev.opsani.com/test-app"
        os.environ["OPSANI_TOKEN"] = "123456789"


    def test_schema_text(self, cli_app: Typer, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli_app, "schema -f text")
        assert result.exit_code == 1
        assert "not yet implemented" in result.stderr


    def test_schema_html(self, cli_app: Typer, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli_app, "schema -f html", catch_exceptions=False)
        assert result.exit_code == 1
        assert "not yet implemented" in result.stderr


    def test_schema_dict(self, cli_app: Typer, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(cli_app, "schema -f dict")
        assert result.exit_code == 0
        dict = eval(result.stdout)
        assert dict["title"] == "Servo Configuration Schema"


    def test_schema_dict_file_output(self, 
        cli_app: Typer, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        path = tmp_path / "output.dict"
        result = cli_runner.invoke(cli_app, f"schema -f dict -o {path}")
        assert result.exit_code == 0
        dict = eval(path.read_text())
        assert dict["title"] == "Servo Configuration Schema"


    def test_validate(self, cli_runner: CliRunner, cli_app: Typer) -> None:
        """Validate servo configuration file"""


    def test_generate(self, cli_runner: CliRunner, cli_app: Typer) -> None:
        """Generate servo configuration"""
        # TODO: Generate this thing in test dir


    def test_developer_test(self, cli_runner: CliRunner, cli_app: Typer) -> None:
        pass


    def test_developer_lint(self, cli_runner: CliRunner, cli_app: Typer) -> None:
        pass


    def test_developer_format(self, cli_runner: CliRunner, cli_app: Typer) -> None:
        pass


    ## CLI Lifecycle tests


    def test_loading_cli_without_specific_connectors_activates_all_optionally(
        self, cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
    ) -> None:
        # temp config file, no connectors key
        pass


    def test_loading_cli_with_specific_connectors_only_activates_required(
        self, cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
    ) -> None:
        pass


    def test_loading_cli_with_empty_connectors_list_disables_all(
        self, cli_runner: CliRunner, cli_app: Typer, servo_yaml: Path
    ) -> None:
        servo_yaml.write_text(yaml.dump({"connectors": []}))
        cli_runner.invoke(cli_app, "info")

from servo.cli import CLI, Context

class TestCLIFoundation():
    class TheTestCLI(CLI):
        pass

    @pytest.fixture()
    def cli(self) -> CLI:
        # TODO: Set the callback. We may need to be able to tell this apart
        return TestCLIFoundation.TheTestCLI(help="This is just a test.", callback=None)

    def test_context_class_in_commands(self, cli: CLI, cli_runner: CliRunner) -> None:
        @cli.command()
        def test(context: Context) -> None:
            assert context is not None
            assert isinstance(context, Context)
            assert context.servo is None
            assert context.assembly is None
            assert context.connector is None

        result = cli_runner.invoke(cli, "", catch_exceptions=False)
        assert result.exit_code == 0

    def test_context_class_in_subcommand_groups(self, cli: CLI, cli_runner: CliRunner) -> None:
        sub_cli = TestCLIFoundation.TheTestCLI(name="another", callback=None)
        @sub_cli.command()
        def test(context: Context) -> None:
            assert context is not None
            assert isinstance(context, Context)
            assert context.servo == None
            assert context.assembly == None
            assert context.connector == None
        
        cli.add_cli(sub_cli)

        result = cli_runner.invoke(cli, "another test", catch_exceptions=False)
        assert result.exit_code == 0
    
    def test_context_inheritance(self, cli: CLI, cli_runner: CliRunner) -> None:
        sub_cli = TestCLIFoundation.TheTestCLI(name="another", callback=None)

        @cli.callback()
        def touch_context(context: Context) -> None:
            context.obj = 31337

        @sub_cli.command()
        def test(context: Context) -> None:
            assert context is not None
            assert context.obj == 31337
        
        cli.add_cli(sub_cli)

        result = cli_runner.invoke(cli, "another test", catch_exceptions=False)
        assert result.exit_code == 0
    
    def test_context_state_for_base_callback(self) -> None:
        # TODO: config file path, optimizer settings
        pass

    def test_context_state_for_servo_callback(self) -> None:
        # TODO: Full servo assembly, check the state -- no connector hydration
        pass

    def test_context_state_for_connector_callback(self) -> None:
        # TODO: Full servo assembly, connector is set to the target
        pass
    
    # TODO: Target arbitrary number of connectors
    def test_context_state_for_connectors_callback(self) -> None:
        # TODO: Full servo assembly, connectors is set to the targets
        pass
    
    def test_that_servo_cli_commands_are_explicitly_ordered(self, cli: CLI, cli_runner: CliRunner) -> None:
        servo_cli = ServoCLI(name="servo", callback=None)

        # Add in explicit non lexical sort order
        @servo_cli.command()
        def zzzz(context: Context) -> None:
            pass
        
        @servo_cli.command()
        def aaaa(context: Context) -> None:
            pass

        @servo_cli.command()
        def mmmm(context: Context) -> None:
            pass

        result = cli_runner.invoke(servo_cli, "--help", catch_exceptions=False)
        assert result.exit_code == 0
        assert re.search(r"zzzz\n.*aaaa\n.*mmmm\n", result.stdout, flags=re.MULTILINE) is not None
    
    # TODO: test errors with callbacks when run with incomplete configurations
    # TODO: Specifically config file doesn't exist, malformed, etc. No optimizer...
