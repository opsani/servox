import pytest
import typer
import os
import yaml
from pathlib import Path
from typer.testing import CliRunner
from servo.connector import VegetaSettings, VegetaConnector, License, Maturity

# test subclass regisration
# test CLI integration
# test env var overrides
# test load from config file
# test aliasing
# test

class OptimizerTests:

    def test_org_domain_validation(self) -> None:
        pass

    def test_app_name_validation(self) -> None:
        pass

    def test_token_validation(self) -> None:
        pass

    def test_base_url_validation(self) -> None:
        pass

class ConnectorSettingsTests:
    pass

class LicenseTests:
    def test_license_from_string(self):
        pass

    def test_license_from_string_invalid_raises(self):
        pass

class MaturityTests:
    def test_license_from_string(self):
        pass

    def test_license_from_string_invalid_raises(self):
        pass

class ConnectorTests:
    pass

    # Register subclass
    # API client, logger
    # default ID behavior

class ServoSettingsTests:
    pass

class ServoTests:
    pass

###
### Connector specific
###

class VegetaSettingsTests:
    pass

class VegetaConnectorTests:
    pass

# TODO: This thing needs settings and an optimizer
# test id
# test default id
# test no settings
# test no version
def test_init_vegeta_connector() -> None:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings)
    assert connector is not None

def test_vegeta_default_id() -> None:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings)
    assert connector.id == 'vegeta'

def test_vegeta_name() -> None:
    assert VegetaConnector.name == 'Vegeta'

def test_vegeta_description() -> None:
    assert VegetaConnector.description == 'Vegeta load testing connector'

def test_vegeta_version() -> None:
    # TODO: Type violation
    assert VegetaConnector.version == '0.5.0'

def test_vegeta_homepage() -> None:
    # TODO: Type violation
    assert VegetaConnector.homepage == 'https://github.com/opsani/vegeta-connector'

def test_vegeta_license() -> None:
    assert VegetaConnector.license == License.APACHE2

def test_vegeta_maturity() -> None:
    assert VegetaConnector.maturity == Maturity.STABLE

@pytest.fixture()
def vegeta_cli() -> typer.Typer:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings)
    return connector.cli()

## Vegeta CLI tests
def test_vegeta_cli_help(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "--help")
    assert result.exit_code == 0
    assert "Usage: vegeta [OPTIONS] COMMAND [ARGS]..." in result.stdout

def test_vegeta_cli_schema(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "schema")
    assert result.exit_code == 0
    assert result.stdout == ('{\n'
        '  "title": "VegetaSettings",\n'
        '  "description": "Configuration of the Vegeta connector",\n'
        '  "type": "object",\n'
        '  "properties": {\n'
        '    "description": {\n'
        '      "title": "Description",\n'
        '      "env_names": [\n'
        '        "servo_description"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "rate": {\n'
        '      "title": "Rate",\n'
        '      "description": "Specifies the request rate per time unit to issue against the targets. Given in the for'
        'mat of request/time unit.",\n'
        '      "env_names": [\n'
        '        "servo_rate"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "duration": {\n'
        '      "title": "Duration",\n'
        '      "description": "Specifies the amount of time to issue requests to the targets.",\n'
        '      "env_names": [\n'
        '        "servo_duration"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "format": {\n'
        '      "title": "Format",\n'
        '      "description": "Specifies the format of the targets input. Valid values are http and json. Refer to the'
        ' Vegeta docs for details.",\n'
        '      "default": "http",\n'
        '      "env_names": [\n'
        '        "servo_format"\n'
        '      ],\n'
        '      "enum": [\n'
        '        "http",\n'
        '        "json"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "target": {\n'
        '      "title": "Target",\n'
        '      "description": "Specifies a single formatted Vegeta target to load. See the format option to learn abou'
        't available target formats. This option is exclusive of the targets option and will provide a target to Veget'
        'a via stdin.",\n'
        '      "env_names": [\n'
        '        "servo_target"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "targets": {\n'
        '      "title": "Targets",\n'
        '      "description": "Specifies the file from which to read targets. See the format option to learn about ava'
        'ilable target formats. This option is exclusive of the target option and will provide targets to via through '
        'a file on disk.",\n'
        '      "default": "stdin",\n'
        '      "env_names": [\n'
        '        "servo_targets"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    },\n'
        '    "connections": {\n'
        '      "title": "Connections",\n'
        '      "description": "Specifies the maximum number of idle open connections per target host.",\n'
        '      "default": 10000,\n'
        '      "env_names": [\n'
        '        "servo_connections"\n'
        '      ],\n'
        '      "type": "integer"\n'
        '    },\n'
        '    "workers": {\n'
        '      "title": "Workers",\n'
        '      "description": "Specifies the initial number of workers used in the attack. The workers will automatica'
        'lly increase to achieve the target request rate, up to max-workers.",\n'
        '      "default": 10,\n'
        '      "env_names": [\n'
        '        "servo_workers"\n'
        '      ],\n'
        '      "type": "integer"\n'
        '    },\n'
        '    "max-workers": {\n'
        '      "title": "Max-Workers",\n'
        '      "description": "The maximum number of workers used to sustain the attack. This can be used to control t'
        'he concurrency of the attack to simulate a target number of clients.",\n'
        '      "default": 18446744073709551615,\n'
        '      "env_names": [\n'
        '        "servo_max_workers"\n'
        '      ],\n'
        '      "type": "integer"\n'
        '    },\n'
        '    "max-body": {\n'
        '      "title": "Max-Body",\n'
        '      "description": "Specifies the maximum number of bytes to capture from the body of each response. Remain'
        'ing unread bytes will be fully read but discarded.",\n'
        '      "default": -1,\n'
        '      "env_names": [\n'
        '        "servo_max_body"\n'
        '      ],\n'
        '      "type": "integer"\n'
        '    },\n'
        '    "http2": {\n'
        '      "title": "Http2",\n'
        '      "description": "Specifies whether to enable HTTP/2 requests to servers which support it.",\n'
        '      "default": true,\n'
        '      "env_names": [\n'
        '        "servo_http2"\n'
        '      ],\n'
        '      "type": "boolean"\n'
        '    },\n'
        '    "keepalive": {\n'
        '      "title": "Keepalive",\n'
        '      "description": "Specifies whether to reuse TCP connections between HTTP requests.",\n'
        '      "default": true,\n'
        '      "env_names": [\n'
        '        "servo_keepalive"\n'
        '      ],\n'
        '      "type": "boolean"\n'
        '    },\n'
        '    "insecure": {\n'
        '      "title": "Insecure",\n'
        '      "description": "Specifies whether to ignore invalid server TLS certificates.",\n'
        '      "default": false,\n'
        '      "env_names": [\n'
        '        "servo_insecure"\n'
        '      ],\n'
        '      "type": "boolean"\n'
        '    }\n'
        '  },\n'
        '  "required": [\n'
        '    "rate",\n'
        '    "duration",\n'
        '    "target"\n'
        '  ],\n'
        '  "additionalProperties": false\n'
        '}\n'
    )

# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)

def test_vegeta_cli_generate(tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "generate")
    assert result.exit_code == 0
    assert "Generated vegeta.yaml" in result.stdout
    config_file = tmp_path / 'vegeta.yaml'
    config = config_file.read_text()
    assert config == (
        'connections: 10000\n'
        'description: null\n'
        'duration: 5m\n'
        'format: http\n'
        'http2: true\n'
        'insecure: false\n'
        'keepalive: true\n'
        'max-body: -1\n'
        'max-workers: 18446744073709551615\n'
        'rate: 50/1s\n'
        'target: GET http://localhost:8080\n'
        'targets: stdin\n'
        'workers: 10\n'
    )

def test_vegeta_cli_validate(tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    config_file = tmp_path / 'vegeta.yaml'
    config_file.write_text((
        'connections: 10000\n'
        'description: null\n'
        'duration: 5m\n'
        'format: http\n'
        'http2: true\n'
        'insecure: false\n'
        'keepalive: true\n'
        'max-body: -1\n'
        'max-workers: 18446744073709551615\n'
        'rate: 50/1s\n'
        'target: GET http://localhost:8080\n'
        'targets: stdin\n'
        'workers: 10\n'
    ))
    result = cli_runner.invoke(vegeta_cli, "validate vegeta.yaml")
    assert result.exit_code == 0
    assert "√ Valid connector configuration" in result.stdout

def test_vegeta_cli_validate_no_such_file(tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "validate doesntexist.yaml")
    assert result.exit_code == 2
    assert "Could not open file: doesntexist.yaml" in result.stderr

def test_vegeta_cli_validate_invalid_config(tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    config_file = tmp_path / 'invalid.yaml'
    config_file.write_text((
        'connections: 10000\n'
        'description: null\n'
        # 'duration: 5m\n'  # Duration is omitted
        'format: http\n'
        'http2: true\n'
        'insecure: false\n'
        'keepalive: true\n'
        'max-body: -1\n'
        'max-workers: 18446744073709551615\n'
        #'rate: 50/1s\n'  # Rate is omitted
        'target: GET http://localhost:8080\n'
        'targets: stdin\n'
        'workers: 10\n'
    ))
    result = cli_runner.invoke(vegeta_cli, "validate invalid.yaml")
    assert result.exit_code == 1
    assert "2 validation errors for VegetaSettings" in result.stderr

def test_vegeta_cli_validate_invalid_syntax(tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    config_file = tmp_path / 'invalid.yaml'
    config_file.write_text((
        'connections: 10000\n'
        'descriptions\n\n null\n'
        'duratio\n\n_   n: 5m\n'
    ))
    result = cli_runner.invoke(vegeta_cli, "validate invalid.yaml")
    assert result.exit_code == 1
    assert "X Invalid connector configuration" in result.stderr
    assert "could not find expected ':'" in result.stderr

# TODO: absolute path

def test_vegeta_cli_info(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    pass

def test_vegeta_cli_version(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    pass

def test_vegeta_cli_loadgen(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    pass
