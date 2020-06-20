import pytest
import typer
import os
import yaml
from pathlib import Path
from typer.testing import CliRunner
from pydantic import ValidationError
from servo.connector import Connector, VegetaSettings, VegetaConnector, License, Maturity, Version, TargetFormat
from typing import ClassVar, Union

# test subclass regisration
# test CLI integration
# test env var overrides
# test load from config file
# test aliasing

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

class TestVegetaSettings:
    # @pytest.fixture()
    # def settings(self) -> VegetaSettings:
    #     return VegetaSettings(rate='50/1s', duration='0')

    # def test_validate_rate(self) -> None:
    #     # 0 is infinity, otherwise requests/duration
        

    def test_validate_infinite_rate(self) -> None:
        s = VegetaSettings(rate='0', duration='0', target="GET http://example.com")
        assert s.rate == '0'

    def test_validate_rate_no_time_unit(self) -> None:
        s = VegetaSettings(rate='500', duration='0', target="GET http://example.com")
        assert s.rate == '500'
    
    def test_validate_rate_integer(self) -> None:
        s = VegetaSettings(rate=500, duration='0', target="GET http://example.com")
        assert s.rate == '500'
    
    def test_validate_rate_connections_over_time(self) -> None:
        s = VegetaSettings(rate='500/1s', duration='0', target="GET http://example.com")
        assert s.rate == '500/1s'
    
    def test_validate_rate_raises_when_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate='INVALID', duration='0', target="GET http://example.com")
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('rate',)
        assert e.value.errors()[0]['msg'] == 'rate strings are of the form hits/interval'
    
    def test_validate_rate_raises_when_invalid_duration(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate='500/1zxzczc', duration='0', target="GET http://example.com")
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('rate',)
        assert e.value.errors()[0]['msg'] == 'Unknown unit zxzczc in duration 1zxzczc'

    def test_validate_duration_infinite_attack(self) -> None:
        s = VegetaSettings(rate='0', duration='0', target="GET http://example.com")
        assert s.duration == '0'

    def test_validate_duration_seconds(self) -> None:
        s = VegetaSettings(rate='0', duration='1s', target="GET http://example.com")
        assert s.duration == '1s'

    def test_validate_duration_hours_minutes_and_seconds(self) -> None:
        s = VegetaSettings(rate='0', duration='1h35m20s', target="GET http://example.com")
        assert s.duration == '1h35m20s'

    def test_validate_duration_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate='0', duration='INVALID', target="GET http://example.com")
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('duration',)
        assert e.value.errors()[0]['msg'] == 'Invalid duration INVALID'
    
    def test_validate_target_with_http_format(self) -> None:
        s = VegetaSettings(rate='0', duration='0', format='http', target="GET http://example.com")
        assert s.format == TargetFormat.http

    # TODO: this will break once validator is written
    def test_validate_target_with_json_format(self) -> None:
        s = VegetaSettings(rate='0', duration='0', format='json', target="GET http://example.com")
        assert s.format == TargetFormat.json
    
    def test_validate_target_with_invalid_format(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate='0', duration='0', format='invalid', target="GET http://example.com")
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('format',)
        assert e.value.errors()[0]['msg'] == "value is not a valid enumeration member; permitted: 'http', 'json'"
    
    def test_validate_taget_or_targets_must_be_selected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as e:
            s = VegetaSettings(rate='0', duration='0')
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('__root__',)
        assert e.value.errors()[0]['msg'] == "target or targets must be configured"

    def test_validate_targets_with_path(self, tmp_path: Path) -> None:
        targets = tmp_path / 'targets'
        targets.touch()
        s = VegetaSettings(rate='0', duration='0', targets=targets)
        assert s.targets == targets

    def test_validate_targets_with_path_doesnt_exist(self, tmp_path: Path) -> None:
        targets = tmp_path / 'targets'
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate='0', duration='0', targets=targets)
        assert '1 validation error for VegetaSettings' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('targets',)
        assert 'file or directory at path' in e.value.errors()[0]['msg']
    
    # TODO: Test the combination of JSON and HTTP targets

class VegetaConnectorTests:
    pass

def test_init_vegeta_connector() -> None:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings)
    assert connector is not None

def test_init_vegeta_connector_no_settings() -> None:
    with pytest.raises(ValidationError) as e:
        VegetaConnector(None)
    assert '1 validation error for VegetaConnector' in str(e.value)

def test_init_connector_no_version_raises() -> None:
    class FakeConnector(Connector):
        pass
    with pytest.raises(ValidationError) as e:
        FakeConnector.version = None
        settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")        
        connector = FakeConnector(settings, id="whatever")
    assert e.value.errors()[0]['loc'] == ('__root__',)
    assert e.value.errors()[0]['msg'] == 'version must be provided'

def test_init_connector_invalid_version_raises() -> None:
    class FakeConnector(Connector):
        pass
    with pytest.raises(ValidationError) as e:
        FakeConnector.version = "invalid"
        settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
        connector = FakeConnector(settings, id="whatever", version="b")
    assert e.value.errors()[0]['loc'] == ('__root__',)
    assert e.value.errors()[0]['msg'] == 'invalid is not valid SemVer string'

def test_init_connector_parses_version_string() -> None:
    class FakeConnector(Connector):
        pass
    FakeConnector.version = "0.5.0"
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = FakeConnector(settings, id="whatever")
    assert connector.version is not None
    assert connector.version == Version.parse("0.5.0")

def test_init_connector_no_name_raises() -> None:
    class FakeConnector(Connector):
        pass
    with pytest.raises(ValidationError) as e:
        FakeConnector.name = None
        settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
        connector = FakeConnector(settings, id='test', name=None)
    assert e.value.errors()[0]['loc'] == ('__root__',)
    assert e.value.errors()[0]['msg'] == 'name must be provided'

def test_vegeta_default_id() -> None:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings)
    assert connector.id == 'vegeta'

def test_vegeta_id_override() -> None:
    settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
    connector = VegetaConnector(settings, id="monkey")
    assert connector.id == 'monkey'

def test_vegeta_id_invalid() -> None:
    with pytest.raises(ValidationError) as e:
        settings = VegetaSettings(rate="50/1s", duration="5m", target="GET http://localhost:8080")
        connector = VegetaConnector(settings, id="THIS IS NOT COOL")
    assert '1 validation error for VegetaConnector' in str(e.value)
    assert e.value.errors()[0]['msg'] == 'id may only contain lowercase alphanumeric characters and underscores'

def test_vegeta_name() -> None:
    assert VegetaConnector.name == 'Vegeta Connector'

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

def test_vegeta_cli_info(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "info")
    assert result.exit_code == 0
    assert (
        "Vegeta Connector v0.5.0 (Stable)\n"
        "Vegeta load testing connector\n"
        "https://github.com/opsani/vegeta-connector\n"
        "Licensed under the terms of Apache 2.0\n"
    ) in result.stdout

def test_vegeta_cli_version(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "version")
    assert result.exit_code == 0
    assert "Vegeta Connector v0.5.0" in result.stdout

def test_vegeta_cli_loadgen(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    pass
