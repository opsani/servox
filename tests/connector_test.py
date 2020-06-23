import os
from pathlib import Path

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from servo.config import (
    License,
    Maturity,
    Optimizer,
    Settings,
    Version,
)
from servo.connector import (
    Connector,          
    MeasureConnector,
    AdjustConnector
)
from servo.servo import (
    Servo,
    BaseServoSettings,
)
from connectors.vegeta.vegeta import (
    TargetFormat,
    VegetaConnector,
    VegetaSettings
)
# test load from config file

class TestOptimizer:
    def test_org_domain_valid(self) -> None:
        optimizer = Optimizer("example.com/my-app", token="123456")
        assert optimizer.org_domain == "example.com"

    def test_org_domain_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer("invalid/my-app", token="123456")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("org_domain",)
        assert (
            e.value.errors()[0]["msg"]
            == 'string does not match regex "(([\\da-zA-Z])([_\\w-]{,62})\\.){,127}(([\\da-zA-Z])[_\\w-]{,61})?([\\da-zA-Z]\\.((xn\\-\\-[a-zA-Z\\d]+)|([a-zA-Z\\d]{2,})))"'
        )

    def test_app_name_valid(self) -> None:
        optimizer = Optimizer("example.com/my-app", token="123456")
        assert optimizer.app_name == "my-app"

    def test_app_name_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer("example.com/$$$invalid$$$", token="123456")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("app_name",)
        assert (
            e.value.errors()[0]["msg"]
            == 'string does not match regex "^[a-z\\-]{3,64}$"'
        )

    def test_token_validation(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer("example.com/my-app", token=None)
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("token",)
        assert e.value.errors()[0]["msg"] == "none is not an allowed value"

    def test_base_url_validation(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer("example.com/my-app", token="123456", base_url="INVALID")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("base_url",)
        assert e.value.errors()[0]["msg"] == "invalid or missing URL scheme"


class ConnectorSettingsTests:
    pass


class TestLicense:
    def test_license_from_string(self):
        l = License.from_str("MIT")
        assert l == License.MIT

    def test_license_from_string_invalid_raises(self):
        with pytest.raises(NameError) as e:
            License.from_str("INVALID")
        assert 'No license identified by "INVALID"' in str(e)


class TestMaturity:
    def test_maturity_from_string(self):
        l = Maturity.from_str("Stable")
        assert l == Maturity.STABLE

    def test_license_from_string_invalid_raises(self):
        with pytest.raises(NameError) as e:
            Maturity.from_str("INVALID")
        assert 'No maturity level identified by "INVALID"' in str(e)


class TestConnector:
    def test_subclass_registration(self) -> None:
        class RegisterMe(Connector):
            pass

        assert RegisterMe in Connector.all()

    def test_default_name(self) -> None:
        class TestConnector(Connector):
            pass

        assert TestConnector.name == "Test Connector"

    def test_default_version(self) -> None:
        class TestConnector(Connector):
            pass

        assert TestConnector.version == "0.0.0"

    def test_default_path(self) -> None:
        class FancyConnector(Connector):
            pass

        c = FancyConnector(Settings())
        assert c.path == "fancy"

from tests.conftest import environment_overrides
class TestSettings:
    def test_configuring_with_environment_variables(self) -> None:
        with environment_overrides({ "SERVO_DESCRIPTION": "this description" }):
            assert os.environ['SERVO_DESCRIPTION'] == 'this description'
            s = Settings()
            assert s.description == "this description"

class TestServoSettings:
    def test_ignores_extra_attributes(self) -> None:
        # Ignored attribute would raise if misconfigured
        s = BaseServoSettings(
            ignored=[], optimizer=Optimizer("example.com/my-app", token="123456")
        )
        with pytest.raises(AttributeError) as e:
            s.ignored
        assert "'BaseServoSettings' object has no attribute 'ignored'" in str(e)

    def test_override_optimizer_settings_with_env_vars(self) -> None:
        with environment_overrides({ "SERVO_OPTIMIZER": '{"token": "abcdefg"}' }):
            assert os.environ['SERVO_OPTIMIZER'] is not None
            s = BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com' }
            )
            assert s.optimizer.token == "abcdefg"
    
    def test_set_connectors_with_env_vars(self) -> None:
        with environment_overrides({ 'SERVO_CONNECTORS': '["measure"]' }):
            assert os.environ['SERVO_CONNECTORS'] is not None
            s = BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' }
            )
            assert s.connectors == {'measure': 'servo.connector.MeasureConnector'}

    def test_connectors_allows_none(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors=None
        )
        assert s.connectors is None
    
    def test_connectors_allows_set_of_classes(self):
        class FooConnector(Connector):
            pass
        class BarConnector(Connector):
            pass
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={FooConnector, BarConnector}
        )
        assert s.connectors == {'foo': 'tests.connector_test.FooConnector', 'bar': 'tests.connector_test.BarConnector'}
    
    def test_connectors_rejects_invalid_connector_set_elements(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={BaseServoSettings}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == "BaseServoSettings is not a Connector subclass"
    
    def test_connectors_allows_set_of_class_names(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'MeasureConnector', 'AdjustConnector'}
        )
        assert s.connectors == {'measure': 'servo.connector.MeasureConnector', 'adjust': 'servo.connector.AdjustConnector'}
    
    def test_connectors_rejects_invalid_connector_set_class_name_elements(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={'BaseServoSettings'}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == "BaseServoSettings is not a Connector subclass"
    
    def test_connectors_allows_set_of_keys(self):
        from typing import Type

        from pydantic import BaseModel
        from pydantic import ValidationError
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'vegeta'}
        )
        assert s.connectors == {'vegeta': 'connectors.vegeta.vegeta.VegetaConnector'}
    
    def test_connectors_allows_dict_of_keys_to_classes(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'alias': VegetaConnector}
        )
        assert s.connectors == {'alias': 'connectors.vegeta.vegeta.VegetaConnector'}
    
    def test_connectors_allows_dict_of_keys_to_class_names(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'alias': 'VegetaConnector'}
        )
        assert s.connectors == {'alias': 'connectors.vegeta.vegeta.VegetaConnector'}

    def test_connectors_allows_dict_with_explicit_map_to_default_path(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'vegeta': 'VegetaConnector'}
        )
        assert s.connectors == {'vegeta': 'connectors.vegeta.vegeta.VegetaConnector'}
    
    def test_connectors_allows_dict_with_explicit_map_to_default_class(self):
        s = BaseServoSettings(
            optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
            connectors={'vegeta': VegetaConnector}
        )
        assert s.connectors == {'vegeta': 'connectors.vegeta.vegeta.VegetaConnector'}    

    def test_connectors_forbids_dict_with_existing_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={'vegeta': 'MeasureConnector'}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == 'Key "vegeta" is reserved by `VegetaConnector`'

    @pytest.fixture(autouse=True, scope='session')
    def discover_connectors(self) -> None:
        from servo.connector import ConnectorLoader
        loader = ConnectorLoader()
        for connector in loader.load():
            print("Loaded ", connector)

    
    def test_connectors_forbids_dict_with_reserved_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={'connectors': 'VegetaConnector'}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == 'Key "connectors" is reserved'
    
    def test_connectors_forbids_dict_with_invalid_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={'This Is Not Valid': 'VegetaConnector'}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == 'Key "This Is Not Valid" is not valid: paths may only contain alphanumeric characters, hyphens, slashes, periods, and underscores'
    
    def test_connectors_rejects_invalid_connector_dict_values(self):
        with pytest.raises(ValidationError) as e:
            BaseServoSettings(
                optimizer = { 'app_name': 'my-app', 'org_domain': 'example.com', 'token': '123456789' },
                connectors={'whatever': 'Not a Real Connector'}
            )
        assert "1 validation error for BaseServoSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == 'Not a Real Connector does not identify a Connector class'

class TestServo:
    def test_init_with_optimizer(self) -> None:
        pass

    def test_all_connectors(self) -> None:
        class FooConnector(Connector):
            pass

        c = ServoAssembly.construct().all_connectors()
        assert FooConnector in c

###
### Connector specific
###


class TestVegetaSettings:
    def test_rate_is_required(self) -> None:
        schema = VegetaSettings.schema()
        assert 'rate' in schema['required']

    def test_duration_is_required(self) -> None:
        schema = VegetaSettings.schema()
        assert 'duration' in schema['required']

    def test_validate_infinite_rate(self) -> None:
        s = VegetaSettings(rate="0", duration="0", target="GET http://example.com")
        assert s.rate == "0"

    def test_validate_rate_no_time_unit(self) -> None:
        s = VegetaSettings(rate="500", duration="0", target="GET http://example.com")
        assert s.rate == "500"

    def test_validate_rate_integer(self) -> None:
        s = VegetaSettings(rate=500, duration="0", target="GET http://example.com")
        assert s.rate == "500"

    def test_validate_rate_connections_over_time(self) -> None:
        s = VegetaSettings(rate="500/1s", duration="0", target="GET http://example.com")
        assert s.rate == "500/1s"

    def test_validate_rate_raises_when_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(
                rate="INVALID", duration="0", target="GET http://example.com"
            )
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("rate",)
        assert (
            e.value.errors()[0]["msg"] == "rate strings are of the form hits/interval"
        )

    def test_validate_rate_raises_when_invalid_duration(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(
                rate="500/1zxzczc", duration="0", target="GET http://example.com"
            )
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("rate",)
        assert e.value.errors()[0]["msg"] == "Unknown unit zxzczc in duration 1zxzczc"

    def test_validate_duration_infinite_attack(self) -> None:
        s = VegetaSettings(rate="0", duration="0", target="GET http://example.com")
        assert s.duration == "0"

    def test_validate_duration_seconds(self) -> None:
        s = VegetaSettings(rate="0", duration="1s", target="GET http://example.com")
        assert s.duration == "1s"

    def test_validate_duration_hours_minutes_and_seconds(self) -> None:
        s = VegetaSettings(
            rate="0", duration="1h35m20s", target="GET http://example.com"
        )
        assert s.duration == "1h35m20s"

    def test_validate_duration_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(
                rate="0", duration="INVALID", target="GET http://example.com"
            )
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("duration",)
        assert e.value.errors()[0]["msg"] == "Invalid duration INVALID"

    def test_validate_target_with_http_format(self) -> None:
        s = VegetaSettings(
            rate="0", duration="0", format="http", target="GET http://example.com"
        )
        assert s.format == TargetFormat.http

    def test_validate_target_with_json_format(self) -> None:
        s = VegetaSettings(
            rate="0",
            duration="0",
            format="json",
            target='{ "url": "http://example.com" }',
        )
        assert s.format == TargetFormat.json

    def test_validate_target_with_invalid_format(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(
                rate="0",
                duration="0",
                format="invalid",
                target="GET http://example.com",
            )
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("format",)
        assert (
            e.value.errors()[0]["msg"]
            == "value is not a valid enumeration member; permitted: 'http', 'json'"
        )

    def test_validate_taget_or_targets_must_be_selected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as e:
            s = VegetaSettings(rate="0", duration="0")
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert e.value.errors()[0]["msg"] == "target or targets must be configured"

    def test_validate_taget_or_targets_cant_both_be_selected(
        self, tmp_path: Path
    ) -> None:
        targets = tmp_path / "targets"
        targets.touch()
        with pytest.raises(ValidationError) as e:
            s = VegetaSettings(
                rate="0",
                duration="0",
                target="GET http://example.com",
                targets="targets",
            )
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert (
            e.value.errors()[0]["msg"] == "target and targets cannot both be configured"
        )

    def test_validate_targets_with_path(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets"
        targets.touch()
        s = VegetaSettings(rate="0", duration="0", targets=targets)
        assert s.targets == targets

    def test_validate_targets_with_path_doesnt_exist(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets"
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate="0", duration="0", targets=targets)
        assert "2 validation errors for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("targets",)
        assert "file or directory at path" in e.value.errors()[0]["msg"]

    def test_providing_invalid_target_with_json_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate="0", duration="0", format="json", target="INVALID")
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert "the target is not valid JSON" in e.value.errors()[0]["msg"]

    def test_providing_invalid_targets_with_json_format(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets.json"
        targets.write_text("<xml>INVALID</xml>")
        with pytest.raises(ValidationError) as e:
            VegetaSettings(rate="0", duration="0", format="json", targets=targets)
        assert "1 validation error for VegetaSettings" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert "the targets file is not valid JSON" in e.value.errors()[0]["msg"]

    # TODO: Test the combination of JSON and HTTP targets


class VegetaConnectorTests:
    pass


def test_init_vegeta_connector() -> None:
    settings = VegetaSettings(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(settings)
    assert connector is not None


def test_init_vegeta_connector_no_settings() -> None:
    with pytest.raises(ValidationError) as e:
        VegetaConnector(None)
    assert "1 validation error for VegetaConnector" in str(e.value)


def test_init_connector_no_version_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.version = None
        settings = VegetaSettings(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(settings, path="whatever")
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "version must be provided"


def test_init_connector_invalid_version_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.version = "invalid"
        settings = VegetaSettings(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(settings, path="whatever", version="b")
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "invalid is not valid SemVer string"


def test_init_connector_parses_version_string() -> None:
    class FakeConnector(Connector):
        pass

    FakeConnector.version = "0.5.0"
    settings = VegetaSettings(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = FakeConnector(settings, path="whatever")
    assert connector.version is not None
    assert connector.version == Version.parse("0.5.0")


def test_init_connector_no_name_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.name = None
        settings = VegetaSettings(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(settings, path="test", name=None)
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "name must be provided"


def test_vegeta_default_path() -> None:
    settings = VegetaSettings(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(settings)
    assert connector.path == "vegeta"


def test_vegeta_id_override() -> None:
    settings = VegetaSettings(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(settings, path="monkey")
    assert connector.path == "monkey"


def test_vegeta_id_invalid() -> None:
    with pytest.raises(ValidationError) as e:
        settings = VegetaSettings(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = VegetaConnector(settings, path="THIS IS NOT COOL")
    assert "1 validation error for VegetaConnector" in str(e.value)
    assert (
        e.value.errors()[0]["msg"]
        == ('paths may only contain alphanumeric characters, hyphens, slashes, periods, \n and underscores')
    )


def test_vegeta_name() -> None:
    assert VegetaConnector.name == "Vegeta Connector"


def test_vegeta_description() -> None:
    assert VegetaConnector.description == "Vegeta load testing connector"


def test_vegeta_version() -> None:
    # TODO: Type violation
    assert VegetaConnector.version == "0.5.0"


def test_vegeta_homepage() -> None:
    # TODO: Type violation
    assert VegetaConnector.homepage == "https://github.com/opsani/vegeta-connector"


def test_vegeta_license() -> None:
    assert VegetaConnector.license == License.APACHE2


def test_vegeta_maturity() -> None:
    assert VegetaConnector.maturity == Maturity.STABLE


@pytest.fixture()
def vegeta_cli() -> typer.Typer:
    settings = VegetaSettings(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(settings)
    return connector.cli()


## Vegeta CLI tests
def test_vegeta_cli_help(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "--help")
    assert result.exit_code == 0
    assert "Usage: vegeta [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_vegeta_cli_schema_json(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "schema")
    assert result.exit_code == 0
    assert result.stdout == (
        '{\n'
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
        '      "$ref": "#/definitions/TargetFormat"\n'
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
        '      "env_names": [\n'
        '        "servo_targets"\n'
        '      ],\n'
        '      "format": "file-path",\n'
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
        '      "env": "",\n'
        '      "env_names": [\n'
        '        ""\n'
        '      ],\n'
        '      "type": "integer"\n'
        '    },\n'
        '    "max-body": {\n'
        '      "title": "Max-Body",\n'
        '      "description": "Specifies the maximum number of bytes to capture from the body of each response. Remain'
        'ing unread bytes will be fully read but discarded.",\n'
        '      "default": -1,\n'
        '      "env": "",\n'
        '      "env_names": [\n'
        '        ""\n'
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
        '    "duration"\n'
        '  ],\n'
        '  "additionalProperties": false,\n'
        '  "definitions": {\n'
        '    "TargetFormat": {\n'
        '      "title": "TargetFormat",\n'
        '      "description": "An enumeration.",\n'
        '      "enum": [\n'
        '        "http",\n'
        '        "json"\n'
        '      ],\n'
        '      "type": "string"\n'
        '    }\n'
        '  }\n'
        '}\n'
    )

@pytest.mark.xfail
def test_vegeta_cli_schema_text(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "schema -f text")
    assert result.exit_code == 2
    assert 'not yet implemented' in result.stderr

@pytest.mark.xfail
def test_vegeta_cli_schema_html(vegeta_cli: typer.Typer, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(vegeta_cli, "schema -f html")
    assert result.exit_code == 2
    assert 'not yet implemented' in result.stderr

# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)


def test_vegeta_cli_generate(
    tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(vegeta_cli, "generate")
    assert result.exit_code == 0
    assert "Generated vegeta.yaml" in result.stdout
    config_file = tmp_path / "vegeta.yaml"
    config = config_file.read_text()
    assert config == (
        "connections: 10000\n"
        "description: null\n"
        "duration: 5m\n"
        "format: http\n"
        "http2: true\n"
        "insecure: false\n"
        "keepalive: true\n"
        "max-body: -1\n"
        "max-workers: 18446744073709551615\n"
        "rate: 50/1s\n"
        "target: GET http://localhost:8080\n"
        "targets: null\n"
        "workers: 10\n"
    )


def test_vegeta_cli_validate(
    tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config_file.write_text(
        (
            "connections: 10000\n"
            "description: null\n"
            "duration: 5m\n"
            "format: http\n"
            "http2: true\n"
            "insecure: false\n"
            "keepalive: true\n"
            "max-body: -1\n"
            "max-workers: 18446744073709551615\n"
            "rate: 50/1s\n"
            "target: GET http://localhost:8080\n"
            "targets: null\n"
            "workers: 10\n"
        )
    )
    result = cli_runner.invoke(vegeta_cli, "validate vegeta.yaml")
    assert result.exit_code == 0
    assert "√ Valid connector configuration" in result.stdout


def test_vegeta_cli_validate_no_such_file(
    tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(vegeta_cli, "validate doesntexist.yaml")
    assert result.exit_code == 2
    assert "Could not open file: doesntexist.yaml" in result.stderr


def test_vegeta_cli_validate_invalid_config(
    tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        (
            "connections: 10000\n"
            "description: null\n"
            # 'duration: 5m\n'  # Duration is omitted
            "format: http\n"
            "http2: true\n"
            "insecure: false\n"
            "keepalive: true\n"
            "max-body: -1\n"
            "max-workers: 18446744073709551615\n"
            #'rate: 50/1s\n'  # Rate is omitted
            "target: GET http://localhost:8080\n"
            "targets: null\n"
            "workers: 10\n"
        )
    )
    result = cli_runner.invoke(vegeta_cli, "validate invalid.yaml")
    assert result.exit_code == 1
    assert "2 validation errors for VegetaSettings" in result.stderr


def test_vegeta_cli_validate_invalid_syntax(
    tmp_path: Path, vegeta_cli: typer.Typer, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        ("connections: 10000\n" "descriptions\n\n null\n" "duratio\n\n_   n: 5m\n")
    )
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

# def test_loading_optimizer_from_environment() -> None:
#     with environment_overrides({ 
#         'SERVO_OPTIMIZER': 'dev.opsani.com/servox',  
#         'SERVO_TOKEN': '123456789',
#     }):
#         # assert os.environ['SERVO_OPTIMIZER'] is not None
#         # o = Optimizer()
#         # assert s.connectors == {'measure': 'servo.connector.MeasureConnector'}
#         # result = cli_runner.invoke(vegeta_cli, "version")
#         # assert result.exit_code == 0
#         # assert "Vegeta Connector v0.5.0" in result.stdout
