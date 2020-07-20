import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import Extra, ValidationError
from typer.testing import CliRunner

from servo import Duration
from servo.cli import ServoCLI
from servo.connector import (
    BaseConfiguration,
    Connector,
    License,
    Maturity,
    Optimizer,
    Version,
    _connector_subclasses,
    EventContext
)
from servo.assembly import BaseServoConfiguration
from servo.connectors.vegeta import TargetFormat, VegetaConfiguration, VegetaConnector
from servo.events import Preposition, _events, event
from tests.test_helpers import *

pytestmark = pytest.mark.asyncio

class TestOptimizer:
    def test_org_domain_valid(self) -> None:
        optimizer = Optimizer(id="example.com/my-app", token="123456")
        assert optimizer.org_domain == "example.com"

    def test_org_domain_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer(id="invalid/my-app", token="123456")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("org_domain",)
        assert (
            e.value.errors()[0]["msg"]
            == 'string does not match regex "(([\\da-zA-Z])([_\\w-]{,62})\\.){,127}(([\\da-zA-Z])[_\\w-]{,61})?([\\da-zA-Z]\\.((xn\\-\\-[a-zA-Z\\d]+)|([a-zA-Z\\d]{2,})))"'
        )

    def test_app_name_valid(self) -> None:
        optimizer = Optimizer(id="example.com/my-app", token="123456")
        assert optimizer.app_name == "my-app"

    def test_app_name_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer(id="example.com/$$$invalid$$$", token="123456")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("app_name",)
        assert (
            e.value.errors()[0]["msg"]
            == 'string does not match regex "^[a-z\\-\\.0-9]{3,64}$"'
        )

    def test_token_validation(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer(id="example.com/my-app", token=None)
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("token",)
        assert e.value.errors()[0]["msg"] == "none is not an allowed value"

    def test_base_url_validation(self) -> None:
        with pytest.raises(ValidationError) as e:
            Optimizer(id="example.com/my-app", token="123456", base_url="INVALID")
        assert "1 validation error for Optimizer" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("base_url",)
        assert e.value.errors()[0]["msg"] == "invalid or missing URL scheme"


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

        assert RegisterMe in _connector_subclasses

    def test_default_name(self) -> None:
        class TestConnector(Connector):
            pass

        assert TestConnector.name == "Test"
        assert TestConnector.full_name == "Test Connector"
        assert TestConnector.__default_name__ == "test"

    def test_default_version(self) -> None:
        class TestConnector(Connector):
            pass

        assert TestConnector.version == "0.0.0"

    def test_default_name(self) -> None:
        class FancyConnector(Connector):
            pass

        c = FancyConnector(config=BaseConfiguration())
        assert c.__default_name__ == "fancy"


class TestBaseConfiguration:
    class SomeConfiguration(BaseConfiguration):
        duration: Duration

    def test_duration_assumes_seconds(self) -> None:
        config = TestBaseConfiguration.SomeConfiguration(duration="60s")
        assert config.yaml(exclude_unset=True) == "duration: 1m\n"

    def test_serializing_duration(self) -> None:
        config = TestBaseConfiguration.SomeConfiguration(duration="60s")
        assert config.yaml(exclude_unset=True) == "duration: 1m\n"

    def test_duration_schema(self) -> None:
        schema = TestBaseConfiguration.SomeConfiguration.schema()
        duration_prop = schema["properties"]["duration"]
        assert duration_prop == {
            "title": "Duration",
            "env_names": {"SOME_DURATION",},
            "type": "string",
            "format": "duration",
            "pattern": "([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
            "examples": ["300ms", "5m", "2h45m", "72h3m0.5s",],
        }

    def test_configuring_with_environment_variables(self) -> None:
        assert BaseConfiguration.__fields__["description"].field_info.extra[
            "env_names"
        ] == {"DESCRIPTION"}
        with environment_overrides({"DESCRIPTION": "this description"}):
            assert os.environ["DESCRIPTION"] == "this description"
            s = BaseConfiguration()
            assert s.description == "this description"


###
### Connector specific
###


class TestVegetaConfiguration:
    def test_rate_is_required(self) -> None:
        schema = VegetaConfiguration.schema()
        assert "rate" in schema["required"]

    def test_duration_is_required(self) -> None:
        schema = VegetaConfiguration.schema()
        assert "duration" in schema["required"]

    def test_duration_str(self) -> None:
        config = VegetaConfiguration.generate()
        assert yaml_key_path(config.yaml(), "duration") == "5m"

    def test_validate_infinite_rate(self) -> None:
        s = VegetaConfiguration(rate="0", duration="0", target="GET http://example.com")
        assert s.rate == "0"

    def test_validate_rate_no_time_unit(self) -> None:
        s = VegetaConfiguration(
            rate="500", duration="0", target="GET http://example.com"
        )
        assert s.rate == "500"

    def test_validate_rate_integer(self) -> None:
        s = VegetaConfiguration(rate=500, duration="0", target="GET http://example.com")
        assert s.rate == "500"

    def test_validate_rate_connections_over_time(self) -> None:
        s = VegetaConfiguration(
            rate="500/1s", duration="0", target="GET http://example.com"
        )
        assert s.rate == "500/1s"

    def test_validate_rate_raises_when_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="INVALID", duration="0", target="GET http://example.com"
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("rate",)
        assert (
            e.value.errors()[0]["msg"] == "rate strings are of the form hits/interval"
        )

    def test_validate_rate_raises_when_invalid_duration(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="500/1zxzczc", duration="0", target="GET http://example.com"
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("rate",)
        assert (
            e.value.errors()[0]["msg"]
            == "Invalid duration '1zxzczc' in rate '500/1zxzczc'"
        )

    def test_validate_duration_infinite_attack(self) -> None:
        s = VegetaConfiguration(rate="0", duration="0", target="GET http://example.com")
        assert s.duration == timedelta(seconds=0)

    def test_validate_duration_seconds(self) -> None:
        s = VegetaConfiguration(
            rate="0", duration="1s", target="GET http://example.com"
        )
        assert s.duration == timedelta(seconds=1)

    def test_validate_duration_hours_minutes_and_seconds(self) -> None:
        s = VegetaConfiguration(
            rate="0", duration="1h35m20s", target="GET http://example.com"
        )
        assert s.duration == timedelta(seconds=5720)

    def test_validate_duration_invalid(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0", duration="INVALID", target="GET http://example.com"
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("duration",)
        assert e.value.errors()[0]["msg"] == "Invalid duration 'INVALID'"

    def test_validate_target_with_http_format(self) -> None:
        s = VegetaConfiguration(
            rate="0", duration="0", format="http", target="GET http://example.com"
        )
        assert s.format == TargetFormat.http

    def test_validate_target_with_json_format(self) -> None:
        s = VegetaConfiguration(
            rate="0",
            duration="0",
            format="json",
            target='{ "url": "http://example.com", "method": "GET" }',
        )
        assert s.format == TargetFormat.json

    def test_validate_target_http_doesnt_match_schema(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0",
                duration="0",
                format="json",
                target='{ "url": "http://example.com", "method": "INVALID" }',
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("target",)
        assert (
            e.value.errors()[0]["msg"]
            == "Invalid Vegeta JSON target: 'INVALID' is not one of ['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'TRACE', 'HEAD', 'DELETE', 'CONNECT']"
        )

    def test_validate_target_json_doesnt_match_schema(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0",
                duration="0",
                format="json",
                target='{ "url": "http://example.com", "method": "INVALID" }',
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("target",)
        assert (
            e.value.errors()[0]["msg"]
            == "Invalid Vegeta JSON target: 'INVALID' is not one of ['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'TRACE', 'HEAD', 'DELETE', 'CONNECT']"
        )

    @pytest.mark.parametrize(
        "http_target",
        [
            "GET http://goku:9090/path/to/dragon?item=ball",
            (
                "GET http://goku:9090/path/to/dragon?item=ball\n"
                "GET http://user:password@goku:9090/path/to\n"
                "HEAD http://goku:9090/path/to/success\n"
            ),
            ("GET http://user:password@goku:9090/path/to\n" "X-Account-ID: 8675309\n"),
            (
                "DELETE http://goku:9090/path/to/remove\n"
                "Confirmation-Token: 90215\n"
                "Authorization: Token DEADBEEF\n"
            ),
            (
                "POST http://goku:9090/things\n"
                "@/path/to/newthing.json\n"
                "\n\n"
                "PATCH http://goku:9090/thing/71988591\n"
                "@/path/to/thing-71988591.json"
            ),
            (
                "POST http://goku:9090/things\n"
                "X-Account-ID: 99\n"
                "@/path/to/newthing.json\n"
                "     \n"
            ),
            (
                "# get a dragon ball\n"
                "GET http://goku:9090/path/to/dragon?item=ball\n"
                "# specify a test accout\n"
                "X-Account-ID: 99\n"
            ),
        ],
    )
    def test_validate_target_http_valid_cases(self, http_target):
        s = VegetaConfiguration(
            rate="0", duration="0", format="http", target=http_target,
        )

    def test_validate_target_http_empty(self):
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0", duration="0", format="http", target="",
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("target",)
        assert e.value.errors()[0]["msg"] == "no targets found"

    @pytest.mark.parametrize(
        "http_target, error_message",
        [
            [
                "ZGET http://goku:9090/path/to/dragon?item=ball",
                "invalid target: ZGET http://goku:9090/path/to/dragon?item=ball",
            ],
            [
                (
                    "GET gopher://goku:9090/path/to/dragon?item=ball\n"
                    "GET http://user:password@goku:9090/path/to\n"
                    "HEAD http://goku:9090/path/to/success\n"
                ),
                "invalid target: GET gopher://goku:9090/path/to/dragon?item=ball",
            ],
            [
                ("GET http://user:password@goku:9090/path/to\n" "X-Account-ID:"),
                "invalid target: X-Account-ID:",
            ],
            [
                (
                    "!!! DELETE http://goku:9090/path/to/remove\n"
                    "Confirmation-Token: 90215\n"
                    "Authorization: Token DEADBEEF\n"
                ),
                "invalid target: !!! DELETE http://goku:9090/path/to/remove",
            ],
            [
                (
                    "POST http://goku:9090/things\n"
                    "0@/path/to/newthing.json\n"
                    "\n\n"
                    "PATCH http://goku:9090/thing/71988591\n"
                    "@/path/to/thing-71988591.json"
                ),
                "invalid target: 0@/path/to/newthing.json",
            ],
            [
                (
                    "JUMP http://goku:9090/things\n"
                    "X-Account-ID: 99\n"
                    "@/path/to/newthing.json\n"
                    "     \n"
                ),
                "invalid target: JUMP http://goku:9090/things",
            ],
        ],
    )
    def test_validate_target_http_invalid_cases(self, http_target, error_message):
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0", duration="0", format="http", target=http_target,
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("target",)
        assert e.value.errors()[0]["msg"] == error_message

    def test_validate_target_with_invalid_format(self) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(
                rate="0",
                duration="0",
                format="invalid",
                target="GET http://example.com",
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("format",)
        assert (
            e.value.errors()[0]["msg"]
            == "value is not a valid enumeration member; permitted: 'http', 'json'"
        )

    def test_validate_target_or_targets_must_be_selected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as e:
            s = VegetaConfiguration(rate="0", duration="0")
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert e.value.errors()[0]["msg"] == "target or targets must be configured"

    def test_validate_target_or_targets_cant_both_be_selected(
        self, tmp_path: Path
    ) -> None:
        targets = tmp_path / "targets"
        targets.touch()
        with pytest.raises(ValidationError) as e:
            s = VegetaConfiguration(
                rate="0",
                duration="0",
                target="GET http://example.com",
                targets="targets",
            )
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("__root__",)
        assert (
            e.value.errors()[0]["msg"] == "target and targets cannot both be configured"
        )

    def test_validate_targets_with_empty_file(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets"
        targets.touch()
        with pytest.raises(ValidationError) as e:
            s = VegetaConfiguration(rate="0", duration="0", targets=targets)
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("targets",)
        assert "no targets found" in e.value.errors()[0]["msg"]

    def test_validate_targets_with_path_doesnt_exist(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets"
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(rate="0", duration="0", targets=targets)
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("targets",)
        assert "file or directory at path" in e.value.errors()[0]["msg"]

    def test_providing_invalid_target_with_json_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(rate="0", duration="0", format="json", target="INVALID")
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("target",)
        assert "target contains invalid JSON" in e.value.errors()[0]["msg"]

    def test_providing_invalid_targets_with_json_format(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets.json"
        targets.write_text("<xml>INVALID</xml>")
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(rate="0", duration="0", format="json", targets=targets)
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("targets",)
        assert "targets contains invalid JSON" in e.value.errors()[0]["msg"]

    def test_validate_targets_json_doesnt_match_schema(self, tmp_path: Path) -> None:
        targets = tmp_path / "targets.json"
        targets.write_text('{ "url": "http://example.com", "method": "INVALID" }')
        with pytest.raises(ValidationError) as e:
            VegetaConfiguration(rate="0", duration="0", format="json", targets=targets)
        assert "1 validation error for VegetaConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("targets",)
        assert (
            "Invalid Vegeta JSON target: 'INVALID' is not one of ['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'TRACE', 'HEAD', 'DELETE', 'CONNECT']"
            in e.value.errors()[0]["msg"]
        )


# TODO: All of these tests need to be expanded
class TestVegetaConnector:
    @pytest.fixture
    def vegeta_connector(self) -> VegetaConnector:
        config = VegetaConfiguration(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        return VegetaConnector(config=config)

    @pytest.fixture(autouse=True)
    def mock_run_vegeta(self, mocker) -> None:
        mocker.patch.object(
            VegetaConnector, "_run_vegeta", return_value=(1, "vegeta attack")
        )

    async def test_vegeta_check(self, vegeta_connector: VegetaConnector, mocker) -> None:
        mocker.patch.object(
            VegetaConnector, "_run_vegeta", return_value=(0, "vegeta attack")
        )
        await vegeta_connector.check()

    def test_vegeta_metrics(self, vegeta_connector: VegetaConnector, mocker) -> None:
        mocker.patch.object(
            VegetaConnector, "_run_vegeta", return_value=(0, "vegeta attack")
        )
        vegeta_connector.metrics()

    async def test_vegeta_check_failed(self, vegeta_connector: VegetaConnector) -> None:
        await vegeta_connector.check()

    def test_vegeta_metrics_failed(self, vegeta_connector: VegetaConnector) -> None:
        vegeta_connector.metrics()

    def test_vegeta_describe(self, vegeta_connector: VegetaConnector) -> None:
        vegeta_connector.describe()

    async def test_vegeta_measure(self, vegeta_connector: VegetaConnector) -> None:
        await vegeta_connector.measure()


def test_init_vegeta_connector() -> None:
    config = VegetaConfiguration(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(config=config)
    assert connector is not None


def test_init_vegeta_connector_no_settings() -> None:
    with pytest.raises(ValidationError) as e:
        VegetaConnector(config=None)
    assert "1 validation error for VegetaConnector" in str(e.value)


def test_init_connector_no_version_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.version = None
        config = VegetaConfiguration(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(config=config, path="whatever")
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "version must be provided"


def test_init_connector_invalid_version_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.version = "invalid"
        config = VegetaConfiguration(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(config=config, path="whatever", version="b")
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "invalid is not valid SemVer string"


def test_init_connector_parses_version_string() -> None:
    class FakeConnector(Connector):
        pass

    FakeConnector.version = "0.5.0"
    config = VegetaConfiguration(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = FakeConnector(config=config, path="whatever")
    assert connector.version is not None
    assert connector.version == Version.parse("0.5.0")


def test_init_connector_no_name_raises() -> None:
    class FakeConnector(Connector):
        pass

    with pytest.raises(ValidationError) as e:
        FakeConnector.name = None
        config = VegetaConfiguration(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = FakeConnector(config=config, path="test", name=None)
    assert e.value.errors()[0]["loc"] == ("__root__",)
    assert e.value.errors()[0]["msg"] == "name must be provided"


def test_vegeta_default_name() -> None:
    config = VegetaConfiguration(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(config=config)
    assert connector.name == "vegeta"


def test_vegeta_config_override() -> None:
    config = VegetaConfiguration(
        rate="50/1s", duration="5m", target="GET http://localhost:8080"
    )
    connector = VegetaConnector(config=config, name="monkey")
    assert connector.name == "monkey"


def test_vegeta_name_invalid() -> None:
    with pytest.raises(ValidationError) as e:
        config = VegetaConfiguration(
            rate="50/1s", duration="5m", target="GET http://localhost:8080"
        )
        connector = VegetaConnector(configuration=config, name="THIS IS NOT COOL")
    error_messages = list(map(lambda error: error["msg"], e.value.errors()))
    assert (
        "names may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
        in error_messages
    )


def test_vegeta_name() -> None:
    assert VegetaConnector.name == "Vegeta"


def test_vegeta_description() -> None:
    assert VegetaConnector.description == "Vegeta load testing connector"


def test_vegeta_version() -> None:
    assert VegetaConnector.version == "0.5.0"


def test_vegeta_homepage() -> None:
    assert VegetaConnector.homepage == "https://github.com/opsani/vegeta-connector"


def test_vegeta_license() -> None:
    assert VegetaConnector.license == License.APACHE2


def test_vegeta_maturity() -> None:
    assert VegetaConnector.maturity == Maturity.STABLE


## Vegeta CLI tests
def test_vegeta_cli_help(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(servo_cli, "--help")
    assert result.exit_code == 0
    assert "Usage: servo [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_env_variable_prefixing() -> None:
    schema_title_and_description_envs = [
        ["Connector Configuration Schema", "DESCRIPTION",],
        ["Vegeta Connector Configuration Schema", "VEGETA_DESCRIPTION",],
        ["Abstract Servo Configuration Schema", "SERVO_DESCRIPTION",],
    ]
    schemas = [
        BaseConfiguration.schema(),
        VegetaConfiguration.schema(),
        BaseServoConfiguration.schema(),
    ]
    # NOTE: popping the env_names without copying is a mistake you will only make once
    values = list(
        map(
            lambda schema: [
                schema["title"],
                schema["properties"]["description"]["env_names"].copy().pop(),
            ],
            schemas,
        )
    )
    assert values == schema_title_and_description_envs


def test_vegeta_cli_schema_json(
    servo_cli: ServoCLI, cli_runner: CliRunner, optimizer_env: None
) -> None:
    result = cli_runner.invoke(servo_cli, "schema vegeta")
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema == {
        "title": "Vegeta Connector Configuration Schema",
        "description": "Configuration of the Vegeta connector",
        "type": "object",
        "properties": {
            "description": {
                "title": "Description",
                "description": "An optional annotation describing the configuration.",
                "env_names": ["VEGETA_DESCRIPTION",],
                "type": "string",
            },
            "rate": {
                "title": "Rate",
                "description": (
                    "Specifies the request rate per time unit to issue against the targets. Given in the format of req"
                    "uest/time unit."
                ),
                "env_names": ["VEGETA_RATE",],
                "type": "string",
            },
            "duration": {
                "title": "Duration",
                "description": "Specifies the amount of time to issue requests to the targets.",
                "env_names": ["VEGETA_DURATION",],
                "type": "string",
                "format": "duration",
                "pattern": "([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
                "examples": ["300ms", "5m", "2h45m", "72h3m0.5s",],
            },
            "format": {"$ref": "#/definitions/TargetFormat",},
            "target": {
                "title": "Target",
                "description": (
                    "Specifies a single formatted Vegeta target to load. See the format option to learn about availabl"
                    "e target formats. This option is exclusive of the targets option and will provide a target to Veg"
                    "eta via stdin."
                ),
                "env_names": ["VEGETA_TARGET",],
                "type": "string",
            },
            "targets": {
                "title": "Targets",
                "description": (
                    "Specifies the file from which to read targets. See the format option to learn about available tar"
                    "get formats. This option is exclusive of the target option and will provide targets to via throug"
                    "h a file on disk."
                ),
                "env_names": ["VEGETA_TARGETS",],
                "format": "file-path",
                "type": "string",
            },
            "connections": {
                "title": "Connections",
                "description": "Specifies the maximum number of idle open connections per target host.",
                "default": 10000,
                "env_names": ["VEGETA_CONNECTIONS",],
                "type": "integer",
            },
            "workers": {
                "title": "Workers",
                "description": (
                    "Specifies the initial number of workers used in the attack. The workers will automatically increa"
                    "se to achieve the target request rate, up to max-workers."
                ),
                "default": 10,
                "env_names": ["VEGETA_WORKERS",],
                "type": "integer",
            },
            "max_workers": {
                "title": "Max Workers",
                "description": (
                    "The maximum number of workers used to sustain the attack. This can be used to control the concurr"
                    "ency of the attack to simulate a target number of clients."
                ),
                "default": 18446744073709551615,
                "env_names": ["VEGETA_MAX_WORKERS",],
                "type": "integer",
            },
            "max_body": {
                "title": "Max Body",
                "description": (
                    "Specifies the maximum number of bytes to capture from the body of each response. Remaining unread"
                    " bytes will be fully read but discarded."
                ),
                "default": -1,
                "env_names": ["VEGETA_MAX_BODY",],
                "type": "integer",
            },
            "http2": {
                "title": "Http2",
                "description": "Specifies whether to enable HTTP/2 requests to servers which support it.",
                "default": True,
                "env_names": ["VEGETA_HTTP2",],
                "type": "boolean",
            },
            "keepalive": {
                "title": "Keepalive",
                "description": "Specifies whether to reuse TCP connections between HTTP requests.",
                "default": True,
                "env_names": ["VEGETA_KEEPALIVE",],
                "type": "boolean",
            },
            "insecure": {
                "title": "Insecure",
                "description": "Specifies whether to ignore invalid server TLS certificates.",
                "default": False,
                "env_names": ["VEGETA_INSECURE",],
                "type": "boolean",
            },
            "reporting_interval": {
                "title": "Reporting Interval",
                "description": "How often to report metrics during a measurement cycle.",
                "default": "15s",
                "env_names": ["VEGETA_REPORTING_INTERVAL",],
                "type": "string",
                "format": "duration",
                "pattern": "([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
                "examples": ["300ms", "5m", "2h45m", "72h3m0.5s",],
            },
        },
        "required": ["rate", "duration",],
        "additionalProperties": False,
        "definitions": {
            "TargetFormat": {
                "title": "TargetFormat",
                "description": "An enumeration.",
                "enum": ["http", "json",],
                "type": "string",
            },
        },
    }


@pytest.mark.xfail
def test_vegeta_cli_schema_text(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(servo_cli, "schema -f text")
    assert result.exit_code == 2
    assert "not yet implemented" in result.stderr


@pytest.mark.xfail
def test_vegeta_cli_schema_html(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(servo_cli, "schema -f html")
    assert result.exit_code == 2
    assert "not yet implemented" in result.stderr


@pytest.fixture
def mock_run_vegeta(mocker) -> None:
    mocker.patch.object(
        VegetaConnector, "_run_vegeta", return_value=(0, "vegeta attack")
    )


@pytest.mark.xfail
def test_vegeta_cli_check(
    tmp_path: Path,
    servo_cli: ServoCLI,
    cli_runner: CliRunner,
    mock_run_vegeta,
    optimizer_env: None,
) -> None:
    result = cli_runner.invoke(servo_cli, "check vegeta")
    debug(result.stderr)
    assert result.exit_code == 0


@pytest.mark.xfail
def test_vegeta_cli_measure(
    tmp_path: Path,
    servo_cli: ServoCLI,
    cli_runner: CliRunner,
    mock_run_vegeta,
    optimizer_env: None,
) -> None:
    result = cli_runner.invoke(servo_cli, "measure vegeta")
    debug(result.stderr)
    assert result.exit_code == 0


# TODO: Need to run a fake vegeta or mock subprocess


def test_vegeta_cli_generate(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "generate vegeta")
    assert result.exit_code == 0
    assert "Generated servo.yaml" in result.stdout
    config_file = tmp_path / "servo.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "connectors": ["vegeta"],
        "vegeta": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
    }


def test_vegeta_cli_generate_filename(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "generate vegeta -f vegeta.yaml")
    assert result.exit_code == 0
    assert "Generated vegeta.yaml" in result.stdout
    config_file = tmp_path / "vegeta.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "connectors": ["vegeta"],
        "vegeta": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
    }


def test_vegeta_cli_generate_quiet(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "generate vegeta -f vegeta.yaml --quiet")
    assert result.exit_code == 0
    assert result.stdout == ""
    config_file = tmp_path / "vegeta.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "connectors": ["vegeta"],
        "vegeta": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
    }


def test_vegeta_cli_generate_standalone(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "generate vegeta -f vegeta.yaml --standalone")
    assert result.exit_code == 0
    config_file = tmp_path / "vegeta.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "vegeta": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
    }


def test_vegeta_cli_generate_aliases(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(
        servo_cli, "generate one:vegeta two:vegeta -f vegeta.yaml"
    )
    assert result.exit_code == 0
    config_file = tmp_path / "vegeta.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "connectors": {"one": "vegeta", "two": "vegeta",},
        "one": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
        "two": {
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "rate": "50/1s",
            "target": "GET https://example.com/",
        },
    }


def test_vegeta_cli_generate_with_defaults(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(
        servo_cli, "generate vegeta --defaults -f vegeta.yaml -s"
    )
    assert result.exit_code == 0
    assert "Generated vegeta.yaml" in result.stdout
    config_file = tmp_path / "vegeta.yaml"
    config = yaml.full_load(config_file.read_text())
    assert config == {
        "description": None,
        "vegeta": {
            "connections": 10000,
            "description": "Update the rate, duration, and target/targets to match your load profile",
            "duration": "5m",
            "format": "http",
            "http2": True,
            "insecure": False,
            "keepalive": True,
            "max_body": -1,
            "max_workers": 18446744073709551615,
            "rate": "50/1s",
            "reporting_interval": "15s",
            "target": "GET https://example.com/",
            "targets": None,
            "workers": 10,
        },
    }


def test_vegeta_cli_validate(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"vegeta": config.dict(exclude_unset=True)}, config_file)

    result = cli_runner.invoke(
        servo_cli, "validate -f vegeta.yaml", catch_exceptions=False
    )
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "√ Valid configuration in vegeta.yaml" in result.stdout


def test_vegeta_cli_validate_quiet(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"vegeta": config.dict(exclude_unset=True)}, config_file)
    result = cli_runner.invoke(servo_cli, "validate -q -f vegeta.yaml")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "" == result.stdout


def test_vegeta_cli_validate_dict(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    config_dict = {
        "connectors": {"first": "vegeta", "second": "vegeta"},
        "first": config.dict(exclude_unset=True),
        "second": config.dict(exclude_unset=True),
    }

    write_config_yaml(config_dict, config_file)
    result = cli_runner.invoke(
        servo_cli, "validate -f vegeta.yaml", catch_exceptions=False
    )
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "√ Valid configuration in vegeta.yaml" in result.stdout


def test_vegeta_cli_validate_invalid(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config_yaml = yaml.dump({"vegeta": {}})
    config_file.write_text(config_yaml)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "X Invalid configuration in vegeta.yaml" in result.stderr


def test_vegeta_cli_validate_invalid_key(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"nonsense": config.dict(exclude_unset=True)}, config_file)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "X Invalid configuration in vegeta.yaml" in result.stderr


def test_vegeta_cli_validate_file_doesnt_exist(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "validate -f wrong.yaml")
    assert (
        result.exit_code == 2
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "File 'wrong.yaml' does not exist" in result.stderr


def test_vegeta_cli_validate_wrong_connector(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"vegeta": config}, config_file)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml measure")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "X Invalid configuration in vegeta.yaml" in result.stderr


def test_vegeta_cli_validate_alias_syntax(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"vegeta": config}, config_file)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml vegeta:vegeta")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "√ Valid configuration in vegeta.yaml" in result.stdout


def test_vegeta_cli_validate_aliasing(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    write_config_yaml({"vegeta_alias": config}, config_file)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml vegeta_alias:vegeta")
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "√ Valid configuration in vegeta.yaml" in result.stdout


def test_vegeta_cli_validate_invalid_dict(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config = VegetaConfiguration.generate()
    config_dict = yaml.safe_load(config.yaml())
    config_yaml = yaml.dump({"nonsense": config_dict})
    config_file.write_text(config_yaml)
    result = cli_runner.invoke(servo_cli, "validate -f vegeta.yaml")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 0 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "X Invalid configuration in vegeta.yaml" in result.stderr


def test_vegeta_cli_validate_quiet_invalid(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "vegeta.yaml"
    config_yaml = yaml.dump({"vegeta": {}})
    config_file.write_text(config_yaml)
    result = cli_runner.invoke(servo_cli, "validate -q -f vegeta.yaml")
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1 but got {result.exit_code} -- stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "" == result.stdout


def test_vegeta_cli_validate_no_such_file(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    result = cli_runner.invoke(servo_cli, "validate -f doesntexist.yaml")
    assert result.exit_code == 2
    assert (
        "Error: Invalid value for '--file' / '-f': File 'doesntexist.yaml' does not exist."
        in result.stderr
    )


def test_vegeta_cli_validate_invalid_config(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
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
            "max_body: -1\n"
            "max_workers: 18446744073709551615\n"
            #'rate: 50/1s\n'  # Rate is omitted
            "target: GET http://localhost:8080\n"
            "targets: null\n"
            "workers: 10\n"
        )
    )
    result = cli_runner.invoke(
        servo_cli, "validate -f invalid.yaml", catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "X Invalid configuration in invalid.yaml" in result.stderr


def test_vegeta_cli_validate_invalid_syntax(
    tmp_path: Path, servo_cli: ServoCLI, cli_runner: CliRunner
) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        ("connections: 10000\n" "descriptions\n\n null\n" "duratio\n\n_   n: 5m\n")
    )
    result = cli_runner.invoke(
        servo_cli, "validate -f invalid.yaml", catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "X Invalid configuration in invalid.yaml" in result.stderr
    assert "could not find expected ':'" in result.stderr


def test_vegeta_cli_version(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(servo_cli, "version vegeta")
    assert result.exit_code == 0
    assert (
        "Vegeta Connector v0.5.0 (Stable)\n"
        "Vegeta load testing connector\n"
        "https://github.com/opsani/vegeta-connector\n"
        "Licensed under the terms of Apache 2.0\n"
    ) in result.stdout


@pytest.mark.xfail
def test_vegeta_cli_version_short(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(servo_cli, "version -s")
    assert result.exit_code == 0
    assert "Vegeta Connector v0.5.0" in result.stdout


def test_vegeta_cli_loadgen(servo_cli: ServoCLI, cli_runner: CliRunner) -> None:
    pass


class TestConnectorEvents:
    class FakeConnector(Connector):
        @event(handler=True)
        def example_event(self) -> None:
            return 12345
        
        @event(handler=True)
        def get_event_context(self) -> EventContext:
            return self.event_context

        class Config:
            extra = Extra.allow

    class AnotherFakeConnector(FakeConnector):
        @event(handler=True)
        def another_example_event(self) -> str:
            return "example_event"

    def test_event_registration(self) -> None:
        assert _events is not None
        assert _events["example_event"]

    def test_event_inheritance(self) -> None:
        assert _events is not None
        assert _events["example_event"]

    def test_responds_to_event(self) -> None:
        assert TestConnectorEvents.FakeConnector.responds_to_event("example_event")
        assert not TestConnectorEvents.FakeConnector.responds_to_event(
            "another_example_event"
        )

    def test_responds_to_event_subclassing(self) -> None:
        assert TestConnectorEvents.AnotherFakeConnector.responds_to_event(
            "example_event"
        )
        assert TestConnectorEvents.AnotherFakeConnector.responds_to_event(
            "another_example_event"
        )

    async def test_event_invoke(self) -> None:
        config = BaseConfiguration.construct()
        connector = TestConnectorEvents.FakeConnector(config=config)
        event = _events["example_event"]
        results = await connector.run_event_handlers(event, Preposition.ON)
        assert results is not None
        result = results[0]
        assert result.event.name == "example_event"
        assert result.connector == connector
        assert result.value == 12345
    
    async def test_event_context_var(self) -> None:
        config = BaseConfiguration.construct()
        connector = TestConnectorEvents.FakeConnector(config=config)
        event = _events["get_event_context"]
        results = await connector.run_event_handlers(event, Preposition.ON)
        assert results is not None
        result = results[0]
        assert result.event.name == "get_event_context"
        assert result.connector == connector        
        assert result.value == EventContext(event=event, preposition=Preposition.ON)

    async def test_event_invoke_not_supported(self) -> None:
        config = BaseConfiguration.construct()
        connector = TestConnectorEvents.FakeConnector(config=config)
        result = await connector.run_event_handlers("unknown_event", Preposition.ON)
        assert result is None

    def test_event_dispatch_standalone(self) -> None:
        config = BaseConfiguration.construct()
        connector = TestConnectorEvents.FakeConnector(config=config)
        event = _events["example_event"]

        # Dispatch back to self
        results = connector.dispatch_event(event)
        assert results is not None
        result = results[0]
        assert result.event.name == "example_event"
        assert result.connector == connector
        assert result.value == 12345

    async def test_event_dispatch_standalone(self) -> None:
        config = BaseConfiguration.construct()
        connector = TestConnectorEvents.FakeConnector(config=config)
        fake_connector = TestConnectorEvents.AnotherFakeConnector(
            config=config, __connectors__=[connector]
        )
        # Dispatch to peer
        results = await fake_connector.dispatch_event("example_event")
        assert results is not None
        result = results[0]
        assert result.event.name == "example_event"
        assert result.connector == connector
        assert result.value == 12345
    
    def test_event_context_str_comparison(self) -> None:
        assert _events is not None
        event = _events["example_event"]
        context = EventContext(event=event, preposition=Preposition.ON)
        assert context == "example_event"
        assert context == "on:example_event"
        assert context != "before:example_event"
        assert context != "after:example_event"
