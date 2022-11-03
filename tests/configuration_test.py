import json
import pathlib

import pytest

import servo
import tests


async def test_config(servo_yaml: pathlib.Path) -> None:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())
    assembly_ = await servo.assembly.Assembly.assemble(config_file=servo_yaml)


@pytest.mark.parametrize(
    ("name"), ["a", "1234", "app-name", "APP-NAME", "name", "this.that.the.other"]
)
def test_validate_name(name: str):
    # will raise on failure
    servo.configuration.OpsaniOptimizer(id=f"test.com/{name}", token="foo")


def test_optimizer_from_string() -> None:
    optimizer = servo.configuration.OpsaniOptimizer.parse_obj(
        {"id": "dev.opsani.com/awesome-app", "token": "8675309"}
    )
    assert isinstance(optimizer, servo.configuration.OpsaniOptimizer)
    assert optimizer.organization == "dev.opsani.com"
    assert optimizer.name == "awesome-app"


def test_setting_url() -> None:
    optimizer = servo.configuration.OpsaniOptimizer.parse_obj(
        {"id": "dev.opsani.com/awesome-app", "token": "8675309"}
    )
    assert isinstance(optimizer, servo.configuration.OpsaniOptimizer)
    assert optimizer.organization == "dev.opsani.com"
    assert optimizer.name == "awesome-app"


def test_token_exports_to_json() -> None:
    optimizer = servo.configuration.OpsaniOptimizer.parse_obj(
        {"id": "dev.opsani.com/awesome-app", "token": "8675309"}
    )
    assert isinstance(optimizer, servo.configuration.OpsaniOptimizer)
    parsed_optimizer = json.loads(optimizer.json())
    assert parsed_optimizer["token"] == "8675309"


def test_base_url_stripping() -> None:
    optimizer = servo.configuration.OpsaniOptimizer.parse_obj(
        {
            "id": "dev.opsani.com/awesome-app",
            "token": "8675309",
            "base_url": "https://foo.opsani.com/",
        }
    )
    assert optimizer.base_url == "https://foo.opsani.com"
    assert (
        optimizer.url
        == "https://foo.opsani.com/accounts/dev.opsani.com/applications/awesome-app/"
    )
