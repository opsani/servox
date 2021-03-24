import pathlib
import json

import servo
import tests
import pytest


async def test_config(servo_yaml: pathlib.Path) -> None:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    optimizer = servo.configuration.Optimizer(
        id="servox.opsani.com/tests",
        token="00000000-0000-0000-0000-000000000000",
    )
    assembly_ = await servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )

@pytest.mark.parametrize(
    ('name'),
    [
        'a',
        '1234',
        'app-name',
        'APP-NAME',
        'name',
        'this.that.the.other'
    ]
)
def test_validate_name(name: str):
    # will raise on failure
    servo.configuration.Optimizer(id=f'test.com/{name}', token='foo')

def test_optimizer_from_string() -> None:
    optimizer = servo.configuration.Optimizer.parse_obj({'id': 'dev.opsani.com/awesome-app', 'token': '8675309'})
    assert isinstance(optimizer, servo.configuration.Optimizer)
    assert optimizer.organization == 'dev.opsani.com'
    assert optimizer.name == 'awesome-app'

def test_setting_url() -> None:
    optimizer = servo.configuration.Optimizer.parse_obj({'id': 'dev.opsani.com/awesome-app', 'token': '8675309'})
    assert isinstance(optimizer, servo.configuration.Optimizer)
    assert optimizer.organization == 'dev.opsani.com'
    assert optimizer.name == 'awesome-app'

def test_token_exports_to_json() -> None:
    optimizer = servo.configuration.Optimizer.parse_obj({'id': 'dev.opsani.com/awesome-app', 'token': '8675309'})
    assert isinstance(optimizer, servo.configuration.Optimizer)
    parsed_optimizer = json.loads(optimizer.json())
    assert parsed_optimizer['token'] == '8675309'
