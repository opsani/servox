import pytest
import servo
import servo.connectors.scripts
import tests.helpers


@pytest.fixture
def configuration() -> servo.connectors.scripts.ScriptsConfiguration:
    return servo.connectors.scripts.ScriptsConfiguration(
        before={"measure": ["echo 123"]},
        after={"measure": ["echo 456"]},
    )


@pytest.fixture
def connector(
    configuration: servo.connectors.scripts.ScriptsConfiguration,
) -> servo.connectors.scripts.ScriptsConnector:
    return servo.connectors.scripts.ScriptsConnector(config=configuration)


async def test_scripts(connector: servo.connectors.scripts.ScriptsConnector) -> None:
    connectors = [
        connector,
        tests.helpers.MeasureConnector(config=servo.BaseConfiguration()),
    ]
    _servo = servo.Servo(
        config={
            "optimizer": servo.OpsaniOptimizer(
                id="dev.opsani.com/servox", token="1234556789"
            )
        },
        connectors=connectors,
        __connectors__=connectors,
    )
    await _servo.dispatch_event("startup")
    result = await _servo.dispatch_event("measure")
    debug("result is", result)
    assert result != None
