from os import name
import pathlib
import pytest

import servo.assembly
from servo.connectors.slack_notifications import (
    SlackNotificationsConnector, 
    SlackNotificationsConfiguration, 
    SlackWebApiNotifier
)

import tests.helpers

@pytest.fixture
def config(monkeypatch) -> SlackNotificationsConfiguration:
    monkeypatch.setenv("SLACK_WEB_API_NOTIFIER_CHANNEL_ID", "test")
    monkeypatch.setenv("SLACK_WEB_API_NOTIFIER_BOT_TOKEN", "test")
    return [
        SlackWebApiNotifier(name="test", events=["after:adjust"]) # Rest of config derived from .env
    ]

@pytest.fixture
async def assembly(servo_yaml: pathlib.Path, config) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
            "slack_notifications": SlackNotificationsConnector,
        }
    )
    gen_config = config_model.generate()
    gen_config.slack_notifications = config
    servo_yaml.write_text(gen_config.yaml())

    optimizer = servo.Optimizer(
        id="servox.opsani.com/tests",
        token="00000000-0000-0000-0000-000000000000",

    )
    assembly_ = await servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly_

@pytest.mark.slack_integration
async def test_slack_notification(assembly: servo.assembly.Assembly):
    await assembly.servos[0].dispatch_event("adjust")
