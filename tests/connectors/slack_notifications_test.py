from os import name
import pathlib
import pytest

import servo.assembly
import servo.errors
import servo.logging

from servo.connectors.slack_notifications import (
    SlackNotificationsConnector, 
    SlackNotificationsConfiguration, 
    SlackWebApiNotifier
)

import tests.helpers

@pytest.fixture
def invalid_auth_webapi_config(monkeypatch):
    monkeypatch.setenv("SLACK_WEB_API_NOTIFIER_CHANNEL_ID", "test")
    monkeypatch.setenv("SLACK_WEB_API_NOTIFIER_BOT_TOKEN", "test")

@pytest.fixture
def invalid_auth_webhook_config(monkeypatch):
    monkeypatch.setenv("SLACK_INCOMING_WEBHOOK_NOTIFIER_URL", "https://localhost/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX")

@pytest.fixture
def config_yaml() -> str:
    """Simple test configuration. Note it must be returned in yaml due to the use of SecretStr from env vars,
    dumping to yaml causes the env var secret to become hidden by a password string (**********) written to
    the yaml file
    """

    return '\n'.join([
        "slack_notifications:",
        "- name: test",
        "  events: ['measure', 'after:describe']",
    ])

@pytest.fixture
async def assembly(servo_yaml: pathlib.Path, config_yaml) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
            "measure": tests.helpers.MeasureConnector,
        }
    )
    gen_config = config_model.generate()
    servo_yaml.write_text(f"{gen_config.yaml()}{config_yaml}")

    optimizer = servo.Optimizer(
        id="servox.opsani.com/tests",
        token="00000000-0000-0000-0000-000000000000",

    )
    assembly_ = await servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly_

@pytest.mark.skip(reason="Testing WIP")
@pytest.mark.web_api
async def test_slack_web_api_error(invalid_auth_webapi_config, assembly: servo.assembly.Assembly):
    # Verify slack error doesn't disrupt normal flow of servo
    # TODO: add log sink, validate expected error
    await assembly.servos[0].dispatch_event("measure")

@pytest.mark.skip(reason="Testing WIP")
@pytest.mark.incoming_webhook
async def test_slack_web_hook_error(invalid_auth_webhook_config, assembly: servo.assembly.Assembly):
    # Verify slack error doesn't disrupt normal flow of servo
    # TODO: add log sink, validate expected error
    await assembly.servos[0].dispatch_event("measure")

@pytest.mark.skip(reason="Testing WIP")
@pytest.mark.incoming_webhook
@pytest.mark.web_api
async def test_slack_notification(assembly: servo.assembly.Assembly):
    await assembly.servos[0].dispatch_event("measure")


# test event errors
@pytest.mark.skip(reason="Testing WIP")
@pytest.mark.incoming_webhook
@pytest.mark.web_api
async def test_slack_notification_handler_error(assembly: servo.assembly.Assembly, mocker):
    connector = assembly.servos[0].get_connector("adjust")
    on_handler = connector.get_event_handlers("describe")[0]
    mock = mocker.patch.object(on_handler, "handler")
    mock.side_effect = servo.errors.EventError()
    await assembly.servos[0].dispatch_event("describe", return_exceptions=True)
