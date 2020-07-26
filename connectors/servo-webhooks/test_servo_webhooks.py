import asyncio
import hmac
import hashlib
from typing import List

import pytest
import servo
from servo import BaseConfiguration, Metric, Unit, on_event
from servo.events import EventContext
from servo_webhooks import CLI, Configuration, Connector, Webhook, __version__
import respx

pytestmark = pytest.mark.asyncio

class WebhookEventConnector(servo.Connector):
    @on_event()
    def metrics(self) -> List[Metric]:
        return [
            Metric("throughput", Unit.REQUESTS_PER_MINUTE),
            Metric("error_rate", Unit.PERCENTAGE),
        ]
    
def test_version():
    assert __version__ == '0.1.0'

@respx.mock
async def test_webhook() -> None:
    webhook = Webhook(url="http://localhost:8080/webhook", events="before:measure", secret="testing")
    config = Configuration([webhook])
    connector = Connector(config=config)

    request = respx.post("http://localhost:8080/webhook", status_code=200)
    await connector.dispatch_event("measure")
    assert request.called

@respx.mock
async def test_webhooks() -> None:
    webhook = Webhook(url="http://localhost:8080/webhook", events=["before:measure", "after:adjust"], secret="test")
    config = Configuration([webhook])
    connector = Connector(config=config)

    request = respx.post("http://localhost:8080/webhook", status_code=200)
    await connector.dispatch_event("measure")
    assert request.called

    await connector.dispatch_event("adjust")
    assert request.called

def test_headers_are_added_to_requests() -> None:
    pass

# TODO: Test after:metrics, test schema

@respx.mock
async def test_after_metrics_webhook() -> None:
    webhook = Webhook(url="http://localhost:8080/webhook", events=["after:metrics"], secret="w00t")
    config = Configuration([webhook])
    connector = Connector(config=config)

    request = respx.post("http://localhost:8080/webhook", status_code=200)
    provider = WebhookEventConnector(config=BaseConfiguration())
    provider.__connectors__.append(connector)
    results = await provider.dispatch_event("metrics")

    assert request.called

async def test_after_metrics_content_type() -> None:
    pass
    # Content-Type: application/vnd.opsani.servo.events.after:metrics+json
    # Content-Type: application/vnd.opsani.servo.webhooks+json
    # Content-Type: application/vnd.opsani.servo-webhooks+json
# await asyncio.sleep(2) 

# no colon, wrong casing, no such event, mixed collection (number and strings)
def test_bad_event_inputs() -> None:
    pass

def test_root_configuration() -> None:
    pass

def test_event_body() -> None:
    pass

# TODO: Content-Types and shit

def test_request_schema() -> None:
    pass

@respx.mock
async def test_hmac_signature() -> None:    
    webhook = Webhook(url="http://localhost:8080/webhook", events="after:measure", secret="testing")
    config = Configuration([webhook])
    connector = Connector(config=config)

    info = {}
    def match_and_mock(request, response):
        if request.method != "POST":
            return None

        if "x-servo-signature" in request.headers:
            signature = request.headers["x-servo-signature"]
            body = request.read()
            info.update(dict(signature=signature, body=body))

        return response

    webhook_request = respx.add(match_and_mock, status_code=204)
    await connector.dispatch_event("measure")
    assert webhook_request.called

    expected_signature = info["signature"]
    signature = str(hmac.new("testing".encode(), info["body"], hashlib.sha1).hexdigest())
    assert signature == expected_signature

def test_cancelling_event_from_before_request() -> None:
    pass

class TestCLI:
    def test_list(self) -> None:
        pass
    
    def test_schema(self) -> None:
        pass

    def test_trigger(self) -> None:
        pass
    
    def test_validate(self) -> None:
        pass

# TODO: Test backoff and retry
# TODO: Test generate

def test_generate():
    config = Configuration.generate()
    debug(config.yaml())
    #debug(config.dict(exclude={"webhooks": {'events': {'__all__': {'signature'} }}}))

@pytest.mark.parametrize(
    "event_str,found,resolved",
    [
        ("before:measure", True, "before:measure"),
        ("on:measure", True, "measure"),
        ("measure", True, "measure"),
        ("after:measure", True, "after:measure"),
        ("invalid:adjust", False, None),
        ("before:invalid", False, None),
        ("BEFORE:adjust", False, None),
        ("before:MEASURE", False, None),
        ("", False, None),
        ("nothing", False, None),
    ]
)
def test_from_str(event_str: str, found: bool, resolved: str):
    ec = EventContext.from_str(event_str)
    assert bool(ec) == found
    assert (ec.__str__() if ec else None) == resolved

##
# CLI

# def test_generate(
#     cli_runner: CliRunner, servo_cli: Typer, optimizer_env: None, stub_servo_yaml: Path
# ) -> None:
#     result = cli_runner.invoke(servo_cli, "show metrics", catch_exceptions=False)
#     assert result.exit_code == 0
#     assert re.match("METRIC\\s+UNIT\\s+CONNECTORS", result.stdout)

