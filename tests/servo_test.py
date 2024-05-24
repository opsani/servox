import asyncio
import inspect
import json
import os
import ssl
from inspect import Signature
from pathlib import Path
from typing import List, Tuple, Type, get_args
from httpcore import Origin

from devtools import debug
import httpx
import pydantic
import pydantic_settings
import pytest
import respx
import yaml
from pydantic import Extra, ValidationError

import servo
from servo import (
    BaseServoConfiguration,
    Duration,
    ErrorSeverity,
    __cryptonym__,
    __version__,
)
from servo.assembly import Assembly
from servo.configuration import (
    BaseConfiguration,
    ChecksConfiguration,
    CommonConfiguration,
    OpsaniOptimizer,
    Timeouts,
)
from servo.connector import BaseConnector
from servo.connectors.vegeta import VegetaConnector, VegetaConfiguration
from servo.errors import *
from servo.events import (
    EventResult,
    Preposition,
    _events,
    after_event,
    before_event,
    create_event,
    event,
    on_event,
)
from servo.servo import Events, Servo
from servo.types import Control, Description, Measurement
from tests.helpers import api_mock, MeasureConnector, environment_overrides


def test_version():
    assert __version__


class FirstTestServoConnector(BaseConnector):
    attached: bool = False
    started_up: bool = False

    @event(handler=True)
    async def this_is_an_event(self) -> str:
        return "this is the result"

    @after_event(Events.adjust)
    def adjust_handler(self, results: List[EventResult]) -> None:
        return "adjusting!"

    @before_event(Events.measure)
    def do_something_before_measuring(
        self, metrics: List[str] = [], control: Control = Control()
    ) -> None:
        return "measuring!"

    @before_event(Events.promote)
    def run_before_promotion(self) -> None:
        return "about to promote!"

    @on_event(Events.promote)
    def run_on_promotion(self) -> None:
        pass

    @after_event(Events.promote)
    def run_after_promotion(self, results: List[EventResult]) -> None:
        return "promoted!!"

    @on_event(Events.attach)
    def handle_attach(self, servo_: servo.Servo) -> None:
        self.attached = True

    @on_event(Events.detach)
    def handle_detach(self, servo_: servo.Servo) -> None:
        pass

    @on_event(Events.startup)
    def handle_startup(self) -> None:
        self.started_up = True

    @on_event(Events.shutdown)
    def handle_shutdown(self) -> None:
        pass

    model_config = pydantic.ConfigDict(extra="allow")


class SecondTestServoConnector(BaseConnector):
    @on_event()
    def this_is_an_event(self) -> str:
        return "this is a different result"

    @event(handler=True)
    async def another_event(self) -> None:
        pass


@pytest.fixture()
def optimizer_config() -> dict[str, str]:
    return {"id": "dev.opsani.com/servox", "token": "1234556789"}


@pytest.fixture()
async def assembly(servo_yaml: Path, optimizer_config: dict[str, str]) -> Assembly:
    config = {
        "optimizer": optimizer_config,
        "connectors": ["first_test_servo", "second_test_servo"],
        "first_test_servo": {},
        "second_test_servo": {},
    }
    servo_yaml.write_text(yaml.dump(config))

    # TODO: Can't pass in like this, needs to be fixed
    assembly = await Assembly.assemble(config_file=servo_yaml)
    return assembly


@pytest.fixture()
def test_servo(assembly: Assembly) -> Servo:
    return assembly.servos[0]


def test_all_connector_types() -> None:
    c = Assembly.model_construct().all_connector_types()
    assert FirstTestServoConnector in c


async def test_servo_routes(test_servo: Servo) -> None:
    first_connector = test_servo.get_connector("first_test_servo")
    assert first_connector.name == "first_test_servo"
    assert first_connector.__class__._name == "FirstTestServo"
    results = await test_servo.dispatch_event(
        "this_is_an_event", include=[first_connector]
    )
    assert len(results) == 1
    assert results[0].value == "this is the result"


def test_servo_routes_and_connectors_reference_same_objects(test_servo: Servo) -> None:
    connector_ids = list(map(lambda c: id(c), test_servo.__connectors__))
    assert connector_ids
    route_ids = list(map(lambda c: id(c), test_servo.connectors))
    assert route_ids
    assert connector_ids == (route_ids + [id(test_servo)])

    # Verify each child has correct references
    for conn in test_servo.__connectors__:
        subconnector_ids = list(map(lambda c: id(c), conn.__connectors__))
        assert subconnector_ids == connector_ids


def test_servo_and_connectors_share_pubsub_exchange(test_servo: Servo) -> None:
    exchange = test_servo.pubsub_exchange
    for connector in test_servo.__connectors__:
        assert connector.pubsub_exchange == exchange
        assert id(connector.pubsub_exchange) == id(exchange)


async def test_dispatch_event(test_servo: Servo) -> None:
    results = await test_servo.dispatch_event("this_is_an_event")
    assert len(results) == 2
    assert results[0].value == "this is the result"


async def test_dispatch_event_first(test_servo: Servo) -> None:
    result = await test_servo.dispatch_event("this_is_an_event", first=True)
    assert isinstance(result, EventResult)
    assert result.value == "this is the result"


async def test_dispatch_event_include(test_servo: Servo) -> None:
    first_connector = test_servo.connectors[0]
    assert first_connector.name == "first_test_servo"
    results = await test_servo.dispatch_event(
        "this_is_an_event", include=[first_connector]
    )
    assert len(results) == 1
    assert results[0].value == "this is the result"


async def test_dispatch_event_exclude(test_servo: Servo) -> None:
    assert len(test_servo.connectors) == 2
    first_connector = test_servo.connectors[0]
    assert first_connector.name == "first_test_servo"
    second_connector = test_servo.connectors[1]
    assert second_connector.name == "second_test_servo"
    event_names = set(_events.keys())
    assert "this_is_an_event" in event_names
    results = await test_servo.dispatch_event(
        "this_is_an_event", exclude=[first_connector]
    )
    assert len(results) == 1
    assert results[0].value == "this is a different result"
    assert results[0].connector == second_connector


def test_get_event_handlers_all(test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    event_handlers = connector.get_event_handlers("promote")
    assert len(event_handlers) == 3
    assert list(map(lambda h: f"{h.preposition}:{h.event}", event_handlers)) == [
        "before:promote",
        "on:promote",
        "after:promote",
    ]


from servo.events import get_event


async def test_add_event_handler_programmatically(mocker, test_servo: Servo) -> None:
    async def fn(self, results: List[EventResult]) -> None:
        print("Test!")

    event = get_event("measure")
    event_handler = FirstTestServoConnector.add_event_handler(
        event, Preposition.after, fn
    )
    spy = mocker.spy(event_handler, "handler")
    await test_servo.dispatch_event("measure")
    spy.assert_called_once()


async def test_before_event(mocker, test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("measure", Preposition.before)[0]
    spy = mocker.spy(event_handler, "handler")
    await test_servo.dispatch_event("measure")
    spy.assert_called_once()


async def test_after_event(mocker, test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    spy = mocker.spy(event_handler, "handler")
    await test_servo.dispatch_event("promote")
    await asyncio.sleep(0.1)
    spy.assert_called_once()


async def test_on_event(mocker, test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    assert connector
    assert test_servo.connectors
    event_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    spy = mocker.spy(event_handler, "handler")
    await test_servo.dispatch_event("promote")
    spy.assert_called_once()


async def test_cancellation_of_event_from_before_handler(mocker, test_servo: Servo):
    connector = test_servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.before)[0]
    on_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    after_spy = mocker.spy(after_handler, "handler")

    # Catch logs
    messages = []
    connector.logger.add(lambda m: messages.append(m), level=0)

    # Mock the before handler to throw a cancel exception
    mock = mocker.patch.object(before_handler, "handler")
    mock.side_effect = EventCancelledError("it burns when I pee", connector=connector)
    results = await test_servo.dispatch_event("promote")

    # Check that on and after callbacks were never called
    on_spy.assert_not_called()
    after_spy.assert_not_called()

    # Check the results
    assert len(results) == 0
    assert messages[0].record["level"].name == "WARNING"
    assert (
        messages[0].record["message"]
        == "event cancelled by before event handler on connector \"first_test_servo\": (EventCancelledError('it burns when I pee'),)"
    )


async def test_cannot_cancel_from_on_handlers_warning(mocker, test_servo: Servo):
    connector = test_servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.on)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()

    messages = []
    connector.logger.add(lambda m: messages.append(m), level=0)
    await test_servo.dispatch_event("promote", return_exceptions=True)
    assert messages[0].record["level"].name == "WARNING"
    assert (
        messages[0].record["message"]
        == "Cannot cancel an event from an on handler: event dispatched"
    )


from servo.errors import EventCancelledError


async def test_cannot_cancel_from_on_handlers(mocker, test_servo: Servo):
    connector = test_servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.on)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()
    with pytest.raises(ExceptionGroup) as error:
        await test_servo.dispatch_event("promote")
    assert str(error.value.exceptions[0]) == "Cannot cancel an event from an on handler"


async def test_cannot_cancel_from_after_handlers_warning(mocker, test_servo: Servo):
    connector = test_servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.after)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()

    with pytest.raises(ExceptionGroup) as error:
        await test_servo.dispatch_event("promote")
    assert (
        str(error.value.exceptions[0]) == "Cannot cancel an event from an after handler"
    )


async def test_after_handlers_are_not_called_on_failure_raises(
    mocker, test_servo: Servo
):
    connector = test_servo.get_connector("first_test_servo")
    after_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    spy = mocker.spy(after_handler, "handler")

    # Mock the before handler to raise an EventError
    on_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    mock = mocker.patch.object(on_handler, "handler")
    mock.side_effect = EventError()
    with pytest.raises(ExceptionGroup) as error:
        await test_servo.dispatch_event("promote", return_exceptions=False)

    assert isinstance(error.value.exceptions[0], EventError)
    spy.assert_not_called()


async def test_after_handlers_are_called_on_failure(mocker, test_servo: Servo):
    connector = test_servo.get_connector("first_test_servo")
    after_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    spy = mocker.spy(after_handler, "handler")

    # Mock the before handler to raise an EventError
    on_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    mock = mocker.patch.object(on_handler, "handler")
    mock.side_effect = EventError()
    results = await test_servo.dispatch_event("promote", return_exceptions=True)
    await asyncio.sleep(0.1)

    spy.assert_called_once()

    # Check the results
    assert len(results) == 1
    result = results[0]
    assert isinstance(result.value, EventError)
    assert result.created_at is not None
    assert result.handler.handler == mock
    assert result.connector == connector
    assert result.event.name == "promote"
    assert result.preposition == Preposition.on


async def test_dispatching_specific_prepositions(mocker, test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.before)[0]
    before_spy = mocker.spy(before_handler, "handler")
    on_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    after_spy = mocker.spy(after_handler, "handler")
    await test_servo.dispatch_event("promote", _prepositions=Preposition.on)
    before_spy.assert_not_called()
    on_spy.assert_called_once()
    after_spy.assert_not_called()


async def test_dispatching_multiple_specific_prepositions(
    mocker, test_servo: Servo
) -> None:
    connector = test_servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.before)[0]
    before_spy = mocker.spy(before_handler, "handler")
    on_handler = connector.get_event_handlers("promote", Preposition.on)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.after)[0]
    after_spy = mocker.spy(after_handler, "handler")
    await test_servo.dispatch_event(
        "promote", _prepositions=Preposition.on | Preposition.before
    )
    before_spy.assert_called_once()
    on_spy.assert_called_once()
    after_spy.assert_not_called()


@api_mock
async def test_startup_event(mocker, test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    await test_servo.startup()
    assert connector.started_up == True


@api_mock
async def test_startup_starts_pubsub_exchange(mocker, test_servo: Servo) -> None:
    test_servo.get_connector("first_test_servo")
    assert not test_servo.pubsub_exchange.running
    await test_servo.startup()
    assert test_servo.pubsub_exchange.running
    await test_servo.pubsub_exchange.shutdown()


@api_mock
async def test_shutdown_event(mocker, test_servo: Servo) -> None:
    await test_servo.startup()
    connector = test_servo.get_connector("first_test_servo")
    on_handler = connector.get_event_handlers("shutdown", Preposition.on)[0]
    on_spy = mocker.spy(on_handler, "handler")
    await test_servo.shutdown()
    on_spy.assert_called()


@api_mock
async def test_shutdown_event_stops_pubsub_exchange(test_servo: Servo) -> None:
    await test_servo.startup()
    assert test_servo.pubsub_exchange.running
    await test_servo.shutdown()
    assert not test_servo.pubsub_exchange.running


async def test_dispatching_event_that_doesnt_exist(test_servo: Servo) -> None:
    with pytest.raises(KeyError) as error:
        await test_servo.dispatch_event(
            "this_is_not_an_event", _prepositions=Preposition.on
        )
    assert str(error.value) == "'this_is_not_an_event'"


##
# Test event handlers


async def test_event(): ...


def test_creating_event_programmatically(random_string: str) -> None:
    signature = Signature.from_callable(test_event)
    create_event(random_string, signature)
    event = _events[random_string]
    assert event.name == random_string
    assert event.signature == signature


def test_creating_event_programmatically_from_callable(random_string: str) -> None:
    create_event("test_creating_event_programmatically_from_callable", test_event)
    event = _events["test_creating_event_programmatically_from_callable"]
    assert event.name == "test_creating_event_programmatically_from_callable"
    assert event.signature == Signature.from_callable(test_event)


def test_redeclaring_an_existing_event_fails() -> None:
    with pytest.raises(ValueError) as error:

        class InvalidConnector:
            @event("adjust")
            def invalid_adjust(self) -> None:
                pass

    assert error
    assert str(error.value) == "Event 'adjust' has already been created"


def test_registering_event_with_wrong_handler_fails() -> None:
    with pytest.raises(TypeError) as error:

        class InvalidConnector:
            @on_event("adjust")
            def invalid_adjust(self, *args, **kwargs) -> dict:
                pass

    assert error
    assert (
        str(error.value)
        == """invalid event handler "adjust": incompatible return type annotation "<class 'dict'>" in callable signature "(self, *args, **kwargs) -> dict", expected "servo.types.Description\""""
    )


def test_event_decorator_disallows_var_positional_args() -> None:
    with pytest.raises(TypeError) as error:

        class InvalidConnector:
            @event("failio")
            async def invalid_event(self, *args) -> None:
                pass

    assert error
    assert (
        str(error.value)
        == "Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
    )


def test_registering_event_handler_with_missing_positional_param_fails() -> None:
    with pytest.raises(TypeError) as error:

        @on_event("adjust")
        def invalid_adjust(self) -> Description:
            pass

    assert error
    expected_error_substrings = [
        "invalid event handler",
        'missing required parameter "adjustments"',
    ]
    for expected_error_substring in expected_error_substrings:
        assert expected_error_substring in str(error.value)


def test_registering_event_handler_with_missing_keyword_param_fails() -> None:
    with pytest.raises(TypeError) as error:

        @on_event("measure")
        def invalid_measure(self, *, control: Control = Control()) -> Measurement:
            pass

    assert error
    expected_error_substrings = [
        "invalid event handler",
        'missing required parameter "metrics"',
    ]
    for expected_error_substring in expected_error_substrings:
        assert expected_error_substring in str(error.value)


def test_registering_event_handler_with_missing_keyword_param_succeeds_with_var_keywords() -> (
    None
):
    @on_event("measure")
    def invalid_measure(self, *, control: Control = Control(), **kwargs) -> Measurement:
        pass


def test_registering_event_handler_with_too_many_positional_params_fails() -> None:
    with pytest.raises(TypeError) as error:

        @on_event("startup")
        def invalid_measure(self, invalid, /) -> None:
            pass

    assert error
    assert (
        str(error.value)
        == 'invalid event handler "startup": encountered unexpected parameter "invalid" in callable signature "(self, invalid, /) -> None", expected "(self) -> \'None\'"'
    )


def test_registering_event_handler_with_too_many_keyword_params_fails() -> None:
    with pytest.raises(TypeError) as error:

        @on_event("startup")
        def invalid_measure(self, invalid: str, another: int) -> None:
            pass

    assert error
    assert (
        str(error.value)
        == """invalid event handler "startup": encountered unexpected parameters "another and invalid" in callable signature "(self, invalid: str, another: int) -> None", expected "(self) -> 'None'\""""
    )


def test_registering_before_handlers() -> None:
    @before_event("measure")
    def before_measure(
        self, metrics: List[str] = [], control: Control = Control()
    ) -> None:
        pass

    assert before_measure.__event_handler__.event.name == "measure"
    assert before_measure.__event_handler__.preposition == Preposition.before


def test_registering_before_handler_fails_with_extra_args() -> None:
    with pytest.raises(TypeError) as error:

        @before_event("measure")
        def invalid_measure(self, invalid: str, another: int) -> None:
            pass

    assert error
    expected_error_substrings = [
        "invalid before event handler",
        "unexpected parameters",
    ]
    for expected_error_substring in expected_error_substrings:
        assert expected_error_substring in str(error.value)


def test_validation_of_before_handlers_ignores_kwargs() -> None:
    @before_event("measure")
    def before_measure(self, **kwargs) -> None:
        pass

    assert before_measure.__event_handler__.event.name == "measure"
    assert before_measure.__event_handler__.preposition == Preposition.before


def test_validation_of_after_handlers() -> None:
    @after_event("measure")
    def after_measure(self, results: List[EventResult]) -> None:
        pass

    assert after_measure.__event_handler__.event.name == "measure"
    assert after_measure.__event_handler__.preposition == Preposition.after


def test_registering_after_handler_fails_with_extra_args() -> None:
    with pytest.raises(TypeError) as error:

        @after_event("measure")
        def invalid_measure(
            self, results: List[EventResult], invalid: str, another: int
        ) -> None:
            pass

    assert error
    assert (
        str(error.value)
        == 'invalid after event handler "after:measure": encountered unexpected parameters "another and invalid" in callable signature "(self, results: List[servo.events.EventResult], invalid: str, another: int) -> None", expected "(self, results: \'List[EventResult]\') -> \'None\'"'
    )


def test_validation_of_after_handlers_ignores_kwargs() -> None:
    @after_event("measure")
    def after_measure(self, results: List[EventResult], **kwargs) -> None:
        pass

    assert after_measure.__event_handler__.event.name == "measure"
    assert after_measure.__event_handler__.preposition == Preposition.after


@pytest.mark.usefixtures("optimizer_env")
class TestAssembly:
    async def test_assemble_assigns_optimizer_to_connectors(
        self, servo_yaml: Path, optimizer_config: dict[str, str]
    ):
        config = {
            "optimizer": optimizer_config,
            "connectors": {"vegeta": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        assembly = await Assembly.assemble(config_file=servo_yaml)

        assert len(assembly.servos) == 1
        assert len(assembly.servos[0].connectors) == 1
        test_servo = assembly.servos[0]

        optimizer = OpsaniOptimizer(**optimizer_config)
        assert test_servo.config.optimizer, "optimizer should not be null"
        assert test_servo.config.optimizer == optimizer
        connector = test_servo.connectors[0]
        assert connector._optimizer == optimizer

    async def test_aliased_connectors_produce_schema(
        self, servo_yaml: Path, mocker
    ) -> None:
        mocker.patch.object(Servo, "version", "100.0.0")
        mocker.patch.object(VegetaConnector, "version", "100.0.0")

        config = {
            "connectors": {"vegeta": "vegeta", "other": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
            "other": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        assembly = await Assembly.assemble(config_file=servo_yaml)
        DynamicServoSettings = assembly.servos[0].config.__class__

        schema = DynamicServoSettings.model_json_schema()

        # Description on parent class can be squirrely
        assert schema == {
            "$defs": {
                "AppdynamicsOptimizer": {
                    "additionalProperties": False,
                    "properties": {
                        "optimizer_id": {"title": "Optimizer Id", "type": "string"},
                        "tenant_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Tenant Id",
                        },
                        "base_url": {
                            "anyOf": [
                                {"format": "uri", "minLength": 1, "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Base Url",
                        },
                        "client_id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Client Id",
                        },
                        "client_secret": {
                            "anyOf": [
                                {
                                    "format": "password",
                                    "type": "string",
                                    "writeOnly": True,
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Client Secret",
                        },
                        "connection_file": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Connection File",
                        },
                        "token": {
                            "anyOf": [
                                {
                                    "format": "password",
                                    "type": "string",
                                    "writeOnly": True,
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Token",
                        },
                        "url": {
                            "anyOf": [
                                {"format": "uri", "minLength": 1, "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Url",
                        },
                        "token_url": {
                            "anyOf": [
                                {"format": "uri", "minLength": 1, "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Token Url",
                        },
                    },
                    "required": ["optimizer_id"],
                    "title": "AppdynamicsOptimizer",
                    "type": "object",
                },
                "BackoffConfigurations": {
                    "additionalProperties": {"$ref": "#/$defs/BackoffSettings"},
                    "description": "A mapping of named backoff configurations.",
                    "title": "BackoffConfigurations",
                    "type": "object",
                },
                "BackoffSettings": {
                    "additionalProperties": False,
                    "description": "BackoffSettings objects model configuration of backoff and retry policies.\n\nSee https://github.com/litl/backoff",
                    "properties": {
                        "max_time": {
                            "anyOf": [
                                {"format": "duration", "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Max Time",
                        },
                        "max_tries": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "title": "Max Tries",
                        },
                    },
                    "required": ["max_time", "max_tries"],
                    "title": "BackoffSettings Connector Configuration Schema",
                    "type": "object",
                },
                "ChecksConfiguration": {
                    "additionalProperties": False,
                    "description": "ChecksConfiguration models configuration for behavior of the checks flow, such as\nwhether to automatically apply remedies.",
                    "properties": {
                        "connectors": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Connectors to check",
                            "title": "Connectors",
                        },
                        "name": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Filter by name",
                            "title": "Name",
                        },
                        "id": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Filter by ID",
                            "title": "Id",
                        },
                        "tag": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Filter by tag",
                            "title": "Tag",
                        },
                        "quiet": {
                            "default": False,
                            "description": "Do not echo generated output to stdout",
                            "title": "Quiet",
                            "type": "boolean",
                        },
                        "verbose": {
                            "default": False,
                            "description": "Display verbose output",
                            "title": "Verbose",
                            "type": "boolean",
                        },
                        "progressive": {
                            "default": True,
                            "description": "Execute checks and emit output progressively",
                            "title": "Progressive",
                            "type": "boolean",
                        },
                        "wait": {
                            "default": "30m",
                            "description": "Wait for checks to pass",
                            "title": "Wait",
                            "type": "string",
                        },
                        "delay": {
                            "default": "expo",
                            "description": "Delay duration. Requires --wait",
                            "title": "Delay",
                            "type": "string",
                        },
                        "halt_on": {
                            "allOf": [{"$ref": "#/$defs/ErrorSeverity"}],
                            "default": "critical",
                            "description": "Halt running on failure severity",
                        },
                        "remedy": {
                            "default": True,
                            "description": "Automatically apply remedies to failed checks if detected",
                            "title": "Remedy",
                            "type": "boolean",
                        },
                        "check_halting": {
                            "default": False,
                            "description": "Halt to wait for each checks success",
                            "title": "Check Halting",
                            "type": "boolean",
                        },
                    },
                    "title": "Checks Connector Configuration Schema",
                    "type": "object",
                },
                "CommonConfiguration": {
                    "additionalProperties": False,
                    "description": "CommonConfiguration models configuration for the Servo connector and establishes default\nsettings for shared services such as networking and logging.",
                    "properties": {
                        "backoff": {
                            "allOf": [{"$ref": "#/$defs/BackoffConfigurations"}],
                            "default": {
                                "__default__": {"max_time": "10m", "max_tries": None},
                                "connect": {"max_time": "1h", "max_tries": None},
                            },
                        },
                        "proxies": {
                            "anyOf": [
                                {"pattern": "^(https?|all)://", "type": "string"},
                                {
                                    "patternProperties": {
                                        "^(https?|all)://": {
                                            "anyOf": [
                                                {"type": "string"},
                                                {"type": "null"},
                                            ]
                                        }
                                    },
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Proxies",
                        },
                        "timeouts": {
                            "anyOf": [{"$ref": "#/$defs/Timeouts"}, {"type": "null"}],
                            "default": None,
                        },
                        "ssl_verify": {
                            "anyOf": [
                                {"type": "boolean"},
                                {"format": "file-path", "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Ssl Verify",
                        },
                    },
                    "title": "Common Connector Configuration Schema",
                    "type": "object",
                },
                "ErrorSeverity": {
                    "description": "ErrorSeverity is an enumeration the describes the severity of an error\nand establishes semantics about how it should be handled.",
                    "enum": ["warning", "common", "critical"],
                    "title": "ErrorSeverity",
                    "type": "string",
                },
                "OpsaniOptimizer": {
                    "additionalProperties": False,
                    "description": "An Optimizer models an Opsani optimization engines that the Servo can connect to\nin order to access the Opsani machine learning technology for optimizing system infrastructure\nand application workloads.\n\nAttributes:\n    id: A friendly identifier formed by joining the `organization` and the `name` with a slash character\n        of the form `example.com/my-app` or `another.com/app-2`.\n    token: An opaque access token for interacting with the Optimizer via HTTP Bearer Token authentication.\n    base_url: The base URL for accessing the Opsani API. This field is typically only useful to Opsani developers or in the context\n        of deployments with specific contractual, firewall, or security mandates that preclude access to the primary API.\n    url: An optional URL that overrides the computed URL for accessing the Opsani API. This option is utilized during development\n        and automated testing to bind the servo to a fixed URL.",
                    "properties": {
                        "opsani_optimizer": {
                            "pattern": "^([A-Za-z0-9-.]{5,50})/[a-zA-Z\\_\\-\\.0-9]{1,64}$",
                            "title": "Opsani Optimizer",
                            "type": "string",
                        },
                        "token": {
                            "format": "password",
                            "title": "Token",
                            "type": "string",
                            "writeOnly": True,
                        },
                        "base_url": {
                            "default": "https://api.opsani.com",
                            "format": "uri",
                            "minLength": 1,
                            "title": "Base Url",
                            "type": "string",
                        },
                        "url": {
                            "anyOf": [
                                {"format": "uri", "minLength": 1, "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Url",
                        },
                    },
                    "required": ["opsani_optimizer", "token"],
                    "title": "OpsaniOptimizer",
                    "type": "object",
                },
                "TargetFormat": {
                    "enum": ["http", "json"],
                    "title": "TargetFormat",
                    "type": "string",
                },
                "Timeouts": {
                    "additionalProperties": False,
                    "description": "Timeouts models the configuration of timeouts for the HTTPX library, which provides HTTP networking capabilities to the\nservo.\n\nSee https://www.python-httpx.org/advanced/#timeout-configuration",
                    "properties": {
                        "connect": {
                            "anyOf": [
                                {"format": "duration", "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Connect",
                        },
                        "read": {
                            "anyOf": [
                                {"format": "duration", "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Read",
                        },
                        "write": {
                            "anyOf": [
                                {"format": "duration", "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Write",
                        },
                        "pool": {
                            "anyOf": [
                                {"format": "duration", "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Pool",
                        },
                    },
                    "required": ["connect", "read", "write", "pool"],
                    "title": "Timeouts Connector Configuration Schema",
                    "type": "object",
                },
                "VegetaConfiguration": {
                    "additionalProperties": False,
                    "properties": {
                        "description": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": "An optional description of the configuration.",
                            "title": "Description",
                        },
                        "rate": {
                            "description": "Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.",
                            "title": "Rate",
                            "type": "string",
                        },
                        "format": {
                            "allOf": [{"$ref": "#/$defs/TargetFormat"}],
                            "default": "http",
                            "description": "Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.",
                        },
                        "target": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": "Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin.",
                            "title": "Target",
                        },
                        "targets": {
                            "anyOf": [
                                {"format": "file-path", "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk.",
                            "title": "Targets",
                        },
                        "connections": {
                            "default": 10000,
                            "description": "Specifies the maximum number of idle open connections per target host.",
                            "title": "Connections",
                            "type": "integer",
                        },
                        "workers": {
                            "default": 10,
                            "description": "Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.",
                            "title": "Workers",
                            "type": "integer",
                        },
                        "max_workers": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "description": "The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.",
                            "title": "Max Workers",
                        },
                        "max_body": {
                            "default": -1,
                            "description": "Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.",
                            "title": "Max Body",
                            "type": "integer",
                        },
                        "http2": {
                            "default": True,
                            "description": "Specifies whether to enable HTTP/2 requests to servers which support it.",
                            "title": "Http2",
                            "type": "boolean",
                        },
                        "keepalive": {
                            "default": True,
                            "description": "Specifies whether to reuse TCP connections between HTTP requests.",
                            "title": "Keepalive",
                            "type": "boolean",
                        },
                        "insecure": {
                            "default": False,
                            "description": "Specifies whether to ignore invalid server TLS certificates.",
                            "title": "Insecure",
                            "type": "boolean",
                        },
                        "reporting_interval": {
                            "default": "15s",
                            "description": "How often to report metrics during a measurement cycle.",
                            "format": "duration",
                            "title": "Reporting Interval",
                            "type": "string",
                        },
                    },
                    "required": ["rate"],
                    "title": "Vegeta Connector Settings (named vegeta)",
                    "type": "object",
                },
                "VegetaConfiguration__other": {
                    "additionalProperties": False,
                    "properties": {
                        "description": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": "An optional description of the configuration.",
                            "title": "Description",
                        },
                        "rate": {
                            "description": "Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.",
                            "title": "Rate",
                            "type": "string",
                        },
                        "format": {
                            "allOf": [{"$ref": "#/$defs/TargetFormat"}],
                            "default": "http",
                            "description": "Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.",
                        },
                        "target": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "description": "Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin.",
                            "title": "Target",
                        },
                        "targets": {
                            "anyOf": [
                                {"format": "file-path", "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "description": "Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk.",
                            "title": "Targets",
                        },
                        "connections": {
                            "default": 10000,
                            "description": "Specifies the maximum number of idle open connections per target host.",
                            "title": "Connections",
                            "type": "integer",
                        },
                        "workers": {
                            "default": 10,
                            "description": "Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.",
                            "title": "Workers",
                            "type": "integer",
                        },
                        "max_workers": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "description": "The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.",
                            "title": "Max Workers",
                        },
                        "max_body": {
                            "default": -1,
                            "description": "Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.",
                            "title": "Max Body",
                            "type": "integer",
                        },
                        "http2": {
                            "default": True,
                            "description": "Specifies whether to enable HTTP/2 requests to servers which support it.",
                            "title": "Http2",
                            "type": "boolean",
                        },
                        "keepalive": {
                            "default": True,
                            "description": "Specifies whether to reuse TCP connections between HTTP requests.",
                            "title": "Keepalive",
                            "type": "boolean",
                        },
                        "insecure": {
                            "default": False,
                            "description": "Specifies whether to ignore invalid server TLS certificates.",
                            "title": "Insecure",
                            "type": "boolean",
                        },
                        "reporting_interval": {
                            "default": "15s",
                            "description": "How often to report metrics during a measurement cycle.",
                            "format": "duration",
                            "title": "Reporting Interval",
                            "type": "string",
                        },
                    },
                    "required": ["rate"],
                    "title": "Vegeta Connector Settings (named other)",
                    "type": "object",
                },
            },
            "description": "Schema for configuration of Servo v100.0.0 with Vegeta Connector v100.0.0",
            "properties": {
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Name",
                },
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Description",
                },
                "SERVO_UID": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Servo Uid",
                },
                "optimizer": {
                    "anyOf": [
                        {"$ref": "#/$defs/AppdynamicsOptimizer"},
                        {"$ref": "#/$defs/OpsaniOptimizer"},
                    ],
                    "default": {},
                    "title": "Optimizer",
                },
                "connectors": {
                    "anyOf": [
                        {"items": {"type": "string"}, "type": "array"},
                        {"additionalProperties": {"type": "string"}, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "description": "An optional, explicit configuration of the active connectors.\n\nConfigurable as either an array of connector identifiers (names or class) or\na dictionary where the keys specify the key path to the connectors configuration\nand the values identify the connector (by name or class name).",
                    "examples": [
                        ["kubernetes", "prometheus"],
                        {"gateway_prom": "prometheus", "staging_prom": "prometheus"},
                    ],
                    "title": "Connectors",
                },
                "no_diagnostics": {
                    "default": True,
                    "description": "Do not poll the Opsani API for diagnostics",
                    "title": "No Diagnostics",
                    "type": "boolean",
                },
                "settings": {
                    "anyOf": [
                        {"$ref": "#/$defs/CommonConfiguration"},
                        {"type": "null"},
                    ],
                    "description": "Configuration of the Servo connector",
                },
                "checks": {
                    "anyOf": [
                        {"$ref": "#/$defs/ChecksConfiguration"},
                        {"type": "null"},
                    ],
                    "description": "Configuration of Checks behavior",
                },
                "other": {
                    "anyOf": [
                        {"$ref": "#/$defs/VegetaConfiguration__other"},
                        {"type": "null"},
                    ]
                },
                "vegeta": {
                    "anyOf": [{"$ref": "#/$defs/VegetaConfiguration"}, {"type": "null"}]
                },
            },
            "required": ["other", "vegeta"],
            "title": "Servo Configuration Schema",
            "type": "object",
        }

    @pytest.mark.usefixtures("optimizer_env")
    async def test_aliased_connectors_get_distinct_env_configuration(
        self, servo_yaml: Path
    ) -> None:
        config = {
            "connectors": {"vegeta": "vegeta", "other": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
            "other": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        assembly = await Assembly.assemble(config_file=servo_yaml)
        DynamicServoConfiguration = assembly.servos[0].config.__class__

        # Grab the vegeta field and check it
        vegeta_field = DynamicServoConfiguration.model_fields["vegeta"]
        vegeta_settings_type = vegeta_field.annotation
        assert (
            str(vegeta_settings_type)
            == "typing.Optional[servo.assembly.VegetaConfiguration]"
        )
        vegeta_settings_inner_type = get_args(vegeta_settings_type)[0]

        # Grab the other field and check it
        other_field = DynamicServoConfiguration.model_fields["other"]
        other_settings_type = other_field.annotation
        assert (
            str(other_settings_type)
            == "typing.Optional[servo.assembly.VegetaConfiguration__other]"
        )

        with environment_overrides({"SERVO_DESCRIPTION": "this description"}):
            assert os.environ["SERVO_DESCRIPTION"] == "this description"
            s = DynamicServoConfiguration(
                other=None,
                vegeta=vegeta_settings_inner_type(
                    rate=10, target="http://example.com/"
                ),
            )
            assert s.description == "this description"

        # Make sure the incorrect case does pass
        with environment_overrides({"SERVO_RATE": "invalid"}):
            with pytest.raises(ValidationError):
                vegeta_settings_inner_type(target="https://foo.com/")

        # Try setting values via env
        with environment_overrides(
            {
                "SERVO_OTHER_RATE": "100/1s",
                "SERVO_OTHER_TARGET": "https://opsani.com/servox",
            }
        ):
            s = DynamicServoConfiguration(
                vegeta=vegeta_settings_inner_type(
                    rate=10, target="http://example.com/"
                ),
            )
            assert s.other.rate == "100/1s"
            assert s.other.target == "https://opsani.com/servox"


async def test_generating_schema_with_test_connectors(
    optimizer_env: None, servo_yaml: Path
) -> None:
    assembly = await Assembly.assemble(config_file=servo_yaml)
    assert len(assembly.servos) == 1, "servo was not assembled"
    DynamicServoConfiguration = assembly.servos[0].config.__class__
    DynamicServoConfiguration.model_json_schema()
    # NOTE: Covers naming conflicts between settings models -- will raise if misconfigured


def test_optimizer_required():
    with pytest.raises(ValidationError) as e:
        BaseServoConfiguration(
            connectors={"test_vegeta": "VegetaConnector"},
        )

    assert "3 validation errors for Abstract Servo Configuration Schema" in str(e.value)
    assert e.value.errors()[0]["loc"] == (
        "optimizer",
        "AppdynamicsOptimizer",
        "optimizer_id",
    )
    assert e.value.errors()[0]["msg"] == "Field required"


# automatically tests for presence of servo_connectors env var
class FooServoConfiguration(BaseServoConfiguration, pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="servo_", env_nested_delimiter="_"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        _: Type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.EnvSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> Tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        assert "servo_connectors" in env_settings.env_vars
        return init_settings, env_settings, dotenv_settings, file_secret_settings


@pytest.mark.usefixtures("optimizer_env")
class TestServoSettings:
    def test_forbids_extra_attributes(self, optimizer) -> None:
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(forbidden=[], optimizer=optimizer)
            assert "extra fields not permitted" in str(e.value)

    def test_override_optimizer_settings_with_env_vars(self) -> None:
        with environment_overrides({"OPSANI_TOKEN": "abcdefg"}):
            assert os.environ["OPSANI_TOKEN"] is not None
            optimizer = OpsaniOptimizer(id="dsada.com/foo")
            assert optimizer.token.get_secret_value() == "abcdefg"

    def test_set_connectors_with_env_vars(self) -> None:
        with environment_overrides({"SERVO_CONNECTORS": '["measure"]'}):
            assert os.environ["SERVO_CONNECTORS"] is not None

            s = FooServoConfiguration()
            assert s is not None
            assert s.connectors is not None
            assert s.connectors == ["measure"]

    def test_connectors_allows_none(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors=None,
        )
        assert s.connectors is None

    def test_connectors_allows_set_of_classes(self, optimizer):
        class FooConnector(BaseConnector):
            pass

        class BarConnector(BaseConnector):
            pass

        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={FooConnector, BarConnector},
        )
        assert set(s.connectors) == {"FooConnector", "BarConnector"}

    def test_connectors_rejects_invalid_connector_set_elements(self, optimizer):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={BaseServoConfiguration},
            )
        assert "1 validation error for Abstract Servo Configuration Schema" in str(
            e.value
        )
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == "Value error, Invalid connectors value: <class 'servo.configuration.BaseServoConfiguration'>"
        )

    def test_connectors_allows_set_of_class_names(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={"MeasureConnector", "AdjustConnector"},
        )
        assert set(s.connectors) == {"MeasureConnector", "AdjustConnector"}

    def test_connectors_rejects_invalid_connector_set_class_name_elements(
        self, optimizer
    ):
        with pytest.raises(TypeError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={"servo.servo.BaseServoConfiguration"},
            )
        assert "BaseServoConfiguration is not a Connector subclass" in str(e.value)

    def test_connectors_allows_set_of_keys(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={"vegeta"},
        )
        assert s.connectors == ["vegeta"]

    def test_connectors_allows_dict_of_keys_to_classes(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={"alias": VegetaConnector},
        )
        assert s.connectors == {"alias": "VegetaConnector"}

    def test_connectors_allows_dict_of_keys_to_class_names(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={"alias": "VegetaConnector"},
        )
        assert s.connectors == {"alias": "VegetaConnector"}

    def test_connectors_allows_dict_with_explicit_map_to_default_name(self, optimizer):
        s = BaseServoConfiguration(
            optimizer=optimizer,
            connectors={"vegeta": "VegetaConnector"},
        )
        assert s.connectors == {"vegeta": "VegetaConnector"}

    def test_connectors_allows_dict_with_explicit_map_to_default_class(
        self, optimizer: OpsaniOptimizer
    ):
        s = BaseServoConfiguration(
            connectors={"vegeta": VegetaConnector}, optimizer=optimizer
        )
        assert s.connectors == {"vegeta": "VegetaConnector"}

    def test_connectors_forbids_dict_with_existing_key(self, optimizer):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={"vegeta": "MeasureConnector"},
            )
        assert "1 validation error for Abstract Servo Configuration Schema" in str(
            e.value
        )
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == 'Value error, Name "vegeta" is reserved by `VegetaConnector`'
        )

    @pytest.fixture(autouse=True, scope="session")
    def discover_connectors(self) -> None:
        from servo.connector import ConnectorLoader

        loader = ConnectorLoader()
        for connector in loader.load():
            pass

    def test_connectors_forbids_dict_with_reserved_key(self, optimizer):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={"connectors": "VegetaConnector"},
            )
        assert "1 validation error for Abstract Servo Configuration Schema" in str(
            e.value
        )
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"] == 'Value error, Name "connectors" is reserved'
        )

    def test_connectors_forbids_dict_with_invalid_key(self, optimizer):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={"This Is Not Valid": "VegetaConnector"},
            )
        assert "1 validation error for Abstract Servo Configuration Schema" in str(
            e.value
        )
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == 'Value error, "This Is Not Valid" is not a valid connector name: names may only contain alphanumeric characters, hyphens, slashes, periods, and underscores'
        )

    def test_connectors_rejects_invalid_connector_dict_values(self, optimizer):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                optimizer=optimizer,
                connectors={"whatever": "Not a Real Connector"},
            )
        assert "1 validation error for Abstract Servo Configuration Schema" in str(
            e.value
        )
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == "Value error, Invalid connectors value: Not a Real Connector"
        )


# Test servo config...


@pytest.mark.parametrize("attr", ["connect", "read", "write", "pool"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (60, 60),
        (5.0, 5.0),
        ("30s", 30),
        (Duration("3h4m"), Duration("3h4m")),
    ],
)
def test_valid_timeouts_input(attr, value, expected) -> None:
    kwargs = {attr: value}
    timeouts = Timeouts(**kwargs)
    assert getattr(timeouts, attr) == expected


@pytest.mark.parametrize("attr", ["connect", "read", "write", "pool"])
@pytest.mark.parametrize("value", [[], "not valid", {}])
def test_invalid_timeouts_input(attr, value) -> None:
    with pytest.raises((ValidationError, TypeError)):
        Timeouts(**{attr: value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (60, 60),
        (5.0, 5.0),
        ("30s", 30),
    ],
)
def test_timeouts_parsing(value, expected) -> None:
    config = CommonConfiguration(timeouts=value)
    if value is None:
        assert config.timeouts is None
    else:
        assert config.timeouts == Timeouts(
            connect=Duration(value),
            read=Duration(value),
            write=Duration(value),
            pool=Duration(value),
        )


@pytest.mark.parametrize(
    "proxies",
    [
        None,
        {
            "http://": "http://localhost:8030",
            "https://": "http://localhost:8031",
        },
        {
            "http://": "http://username:password@localhost:8030",
        },
        {
            "all://": "http://localhost:8030",
        },
        {
            "all://example.com": "http://localhost:8030",
        },
        {
            "http://example.com": "http://localhost:8030",
        },
        {
            "all://*example.com": "http://localhost:8030",
        },
        {
            "all://*.example.com": "http://localhost:8030",
        },
        {
            "https://example.com:1234": "http://localhost:8030",
        },
        {
            "all://*:1234": "http://localhost:8030",
        },
        {
            # Route requests through a proxy by default...
            "all://": "http://localhost:8031",
            # Except those for "example.com".
            "all://example.com": None,
        },
        {
            # Route all traffic through a proxy by default...
            "all://": "http://localhost:8030",
            # But don't use proxies for HTTPS requests to "domain.io"...
            "https://domain.io": None,
            # And use another proxy for requests to "example.com" and its subdomains...
            "all://*example.com": "http://localhost:8031",
            # And yet another proxy if HTTP is used,
            # and the "internal" subdomain on port 5550 is requested...
            "http://internal.example.com:5550": "http://localhost:8032",
        },
        {},
    ],
)
def test_valid_proxies(proxies) -> None:
    CommonConfiguration(proxies=proxies)


@pytest.mark.parametrize("proxies", [0.5, "not valid", 1234])
def test_invalid_proxies(proxies) -> None:
    with pytest.raises(ValidationError):
        CommonConfiguration(proxies=proxies)


def test_api_client_options() -> None:
    optimizer = OpsaniOptimizer(id="test.com/foo", token="12345")
    settings = CommonConfiguration(proxies="http://localhost:1234", ssl_verify=False)

    # NOTE: SETTINGS AND OPTIMIZER NOT TOGETHER!!!
    test_servo = Servo(
        config={"settings": settings, "optimizer": optimizer}, connectors=[]
    )
    assert test_servo.config.optimizer, "expected config to have an optimizer"
    assert test_servo.optimizer, "expected to have an optimizer"
    assert test_servo.optimizer == optimizer

    assert test_servo.config.settings, "expected settings"
    assert test_servo.config.settings == settings, "expected settings"
    assert test_servo.config.settings.proxies

    assert test_servo._api_client._timeout == httpx.Timeout(timeout=None)
    assert (
        test_servo._api_client._transport._pool._ssl_context.verify_mode
        == ssl.CERT_NONE
    )
    assert test_servo._api_client._transport._pool._ssl_context.check_hostname == False
    for k, v in test_servo._api_client._mounts.items():
        assert v._pool._proxy_url.scheme == b"http"
        assert v._pool._proxy_url.host == b"localhost"
        assert v._pool._proxy_url.port == 1234


async def test_models() -> None:
    assert MeasureConnector(config={})


async def test_httpx_client_config() -> None:
    optimizer = OpsaniOptimizer(id="test.com/foo", token="12345")
    common = CommonConfiguration(proxies="http://localhost:1234", ssl_verify=False)

    # TODO: init with config that has optimizer, use optimizer + config? allow optimizer=UUU only on Servo class?
    connector = MeasureConnector(config={})

    test_servo = Servo(
        config={"settings": common, "optimizer": optimizer}, connectors=[connector]
    )
    assert connector.optimizer == optimizer
    assert connector._global_config
    assert connector._global_config == common

    for k, v in test_servo._api_client._mounts.items():
        assert k.pattern == "all://"
    assert (
        test_servo._api_client._transport._pool._ssl_context.verify_mode
        == ssl.CERT_NONE
    )
    assert test_servo._api_client._transport._pool._ssl_context.check_hostname == False


def test_backoff_defaults() -> None:
    config = CommonConfiguration()
    assert config.backoff
    assert config.backoff["__default__"]
    assert config.backoff["__default__"].max_time is not None
    assert config.backoff["__default__"].max_time == Duration("10m")
    assert config.backoff["__default__"].max_tries is None


def test_backoff_contexts() -> None:
    contexts = servo.configuration.BackoffConfigurations(
        **{
            "__default__": {"max_time": "10m", "max_tries": None},
            "connect": {"max_time": "1h", "max_tries": None},
        }
    )
    assert contexts

    config = servo.configuration.CommonConfiguration(backoff=contexts)
    assert config


def test_backoff_context() -> None:
    config = CommonConfiguration()
    assert config.backoff
    assert config.backoff.max_time()
    assert config.backoff.max_time("whatever")

    assert config.backoff["__default__"].max_time is not None
    assert config.backoff["__default__"].max_time == Duration("10m")
    assert config.backoff["__default__"].max_tries is None


def test_checks_defaults() -> None:
    checks_config = ChecksConfiguration()
    assert checks_config
    assert checks_config.name is None
    assert checks_config.id is None
    assert checks_config.tag is None
    assert checks_config.quiet == False
    assert checks_config.verbose == False
    assert checks_config.progressive == True
    assert checks_config.wait == Duration("30m")
    assert checks_config.delay == "expo"
    assert checks_config.halt_on == ErrorSeverity.critical
    assert checks_config.remedy == True
    assert checks_config.check_halting == False


@pytest.mark.parametrize(
    ("proxies"),
    [
        "http://localhost:1234",
        {"all://": "http://localhost:1234"},
        {"https://": "http://localhost:1234"},
        {"https://api.opsani.com": "http://localhost:1234"},
        {"https://*.opsani.com": "http://localhost:1234"},
    ],
)
async def test_proxy_utilization(proxies) -> None:
    optimizer = OpsaniOptimizer(id="test.com/foo", token="12345")
    config = CommonConfiguration(proxies=proxies)
    test_servo = Servo(
        config={"settings": config, "optimizer": optimizer}, connectors=[]
    )
    transport = test_servo._api_client._transport_for_url(httpx.URL(optimizer.base_url))
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert transport._pool._proxy_url.origin == Origin(
        scheme=b"http", host=b"localhost", port=1234
    )


def test_codename() -> None:
    assert __cryptonym__


async def test_add_connector(test_servo: Servo) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    assert connector not in test_servo.connectors
    await test_servo.add_connector("whatever", connector)
    assert connector in test_servo.connectors
    assert test_servo.config.whatever == connector.config


async def test_add_connector_sends_attach_event(test_servo: Servo) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    assert connector.attached is False
    await test_servo.add_connector("whatever", connector)
    assert connector.attached is True


async def test_add_connector_can_handle_events(test_servo: Servo) -> None:
    results = await test_servo.dispatch_event("this_is_an_event")
    assert len(results) == 2

    connector = FirstTestServoConnector(config=BaseConfiguration())
    await test_servo.add_connector("whatever", connector)

    results = await test_servo.dispatch_event("this_is_an_event")
    assert len(results) == 3


async def test_add_connector_raises_if_name_exists(test_servo: Servo) -> None:
    connector_1 = FirstTestServoConnector(config=BaseConfiguration())
    await test_servo.add_connector("whatever", connector_1)

    connector_2 = FirstTestServoConnector(config=BaseConfiguration())
    with pytest.raises(ValueError) as error:
        await test_servo.add_connector("whatever", connector_2)

    assert (
        str(error.value)
        == "invalid name: a connector named 'whatever' already exists in the servo"
    )


async def test_remove_connector(test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    assert connector in test_servo.connectors
    assert test_servo.config.first_test_servo == connector.config
    await test_servo.remove_connector(connector)
    assert connector not in test_servo.connectors

    with pytest.raises(AttributeError):
        assert test_servo.config.first_test_servo


async def test_remove_connector_by_name(test_servo: Servo) -> None:
    connector = test_servo.get_connector("first_test_servo")
    assert connector in test_servo.connectors
    assert test_servo.config.first_test_servo == connector.config
    await test_servo.remove_connector("first_test_servo")
    assert connector not in test_servo.connectors

    with pytest.raises(AttributeError):
        assert test_servo.config.first_test_servo


# TODO: shutdown if running
async def test_remove_connector_sends_detach_event(test_servo: Servo, mocker) -> None:
    connector = test_servo.get_connector("first_test_servo")
    on_handler = connector.get_event_handlers("detach", Preposition.on)[0]
    on_spy = mocker.spy(on_handler, "handler")
    await test_servo.remove_connector(connector)
    on_spy.assert_called()


async def test_remove_connector_raises_if_name_does_not_exists(
    test_servo: Servo,
) -> None:
    with pytest.raises(ValueError) as error:
        await test_servo.remove_connector("whatever")

    assert (
        str(error.value)
        == "invalid connector: a connector named 'whatever' does not exist in the servo"
    )


async def test_remove_connector_raises_if_obj_does_not_exists(
    test_servo: Servo,
) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    with pytest.raises(ValueError) as error:
        await test_servo.remove_connector(connector)

    assert (
        str(error.value)
        == "invalid connector: a connector named 'first_test_servo' does not exist in the servo"
    )


async def test_backoff() -> None:
    config = CommonConfiguration(proxies="http://localhost:1234", ssl_verify=False)
    assert config.backoff
    assert config.backoff.max_time() == Duration("10m").total_seconds()
    assert config.backoff.max_time("connect") == Duration("1h").total_seconds()


def test_servo_name_literal(test_servo: Servo) -> None:
    test_servo.name = "hrm"
    assert test_servo.name == "hrm"


def test_servo_name_from_config(optimizer_config: dict[str, str]) -> None:
    config = BaseServoConfiguration(name="archibald", optimizer=optimizer_config)
    test_servo = Servo(config=config, connectors=[])
    assert test_servo.name == "archibald"


def test_servo_name_falls_back_to_optimizer_id(test_servo: Servo) -> None:
    debug("SERVO IS: ", test_servo)
    assert test_servo.name == "dev.opsani.com/servox"
