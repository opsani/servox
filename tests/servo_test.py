import asyncio
import json
import os
import ssl
from inspect import Signature
from pathlib import Path
from typing import List

import httpcore
import httpx
import pytest
import yaml
from pydantic import Extra, ValidationError

import servo as servox
from servo.errors import *
from servo import BaseServoConfiguration, Duration, __cryptonym__, __version__
from servo.assembly import Assembly
from servo.configuration import BaseConfiguration, Optimizer, ServoConfiguration, Timeouts
from servo.connector import BaseConnector
from servo.connectors.vegeta import VegetaConnector
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
from tests.helpers import MeasureConnector, environment_overrides


def test_version():
    assert __version__


class FirstTestServoConnector(BaseConnector):
    started_up: bool = False

    @event(handler=True)
    async def this_is_an_event(self) -> str:
        return "this is the result"

    @after_event(Events.ADJUST)
    def adjust_handler(self, results: List[EventResult]) -> None:
        return "adjusting!"

    @before_event(Events.MEASURE)
    def do_something_before_measuring(self) -> None:
        return "measuring!"

    @before_event(Events.PROMOTE)
    def run_before_promotion(self) -> None:
        return "about to promote!"

    @on_event(Events.PROMOTE)
    def run_on_promotion(self) -> None:
        pass

    @after_event(Events.PROMOTE)
    def run_after_promotion(self, results: List[EventResult]) -> None:
        return "promoted!!"

    @on_event(Events.STARTUP)
    def handle_startup(self) -> None:
        self.started_up = True

    @on_event(Events.SHUTDOWN)
    def handle_shutdown(self) -> None:
        pass

    class Config:
        # NOTE: Necessary to utilize mocking
        extra = Extra.allow


class SecondTestServoConnector(BaseConnector):
    @on_event()
    def this_is_an_event(self) -> str:
        return "this is a different result"

    @event(handler=True)
    async def another_event(self) -> None:
        pass


@pytest.fixture()
def assembly(servo_yaml: Path) -> Assembly:
    config = {
        "connectors": ["first_test_servo", "second_test_servo"],
        "first_test_servo": {},
        "second_test_servo": {},
    }
    servo_yaml.write_text(yaml.dump(config))

    optimizer = Optimizer(id="dev.opsani.com/servox", token="1234556789")

    assembly = Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly


@pytest.fixture()
def servo(assembly: Assembly) -> Servo:
    return assembly.servos[0]


def test_all_connector_types() -> None:
    c = Assembly.construct().all_connector_types()
    assert FirstTestServoConnector in c


async def test_servo_routes(servo: Servo) -> None:
    first_connector = servo.get_connector("first_test_servo")
    assert first_connector.name == "first_test_servo"
    assert first_connector.__class__.name == "FirstTestServo"
    results = await servo.dispatch_event("this_is_an_event", include=[first_connector])
    assert len(results) == 1
    assert results[0].value == "this is the result"


def test_servo_routes_and_connectors_reference_same_objects(servo: Servo) -> None:
    connector_ids = list(map(lambda c: id(c), servo.__connectors__))
    assert connector_ids
    route_ids = list(map(lambda c: id(c), servo.connectors))
    assert route_ids
    assert connector_ids == (route_ids + [id(servo)])

    # Verify each child has correct references
    for conn in servo.__connectors__:
        subconnector_ids = list(map(lambda c: id(c), conn.__connectors__))
        assert subconnector_ids == connector_ids


async def test_dispatch_event(servo: Servo) -> None:
    results = await servo.dispatch_event("this_is_an_event")
    assert len(results) == 2
    assert results[0].value == "this is the result"


async def test_dispatch_event_first(servo: Servo) -> None:
    result = await servo.dispatch_event("this_is_an_event", first=True)
    assert isinstance(result, EventResult)
    assert result.value == "this is the result"


async def test_dispatch_event_include(servo: Servo) -> None:
    first_connector = servo.connectors[0]
    assert first_connector.name == "first_test_servo"
    results = await servo.dispatch_event("this_is_an_event", include=[first_connector])
    assert len(results) == 1
    assert results[0].value == "this is the result"


async def test_dispatch_event_exclude(servo: Servo) -> None:
    assert len(servo.connectors) == 2
    first_connector = servo.connectors[0]
    assert first_connector.name == "first_test_servo"
    second_connector = servo.connectors[1]
    assert second_connector.name == "second_test_servo"
    event_names = set(_events.keys())
    assert "this_is_an_event" in event_names
    results = await servo.dispatch_event("this_is_an_event", exclude=[first_connector])
    assert len(results) == 1
    assert results[0].value == "this is a different result"
    assert results[0].connector == second_connector


def test_get_event_handlers_all(servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    event_handlers = connector.get_event_handlers("promote")
    assert len(event_handlers) == 3
    assert list(map(lambda h: f"{h.preposition}:{h.event}", event_handlers)) == [
        "before:promote",
        "on:promote",
        "after:promote",
    ]


from servo.events import get_event


async def test_add_event_handler_programmatically(mocker, servo: servo) -> None:
    async def fn(self, results: List[EventResult]) -> None:
        print("Test!")

    event = get_event("measure")
    event_handler = FirstTestServoConnector.add_event_handler(
        event, Preposition.AFTER, fn
    )
    spy = mocker.spy(event_handler, "handler")
    await servo.dispatch_event("measure")
    spy.assert_called_once()


async def test_before_event(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("measure", Preposition.BEFORE)[0]
    spy = mocker.spy(event_handler, "handler")
    await servo.dispatch_event("measure")
    spy.assert_called_once()


async def test_after_event(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    spy = mocker.spy(event_handler, "handler")
    await servo.dispatch_event("promote")
    await asyncio.sleep(0.1)
    spy.assert_called_once()


async def test_on_event(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    spy = mocker.spy(event_handler, "handler")
    await servo.dispatch_event("promote")
    spy.assert_called_once()


async def test_cancellation_of_event_from_before_handler(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.BEFORE)[0]
    on_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    after_spy = mocker.spy(after_handler, "handler")

    # Catch logs
    messages = []
    connector.logger.add(lambda m: messages.append(m), level=0)

    # Mock the before handler to throw a cancel exception
    mock = mocker.patch.object(before_handler, "handler")
    mock.side_effect = EventCancelledError("it burns when I pee")
    results = await servo.dispatch_event("promote")

    # Check that on and after callbacks were never called
    on_spy.assert_not_called()
    after_spy.assert_not_called()

    # Check the results
    assert len(results) == 0
    assert messages[0].record["level"].name == "WARNING"
    assert messages[0].record["message"] == 'event cancelled by before event handler on connector "first_test_servo": it burns when I pee'

async def test_cannot_cancel_from_on_handlers_warning(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.ON)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()

    messages = []
    connector.logger.add(lambda m: messages.append(m), level=0)
    await servo.dispatch_event("promote", return_exceptions=True)
    assert messages[0].record["level"].name == "WARNING"
    assert messages[0].record["message"] == "Cannot cancel an event from an on handler: event dispatched"

from servo.errors import EventCancelledError
async def test_cannot_cancel_from_on_handlers(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.ON)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()
    with pytest.raises(TypeError) as error:
        await servo.dispatch_event("promote")
    assert str(error.value) == "Cannot cancel an event from an on handler"


async def test_cannot_cancel_from_after_handlers_warning(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    event_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]

    mock = mocker.patch.object(event_handler, "handler")
    mock.side_effect = EventCancelledError()

    with pytest.raises(TypeError) as error:
        await servo.dispatch_event("promote")
    assert str(error.value) == "Cannot cancel an event from an after handler"

async def test_after_handlers_are_not_called_on_failure_raises(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    after_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    spy = mocker.spy(after_handler, "handler")

    # Mock the before handler to raise an EventError
    on_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    mock = mocker.patch.object(on_handler, "handler")
    mock.side_effect = EventError()
    with pytest.raises(EventError):
        await servo.dispatch_event("promote", return_exceptions=False)

    spy.assert_not_called()

async def test_after_handlers_are_called_on_failure(mocker, servo: servo):
    connector = servo.get_connector("first_test_servo")
    after_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    spy = mocker.spy(after_handler, "handler")

    # Mock the before handler to raise an EventError
    on_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    mock = mocker.patch.object(on_handler, "handler")
    mock.side_effect = EventError()
    results = await servo.dispatch_event("promote", return_exceptions=True)
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
    assert result.preposition == Preposition.ON


async def test_dispatching_specific_prepositions(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.BEFORE)[0]
    before_spy = mocker.spy(before_handler, "handler")
    on_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    after_spy = mocker.spy(after_handler, "handler")
    await servo.dispatch_event("promote", _prepositions=Preposition.ON)
    before_spy.assert_not_called()
    on_spy.assert_called_once()
    after_spy.assert_not_called()


async def test_dispatching_multiple_specific_prepositions(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    before_handler = connector.get_event_handlers("promote", Preposition.BEFORE)[0]
    before_spy = mocker.spy(before_handler, "handler")
    on_handler = connector.get_event_handlers("promote", Preposition.ON)[0]
    on_spy = mocker.spy(on_handler, "handler")
    after_handler = connector.get_event_handlers("promote", Preposition.AFTER)[0]
    after_spy = mocker.spy(after_handler, "handler")
    await servo.dispatch_event(
        "promote", _prepositions=Preposition.ON | Preposition.BEFORE
    )
    before_spy.assert_called_once()
    on_spy.assert_called_once()
    after_spy.assert_not_called()


async def test_startup_event(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    await servo.startup()
    assert connector.started_up == True


async def test_shutdown_event(mocker, servo: servo) -> None:
    connector = servo.get_connector("first_test_servo")
    on_handler = connector.get_event_handlers("shutdown", Preposition.ON)[0]
    on_spy = mocker.spy(on_handler, "handler")
    await servo.shutdown()
    on_spy.assert_called()


async def test_dispatching_event_that_doesnt_exist(mocker, servo: servo) -> None:
    with pytest.raises(KeyError) as error:
        await servo.dispatch_event("this_is_not_an_event", _prepositions=Preposition.ON)
    assert str(error.value) == "'this_is_not_an_event'"


##
# Test event handlers


async def test_event():
    ...


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
    assert (
        str(error.value)
        == """invalid event handler "adjust": missing required parameter "adjustments" in callable signature "(self) -> servo.types.Description", expected "(self, adjustments: 'List[servo.types.Adjustment]', control: 'servo.types.Control' = Control(duration=Duration('0' 0:00:00), delay=Duration('0' 0:00:00), warmup=Duration('0' 0:00:00), settlement=None, load=None, userdata=None)) -> 'servo.types.Description'\""""
    )


def test_registering_event_handler_with_missing_keyword_param_fails() -> None:
    with pytest.raises(TypeError) as error:

        @on_event("measure")
        def invalid_measure(self, *, control: Control = Control()) -> Measurement:
            pass

    assert error
    assert (
        str(error.value)
        == """invalid event handler "measure": missing required parameter "metrics" in callable signature "(self, *, control: servo.types.Control = Control(duration=Duration('0' 0:00:00), delay=Duration('0' 0:00:00), warmup=Duration('0' 0:00:00), settlement=None, load=None, userdata=None)) -> servo.types.Measurement", expected "(self, *, metrics: 'List[str]' = None, control: 'servo.types.Control' = Control(duration=Duration('0' 0:00:00), delay=Duration('0' 0:00:00), warmup=Duration('0' 0:00:00), settlement=None, load=None, userdata=None)) -> 'servo.types.Measurement'\""""
    )


def test_registering_event_handler_with_missing_keyword_param_succeeds_with_var_keywords() -> None:
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
        == "invalid event handler \"startup\": encountered unexpected parameter \"invalid\" in callable signature \"(self, invalid, /) -> None\", expected \"(self) -> 'None'\""
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
    def before_measure(self) -> None:
        pass

    assert before_measure.__event_handler__.event.name == "measure"
    assert before_measure.__event_handler__.preposition == Preposition.BEFORE


def test_registering_before_handler_fails_with_extra_args() -> None:
    with pytest.raises(TypeError) as error:

        @before_event("measure")
        def invalid_measure(self, invalid: str, another: int) -> None:
            pass

    assert error
    assert (
        str(error.value)
        == """invalid before event handler "before:measure": encountered unexpected parameters "another and invalid" in callable signature "(self, invalid: str, another: int) -> None", expected "(self) -> 'None'\""""
    )


def test_validation_of_before_handlers_ignores_kwargs() -> None:
    @before_event("measure")
    def before_measure(self, **kwargs) -> None:
        pass

    assert before_measure.__event_handler__.event.name == "measure"
    assert before_measure.__event_handler__.preposition == Preposition.BEFORE


def test_validation_of_after_handlers() -> None:
    @after_event("measure")
    def after_measure(self, results: List[EventResult]) -> None:
        pass

    assert after_measure.__event_handler__.event.name == "measure"
    assert after_measure.__event_handler__.preposition == Preposition.AFTER


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
        == "invalid after event handler \"after:measure\": encountered unexpected parameters \"another and invalid\" in callable signature \"(self, results: List[servo.events.EventResult], invalid: str, another: int) -> None\", expected \"(self, results: 'List[EventResult]') -> 'None'\""
    )


def test_validation_of_after_handlers_ignores_kwargs() -> None:
    @after_event("measure")
    def after_measure(self, results: List[EventResult], **kwargs) -> None:
        pass

    assert after_measure.__event_handler__.event.name == "measure"
    assert after_measure.__event_handler__.preposition == Preposition.AFTER


class TestAssembly:
    def test_assemble_assigns_optimizer_to_connectors(self, servo_yaml: Path):
        config = {
            "connectors": {"vegeta": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        optimizer = Optimizer(id="dev.opsani.com/servox", token="1234556789")

        assembly = Assembly.assemble(
            config_file=servo_yaml, optimizer=optimizer
        )
        servo = assembly.servos[0]
        connector = servo.connectors[0]
        assert connector.optimizer == optimizer

    def test_aliased_connectors_produce_schema(self, servo_yaml: Path, mocker) -> None:
        mocker.patch.object(Servo, "version", "100.0.0")
        mocker.patch.object(VegetaConnector, "version", "100.0.0")

        config = {
            "connectors": {"vegeta": "vegeta", "other": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
            "other": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        optimizer = Optimizer(id="dev.opsani.com/servox", token="1234556789")

        assembly = Assembly.assemble(
            config_file=servo_yaml, optimizer=optimizer
        )
        DynamicServoSettings = assembly.servos[0].config.__class__

        schema = json.loads(DynamicServoSettings.schema_json())

        # Description on parent class can be squirrely
        assert schema["properties"]["description"]["env_names"] == ["SERVO_DESCRIPTION"]
        assert schema == {
        'title': 'Servo Configuration Schema',
        'description': 'Schema for configuration of Servo v100.0.0 with Vegeta Connector v100.0.0',
        'type': 'object',
        'properties': {
            'description': {
                'title': 'Description',
                'env_names': [
                    'SERVO_DESCRIPTION',
                ],
                'type': 'string',
            },
            'name': {
                'title': 'Name',
                'env_names': [
                    'SERVO_NAME',
                ],
                'type': 'string',
            },
            'optimizer': {
                'title': 'Optimizer',
                'description': 'Configuration of the Servo connector',
                'env_names': [
                    'SERVO_OPTIMIZER',
                ],
                'allOf': [
                    {
                        '$ref': '#/definitions/Optimizer',
                    },
                ],
            },
            'connectors': {
                'title': 'Connectors',
                'description': (
                    'An optional, explicit configuration of the active connectors.\n'
                    '\n'
                    'Configurable as either an array of connector identifiers (names or class) or\n'
                    'a dictionary where the keys specify the key path to the connectors configuration\n'
                    'and the values identify the connector (by name or class name).'
                ),
                'examples': [
                    [
                        'kubernetes',
                        'prometheus',
                    ],
                    {
                        'staging_prom': 'prometheus',
                        'gateway_prom': 'prometheus',
                    },
                ],
                'env_names': [
                    'SERVO_CONNECTORS',
                ],
                'anyOf': [
                    {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                        },
                    },
                    {
                        'type': 'object',
                        'additionalProperties': {
                            'type': 'string',
                        },
                    },
                ],
            },
            'servo': {
                'title': 'Servo',
                'description': 'Configuration of the Servo connector',
                'env_names': [
                    'SERVO_SERVO',
                ],
                'allOf': [
                    {
                        '$ref': '#/definitions/servo__configuration__ServoConfiguration',
                    },
                ],
            },
            'other': {
                'title': 'Other',
                'env_names': [
                    'SERVO_OTHER',
                ],
                'allOf': [
                    {
                        '$ref': '#/definitions/VegetaConfiguration__other',
                    },
                ],
            },
            'vegeta': {
                'title': 'Vegeta',
                'env_names': [
                    'SERVO_VEGETA',
                ],
                'allOf': [
                    {
                        '$ref': '#/definitions/VegetaConfiguration',
                    },
                ],
            },
        },
        'required': [
            'other',
            'vegeta',
        ],
        'additionalProperties': False,
        'definitions': {
            'Optimizer': {
                'title': 'Optimizer',
                'description': (
                    'An Optimizer models an Opsani optimization engines that the Servo can connect to\n'
                    'in order to access the Opsani machine learning technology for optimizing system infrastructure\n'
                    'and application workloads.'
                ),
                'type': 'object',
                'properties': {
                    'org_domain': {
                        'title': 'Org Domain',
                        'env_names': [
                            'org_domain',
                        ],
                        'pattern': (
                            '(([\\da-zA-Z])([_\\w-]{,62})\\.){,127}(([\\da-zA-Z])[_\\w-]{,61})?([\\da-zA-Z]\\.((xn\\-\\-[a-zA-Z'
                            '\\d]+)|([a-zA-Z\\d]{2,})))'
                        ),
                        'type': 'string',
                    },
                    'app_name': {
                        'title': 'App Name',
                        'env_names': [
                            'app_name',
                        ],
                        'pattern': '^[a-z\\-\\.0-9]{3,64}$',
                        'type': 'string',
                    },
                    'token': {
                        'title': 'Token',
                        'env': 'OPSANI_TOKEN',
                        'env_names': [
                            'OPSANI_TOKEN',
                        ],
                        'type': 'string',
                    },
                    'base_url': {
                        'title': 'Base Url',
                        'default': 'https://api.opsani.com/',
                        'env': 'OPSANI_BASE_URL',
                        'env_names': [
                            'OPSANI_BASE_URL',
                        ],
                        'minLength': 1,
                        'maxLength': 65536,
                        'format': 'uri',
                        'type': 'string',
                    },
                    'url': {
                        'title': 'Url',
                        'env_names': ['url'],
                        'minLength': 1,
                        'maxLength': 65536,
                        'format': 'uri',
                        'type': 'string',
                    },
                },
                'required': [
                    'org_domain',
                    'app_name',
                    'token',
                ],
                'additionalProperties': False,
            },
            'BackoffSettings': {
                'title': 'BackoffSettings Connector Configuration Schema',
                'description': (
                    'BackoffSettings objects model configuration of backoff and retry policies.\n'
                    '\n'
                    'See https://github.com/litl/backoff'
                ),
                'type': 'object',
                'properties': {
                    'description': {
                        'title': 'Description',
                        'description': 'An optional annotation describing the configuration.',
                        'env_names': [
                            'BACKOFF_SETTINGS_DESCRIPTION',
                        ],
                        'type': 'string',
                    },
                    'max_time': {
                        'title': 'Max Time',
                        'env_names': [
                            'BACKOFF_SETTINGS_MAX_TIME',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                    'max_tries': {
                        'title': 'Max Tries',
                        'env_names': [
                            'BACKOFF_SETTINGS_MAX_TRIES',
                        ],
                        'type': 'integer',
                    },
                },
                'additionalProperties': False,
            },
            'BackoffConfigurations': {
                'title': 'BackoffConfigurations',
                'description': 'A mapping of named backoff configurations.',
                'type': 'object',
                'additionalProperties': {
                    '$ref': '#/definitions/BackoffSettings',
                },
            },
            'Timeouts': {
                'title': 'Timeouts Connector Configuration Schema',
                'description': (
                    'Timeouts models the configuration of timeouts for the HTTPX library, which provides HTTP networki'
                    'ng capabilities to the\n'
                    'servo.\n'
                    '\n'
                    'See https://www.python-httpx.org/advanced/#timeout-configuration'
                ),
                'type': 'object',
                'properties': {
                    'description': {
                        'title': 'Description',
                        'description': 'An optional annotation describing the configuration.',
                        'env_names': [
                            'TIMEOUTS_DESCRIPTION',
                        ],
                        'type': 'string',
                    },
                    'connect': {
                        'title': 'Connect',
                        'env_names': [
                            'TIMEOUTS_CONNECT',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                    'read': {
                        'title': 'Read',
                        'env_names': [
                            'TIMEOUTS_READ',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                    'write': {
                        'title': 'Write',
                        'env_names': [
                            'TIMEOUTS_WRITE',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                    'pool': {
                        'title': 'Pool',
                        'env_names': [
                            'TIMEOUTS_POOL',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                },
                'additionalProperties': False,
            },
            'servo__configuration__ServoConfiguration': {
                'title': 'Servo Connector Configuration Schema',
                'description': (
                    'ServoConfiguration models configuration for the Servo connector and establishes default\n'
                    'settings for shared services such as networking and logging.'
                ),
                'type': 'object',
                'properties': {
                    'description': {
                        'title': 'Description',
                        'description': 'An optional annotation describing the configuration.',
                        'env_names': [
                            'SERVO_DESCRIPTION',
                        ],
                        'type': 'string',
                    },
                    'backoff': {
                        'title': 'Backoff',
                        'env_names': [
                            'SERVO_BACKOFF',
                        ],
                        'allOf': [
                            {
                                '$ref': '#/definitions/BackoffConfigurations',
                            },
                        ],
                    },
                    'proxies': {
                        'title': 'Proxies',
                        'env_names': [
                            'SERVO_PROXIES',
                        ],
                        'anyOf': [
                            {
                                'type': 'string',
                                'pattern': '^(https?|all)://',
                            },
                            {
                                'type': 'object',
                                'patternProperties': {
                                    '^(https?|all)://': {
                                        'type': 'string',
                                        'minLength': 1,
                                        'maxLength': 65536,
                                        'format': 'uri',
                                    },
                                },
                            },
                        ],
                    },
                    'timeouts': {
                        'title': 'Timeouts',
                        'env_names': [
                            'SERVO_TIMEOUTS',
                        ],
                        'allOf': [
                            {
                                '$ref': '#/definitions/Timeouts',
                            },
                        ],
                    },
                    'ssl_verify': {
                        'title': 'Ssl Verify',
                        'env_names': [
                            'SERVO_SSL_VERIFY',
                        ],
                        'anyOf': [
                            {
                                'type': 'boolean',
                            },
                            {
                                'type': 'string',
                                'format': 'file-path',
                            },
                        ],
                    },
                },
                'additionalProperties': False,
            },
            'TargetFormat': {
                'title': 'TargetFormat',
                'description': 'An enumeration.',
                'enum': [
                    'http',
                    'json',
                ],
                'type': 'string',
            },
            'VegetaConfiguration__other': {
                'title': 'Vegeta Connector Settings (named other)',
                'description': 'Configuration of the Vegeta connector',
                'type': 'object',
                'properties': {
                    'description': {
                        'title': 'Description',
                        'description': 'An optional annotation describing the configuration.',
                        'env_names': [
                            'SERVO_OTHER_DESCRIPTION',
                        ],
                        'type': 'string',
                    },
                    'rate': {
                        'title': 'Rate',
                        'description': (
                            'Specifies the request rate per time unit to issue against the targets. Given in the forma'
                            't of request/time unit.'
                        ),
                        'env_names': [
                            'SERVO_OTHER_RATE',
                        ],
                        'type': 'string',
                    },
                    'format': {
                        'description': (
                            'Specifies the format of the targets input. Valid values are http and json. Refer to the V'
                            'egeta docs for details.'
                        ),
                        'default': 'http',
                        'allOf': [
                            {
                                '$ref': '#/definitions/TargetFormat',
                            },
                        ],
                    },
                    'target': {
                        'title': 'Target',
                        'description': (
                            'Specifies a single formatted Vegeta target to load. See the format option to learn about '
                            'available target formats. This option is exclusive of the targets option and will provide'
                            ' a target to Vegeta via stdin.'
                        ),
                        'env_names': [
                            'SERVO_OTHER_TARGET',
                        ],
                        'type': 'string',
                    },
                    'targets': {
                        'title': 'Targets',
                        'description': (
                            'Specifies the file from which to read targets. See the format option to learn about avail'
                            'able target formats. This option is exclusive of the target option and will provide targe'
                            'ts to via through a file on disk.'
                        ),
                        'env_names': [
                            'SERVO_OTHER_TARGETS',
                        ],
                        'format': 'file-path',
                        'type': 'string',
                    },
                    'connections': {
                        'title': 'Connections',
                        'description': 'Specifies the maximum number of idle open connections per target host.',
                        'default': 10000,
                        'env_names': [
                            'SERVO_OTHER_CONNECTIONS',
                        ],
                        'type': 'integer',
                    },
                    'workers': {
                        'title': 'Workers',
                        'description': (
                            'Specifies the initial number of workers used in the attack. The workers will automaticall'
                            'y increase to achieve the target request rate, up to max-workers.'
                        ),
                        'default': 10,
                        'env_names': [
                            'SERVO_OTHER_WORKERS',
                        ],
                        'type': 'integer',
                    },
                    'max_workers': {
                        'title': 'Max Workers',
                        'description': (
                            'The maximum number of workers used to sustain the attack. This can be used to control the'
                            ' concurrency of the attack to simulate a target number of clients.'
                        ),
                        'env_names': [
                            'SERVO_OTHER_MAX_WORKERS',
                        ],
                        'type': 'integer',
                    },
                    'max_body': {
                        'title': 'Max Body',
                        'description': (
                            'Specifies the maximum number of bytes to capture from the body of each response. Remainin'
                            'g unread bytes will be fully read but discarded.'
                        ),
                        'default': -1,
                        'env_names': [
                            'SERVO_OTHER_MAX_BODY',
                        ],
                        'type': 'integer',
                    },
                    'http2': {
                        'title': 'Http2',
                        'description': 'Specifies whether to enable HTTP/2 requests to servers which support it.',
                        'default': True,
                        'env_names': [
                            'SERVO_OTHER_HTTP2',
                        ],
                        'type': 'boolean',
                    },
                    'keepalive': {
                        'title': 'Keepalive',
                        'description': 'Specifies whether to reuse TCP connections between HTTP requests.',
                        'default': True,
                        'env_names': [
                            'SERVO_OTHER_KEEPALIVE',
                        ],
                        'type': 'boolean',
                    },
                    'insecure': {
                        'title': 'Insecure',
                        'description': 'Specifies whether to ignore invalid server TLS certificates.',
                        'default': False,
                        'env_names': [
                            'SERVO_OTHER_INSECURE',
                        ],
                        'type': 'boolean',
                    },
                    'reporting_interval': {
                        'title': 'Reporting Interval',
                        'description': 'How often to report metrics during a measurement cycle.',
                        'default': '15s',
                        'env_names': [
                            'SERVO_OTHER_REPORTING_INTERVAL',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                },
                'required': ['rate'],
                'additionalProperties': False,
            },
            'VegetaConfiguration': {
                'title': 'Vegeta Connector Settings (named vegeta)',
                'description': 'Configuration of the Vegeta connector',
                'type': 'object',
                'properties': {
                    'description': {
                        'title': 'Description',
                        'description': 'An optional annotation describing the configuration.',
                        'env_names': [
                            'SERVO_VEGETA_DESCRIPTION',
                        ],
                        'type': 'string',
                    },
                    'rate': {
                        'title': 'Rate',
                        'description': (
                            'Specifies the request rate per time unit to issue against the targets. Given in the forma'
                            't of request/time unit.'
                        ),
                        'env_names': [
                            'SERVO_VEGETA_RATE',
                        ],
                        'type': 'string',
                    },
                    'format': {
                        'description': (
                            'Specifies the format of the targets input. Valid values are http and json. Refer to the V'
                            'egeta docs for details.'
                        ),
                        'default': 'http',
                        'allOf': [
                            {
                                '$ref': '#/definitions/TargetFormat',
                            },
                        ],
                    },
                    'target': {
                        'title': 'Target',
                        'description': (
                            'Specifies a single formatted Vegeta target to load. See the format option to learn about '
                            'available target formats. This option is exclusive of the targets option and will provide'
                            ' a target to Vegeta via stdin.'
                        ),
                        'env_names': [
                            'SERVO_VEGETA_TARGET',
                        ],
                        'type': 'string',
                    },
                    'targets': {
                        'title': 'Targets',
                        'description': (
                            'Specifies the file from which to read targets. See the format option to learn about avail'
                            'able target formats. This option is exclusive of the target option and will provide targe'
                            'ts to via through a file on disk.'
                        ),
                        'env_names': [
                            'SERVO_VEGETA_TARGETS',
                        ],
                        'format': 'file-path',
                        'type': 'string',
                    },
                    'connections': {
                        'title': 'Connections',
                        'description': 'Specifies the maximum number of idle open connections per target host.',
                        'default': 10000,
                        'env_names': [
                            'SERVO_VEGETA_CONNECTIONS',
                        ],
                        'type': 'integer',
                    },
                    'workers': {
                        'title': 'Workers',
                        'description': (
                            'Specifies the initial number of workers used in the attack. The workers will automaticall'
                            'y increase to achieve the target request rate, up to max-workers.'
                        ),
                        'default': 10,
                        'env_names': [
                            'SERVO_VEGETA_WORKERS',
                        ],
                        'type': 'integer',
                    },
                    'max_workers': {
                        'title': 'Max Workers',
                        'description': (
                            'The maximum number of workers used to sustain the attack. This can be used to control the'
                            ' concurrency of the attack to simulate a target number of clients.'
                        ),
                        'env_names': [
                            'SERVO_VEGETA_MAX_WORKERS',
                        ],
                        'type': 'integer',
                    },
                    'max_body': {
                        'title': 'Max Body',
                        'description': (
                            'Specifies the maximum number of bytes to capture from the body of each response. Remainin'
                            'g unread bytes will be fully read but discarded.'
                        ),
                        'default': -1,
                        'env_names': [
                            'SERVO_VEGETA_MAX_BODY',
                        ],
                        'type': 'integer',
                    },
                    'http2': {
                        'title': 'Http2',
                        'description': 'Specifies whether to enable HTTP/2 requests to servers which support it.',
                        'default': True,
                        'env_names': [
                            'SERVO_VEGETA_HTTP2',
                        ],
                        'type': 'boolean',
                    },
                    'keepalive': {
                        'title': 'Keepalive',
                        'description': 'Specifies whether to reuse TCP connections between HTTP requests.',
                        'default': True,
                        'env_names': [
                            'SERVO_VEGETA_KEEPALIVE',
                        ],
                        'type': 'boolean',
                    },
                    'insecure': {
                        'title': 'Insecure',
                        'description': 'Specifies whether to ignore invalid server TLS certificates.',
                        'default': False,
                        'env_names': [
                            'SERVO_VEGETA_INSECURE',
                        ],
                        'type': 'boolean',
                    },
                    'reporting_interval': {
                        'title': 'Reporting Interval',
                        'description': 'How often to report metrics during a measurement cycle.',
                        'default': '15s',
                        'env_names': [
                            'SERVO_VEGETA_REPORTING_INTERVAL',
                        ],
                        'type': 'string',
                        'format': 'duration',
                        'pattern': (
                            '([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)'
                            '?([\\d\\.]+us)?([\\d\\.]+ns)?'
                        ),
                        'examples': [
                            '300ms',
                            '5m',
                            '2h45m',
                            '72h3m0.5s',
                        ],
                    },
                },
                'required': ['rate'],
                'additionalProperties': False,
            },
        },
    }

    def test_aliased_connectors_get_distinct_env_configuration(
        self, servo_yaml: Path
    ) -> None:
        config = {
            "connectors": {"vegeta": "vegeta", "other": "vegeta"},
            "vegeta": {"rate": 0, "target": "https://opsani.com/"},
            "other": {"rate": 0, "target": "https://opsani.com/"},
        }
        servo_yaml.write_text(yaml.dump(config))

        optimizer = Optimizer(id="dev.opsani.com/servox", token="1234556789")

        assembly = Assembly.assemble(
            config_file=servo_yaml, optimizer=optimizer
        )
        DynamicServoConfiguration = assembly.servos[0].config.__class__

        # Grab the vegeta field and check it
        vegeta_field = DynamicServoConfiguration.__fields__["vegeta"]
        vegeta_settings_type = vegeta_field.type_
        assert vegeta_settings_type.__name__ == "VegetaConfiguration"
        assert vegeta_field.field_info.extra["env_names"] == {"SERVO_VEGETA"}

        # Grab the other field and check it
        other_field = DynamicServoConfiguration.__fields__["other"]
        other_settings_type = other_field.type_
        assert other_settings_type.__name__ == "VegetaConfiguration__other"
        assert other_field.field_info.extra["env_names"] == {"SERVO_OTHER"}

        with environment_overrides({"SERVO_DESCRIPTION": "this description"}):
            assert os.environ["SERVO_DESCRIPTION"] == "this description"
            s = DynamicServoConfiguration(
                other=other_settings_type.construct(),
                vegeta=vegeta_settings_type(
                    rate=10, target="http://example.com/"
                ),
            )
            assert s.description == "this description"

        # Make sure the incorrect case does pass
        with environment_overrides({"SERVO_RATE": "invalid"}):
            with pytest.raises(ValidationError):
                vegeta_settings_type(target="https://foo.com/")

        # Try setting values via env
        with environment_overrides(
            {
                "SERVO_OTHER_RATE": "100/1s",
                "SERVO_OTHER_TARGET": "https://opsani.com/servox",
            }
        ):
            s = other_settings_type()
            assert s.rate == "100/1s"
            assert s.target == "https://opsani.com/servox"


def test_generating_schema_with_test_connectors(
    optimizer_env: None, servo_yaml: Path
) -> None:
    optimizer = Optimizer(id="dev.opsani.com/servox", token="1234556789")

    assembly = Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    assert len(assembly.servos) == 1, "servo was not assembled"
    DynamicServoConfiguration = assembly.servos[0].config.__class__
    DynamicServoConfiguration.schema()
    # NOTE: Covers naming conflicts between settings models -- will raise if misconfigured


class TestServoSettings:
    def test_forbids_extra_attributes(self) -> None:
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(forbidden=[])
            assert "extra fields not permitted" in str(e)

    def test_override_optimizer_settings_with_env_vars(self) -> None:
        with environment_overrides({"OPSANI_TOKEN": "abcdefg"}):
            assert os.environ["OPSANI_TOKEN"] is not None
            optimizer = Optimizer(app_name="foo", org_domain="dsada.com")
            assert optimizer.token == "abcdefg"

    def test_set_connectors_with_env_vars(self) -> None:
        with environment_overrides({"SERVO_CONNECTORS": '["measure"]'}):
            assert os.environ["SERVO_CONNECTORS"] is not None
            s = BaseServoConfiguration()
            assert s is not None
            schema = s.schema()
            assert schema["properties"]["connectors"]["env_names"] == {
                "SERVO_CONNECTORS"
            }
            assert s.connectors is not None
            assert s.connectors == ["measure"]

    def test_connectors_allows_none(self):
        s = BaseServoConfiguration(
            connectors=None,
        )
        assert s.connectors is None

    def test_connectors_allows_set_of_classes(self):
        class FooConnector(BaseConnector):
            pass

        class BarConnector(BaseConnector):
            pass

        s = BaseServoConfiguration(
            connectors={FooConnector, BarConnector},
        )
        assert set(s.connectors) == {"FooConnector", "BarConnector"}

    def test_connectors_rejects_invalid_connector_set_elements(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={BaseServoConfiguration},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == "Invalid connectors value: <class 'servo.configuration.BaseServoConfiguration'>"
        )

    def test_connectors_allows_set_of_class_names(self):
        s = BaseServoConfiguration(
            connectors={"MeasureConnector", "AdjustConnector"},
        )
        assert set(s.connectors) == {"MeasureConnector", "AdjustConnector"}

    def test_connectors_rejects_invalid_connector_set_class_name_elements(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={"servo.servo.BaseServoConfiguration"},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == "BaseServoConfiguration is not a Connector subclass"
        )

    def test_connectors_allows_set_of_keys(self):
        s = BaseServoConfiguration(
            connectors={"vegeta"},
        )
        assert s.connectors == ["vegeta"]

    def test_connectors_allows_dict_of_keys_to_classes(self):
        s = BaseServoConfiguration(
            connectors={"alias": VegetaConnector},
        )
        assert s.connectors == {"alias": "VegetaConnector"}

    def test_connectors_allows_dict_of_keys_to_class_names(self):
        s = BaseServoConfiguration(
            connectors={"alias": "VegetaConnector"},
        )
        assert s.connectors == {"alias": "VegetaConnector"}

    def test_connectors_allows_dict_with_explicit_map_to_default_name(self):
        s = BaseServoConfiguration(
            connectors={"vegeta": "VegetaConnector"},
        )
        assert s.connectors == {"vegeta": "VegetaConnector"}

    def test_connectors_allows_dict_with_explicit_map_to_default_class(self):
        s = BaseServoConfiguration(
            connectors={"vegeta": VegetaConnector},
        )
        assert s.connectors == {"vegeta": "VegetaConnector"}

    def test_connectors_forbids_dict_with_existing_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={"vegeta": "MeasureConnector"},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == 'Name "vegeta" is reserved by `VegetaConnector`'
        )

    @pytest.fixture(autouse=True, scope="session")
    def discover_connectors(self) -> None:
        from servo.connector import ConnectorLoader

        loader = ConnectorLoader()
        for connector in loader.load():
            pass

    def test_connectors_forbids_dict_with_reserved_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={"connectors": "VegetaConnector"},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert e.value.errors()[0]["msg"] == 'Name "connectors" is reserved'

    def test_connectors_forbids_dict_with_invalid_key(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={"This Is Not Valid": "VegetaConnector"},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == '"This Is Not Valid" is not a valid connector name: names may only contain alphanumeric characters, hyphens, slashes, periods, and underscores'
        )

    def test_connectors_rejects_invalid_connector_dict_values(self):
        with pytest.raises(ValidationError) as e:
            BaseServoConfiguration(
                connectors={"whatever": "Not a Real Connector"},
            )
        assert "1 validation error for BaseServoConfiguration" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("connectors",)
        assert (
            e.value.errors()[0]["msg"]
            == "Invalid connectors value: Not a Real Connector"
        )


# Test servo config...


def test_backoff_settings() -> None:
    BaseServoConfiguration()


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
    with pytest.raises(ValidationError):
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
    config = ServoConfiguration(timeouts=value)
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
        [],
        {},
    ],
)
def test_valid_proxies(proxies) -> None:
    ServoConfiguration(proxies=proxies)


@pytest.mark.parametrize("proxies", [0.5, "not valid", 1234])
def test_invalid_proxies(proxies) -> None:
    with pytest.raises(ValidationError):
        ServoConfiguration(proxies=proxies)


def test_api_client_options() -> None:
    config = ServoConfiguration(proxies="http://localhost:1234", ssl_verify=False)

    optimizer = Optimizer("test.com/foo", token="12345")
    servo = Servo(config={"servo": config}, optimizer=optimizer, connectors=[])

    assert {
        "proxies": "http://localhost:1234",
        "timeout": None,
        "verify": False,
    }.items() <= servo.api_client_options.items()


async def test_httpx_client_config() -> None:
    config = ServoConfiguration(proxies="http://localhost:1234", ssl_verify=False)

    from httpx._utils import URLPattern

    optimizer = Optimizer("test.com/foo", token="12345")
    connector = MeasureConnector(config=BaseConfiguration(), optimizer=optimizer)
    servo = Servo(config={"servo": config}, optimizer=optimizer, connectors=[connector])

    for c in [servo, connector]:
        async with c.api_client() as client:
            for k, v in client._proxies.items():
                assert k == URLPattern("all://")
            assert client._transport._ssl_context.verify_mode == ssl.CERT_NONE
            assert client._transport._ssl_context.check_hostname == False


def test_backoff_defaults() -> None:
    config = ServoConfiguration()
    assert config.backoff
    assert config.backoff["__default__"]
    assert config.backoff["__default__"].max_time is not None
    assert config.backoff["__default__"].max_time == Duration("10m")
    assert config.backoff["__default__"].max_tries is None


def test_backoff_contexts() -> None:
    contexts = servox.configuration.BackoffConfigurations(__root__={
            "__default__": {"max_time": "10m", "max_tries": None},
            "connect": {"max_time": "1h", "max_tries": None},
        })
    debug(contexts)

    config = servox.configuration.ServoConfiguration(backoff=contexts)
    debug(config)

def test_backoff_context() -> None:
    config = ServoConfiguration()
    assert config.backoff
    debug(config)
    # assert config.backoff.max_time()
    # assert config.backoff.max_time("whatever")


    assert config.backoff["__default__"].max_time is not None
    assert config.backoff["__default__"].max_time == Duration("10m")
    assert config.backoff["__default__"].max_tries is None


@pytest.mark.integration
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
    config = ServoConfiguration(proxies=proxies)
    optimizer = Optimizer("test.com/foo", token="12345")
    servo = Servo(config={"servo": config}, optimizer=optimizer, connectors=[])
    async with servo.api_client() as client:
        transport = client._transport_for_url(httpx.URL(optimizer.base_url))
        assert isinstance(transport, httpcore.AsyncHTTPProxy)
        assert transport.proxy_origin == (b'http', b'localhost', 1234)


def test_codename() -> None:
    assert __cryptonym__


async def test_add_connector(servo: Servo) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    assert connector not in servo.connectors
    await servo.add_connector("whatever", connector)
    assert connector in servo.connectors
    assert servo.config.whatever == connector.config


async def test_add_connector_sends_startup_event(servo: Servo) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    assert connector.started_up is False
    await servo.add_connector("whatever", connector)
    assert connector.started_up is True


async def test_add_connector_can_handle_events(servo: Servo) -> None:
    results = await servo.dispatch_event("this_is_an_event")
    assert len(results) == 2

    connector = FirstTestServoConnector(config=BaseConfiguration())
    await servo.add_connector("whatever", connector)

    results = await servo.dispatch_event("this_is_an_event")
    assert len(results) == 3


async def test_add_connector_raises_if_name_exists(servo: Servo) -> None:
    connector_1 = FirstTestServoConnector(config=BaseConfiguration())
    await servo.add_connector("whatever", connector_1)

    connector_2 = FirstTestServoConnector(config=BaseConfiguration())
    with pytest.raises(ValueError) as error:
        await servo.add_connector("whatever", connector_2)

    assert (
        str(error.value)
        == "invalid name: a connector named 'whatever' already exists in the servo"
    )


async def test_remove_connector(servo: Servo) -> None:
    connector = servo.get_connector("first_test_servo")
    assert connector in servo.connectors
    assert servo.config.first_test_servo == connector.config
    await servo.remove_connector(connector)
    assert connector not in servo.connectors

    with pytest.raises(AttributeError):
        assert servo.config.first_test_servo


async def test_remove_connector_by_name(servo: Servo) -> None:
    connector = servo.get_connector("first_test_servo")
    assert connector in servo.connectors
    assert servo.config.first_test_servo == connector.config
    await servo.remove_connector("first_test_servo")
    assert connector not in servo.connectors

    with pytest.raises(AttributeError):
        assert servo.config.first_test_servo


async def test_remove_connector_sends_shutdown_event(servo: Servo, mocker) -> None:
    connector = servo.get_connector("first_test_servo")
    on_handler = connector.get_event_handlers("shutdown", Preposition.ON)[0]
    on_spy = mocker.spy(on_handler, "handler")
    assert connector.started_up is False
    await servo.remove_connector(connector)
    on_spy.assert_called()


async def test_remove_connector_raises_if_name_does_not_exists(servo: Servo) -> None:
    with pytest.raises(ValueError) as error:
        await servo.remove_connector("whatever")

    assert (
        str(error.value)
        == "invalid connector: a connector named 'whatever' does not exist in the servo"
    )


async def test_remove_connector_raises_if_obj_does_not_exists(servo: Servo) -> None:
    connector = FirstTestServoConnector(config=BaseConfiguration())
    with pytest.raises(ValueError) as error:
        await servo.remove_connector(connector)

    assert (
        str(error.value)
        == "invalid connector: a connector named 'first_test_servo' does not exist in the servo"
    )

async def test_backoff() -> None:
    config = ServoConfiguration(proxies="http://localhost:1234", ssl_verify=False)
    debug(config)
    assert config.backoff
    assert config.backoff.max_time()
    # servo.Servo.current().config.backoff.max_time(),
def test_servo_name_literal(servo: Servo) -> None:
    servo.name = "hrm"
    assert servo.name == "hrm"

def test_servo_name_from_config() -> None:
    config = BaseServoConfiguration(name="archibald")
    servo = Servo(config=config, connectors=[])
    assert servo.name == "archibald"

def test_servo_name_falls_back_to_optimizer_id(servo: Servo) -> None:
    assert servo.name == "dev.opsani.com/servox"
