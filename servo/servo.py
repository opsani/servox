import abc
import contextlib
import inspect
import json
import os
import re
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Type, Union, Sequence

import httpx
import yaml
from pydantic import BaseModel, Extra, Field, create_model, validator
from pydantic.json import pydantic_encoder
from pydantic.schema import schema as pydantic_schema

import servo
from servo import api, connector
from servo.connector import (
    BaseConfiguration,
    Connector,
    ConnectorLoader,
    License,
    Maturity,
    Optimizer,
    _connector_subclasses,
    _connector_class_from_string,
    _validate_class
)
from servo.events import Preposition, event, on_event
from servo.types import Check, Control, Description, Measurement, Metric
from servo.utilities import join_to_series


_servo_context_var = ContextVar("servo.Servo.current", default=None)


class Events(str, Enum):
    """
    Events is an enumeration of the names of standard events defined by the servo.
    """

    # Lifecycle events
    STARTUP = "startup"
    SHUTDOWN = "shutdown"

    # Informational events
    METRICS = "metrics"
    COMPONENTS = "components"

    # Operational events
    CHECK = "check"
    DESCRIBE = "describe"
    MEASURE = "measure"
    ADJUST = "adjust"
    PROMOTE = "promote"


class _EventDefinitions:
    """
    Defines the default events. This class is declarative and is never directly referenced.

    The event signature is inferred from the decorated function.
    """

    # Lifecycle events
    @event(Events.STARTUP)
    def startup(self) -> None:
        pass

    @event(Events.SHUTDOWN)
    def shutdown(self) -> None:
        pass

    # Informational events
    @event(Events.METRICS)
    def metrics(self) -> List[Metric]:
        pass

    @event(Events.COMPONENTS)
    def components(self) -> Description:
        pass

    # Operational events
    @event(Events.MEASURE)
    def measure(
        self, *, metrics: List[str] = None, control: Control = Control()
    ) -> Measurement:
        pass

    @event(Events.CHECK)
    def check(self) -> List[Check]:
        pass

    @event(Events.DESCRIBE)
    def describe(self) -> Description:
        pass

    @event(Events.ADJUST)
    def adjust(self, data: dict) -> dict:
        pass

    @event(Events.PROMOTE)
    def promote(self) -> None:
        pass


class BaseServoConfiguration(BaseConfiguration, abc.ABC):
    """
    Abstract base class for Servo settings

    Note that the concrete BaseServoConfiguration class is built dynamically at runtime
    based on the avilable connectors and configuration in effect.

    See `Assembly` for details on how the concrete model is built.
    """

    connectors: Optional[Union[List[str], Dict[str, str]]] = Field(
        None,
        description=(
            "An optional, explicit configuration of the active connectors.\n"
            "\nConfigurable as either an array of connector identifiers (names or class) or\n"
            "a dictionary where the keys specify the key path to the connectors configuration\n"
            "and the values identify the connector (by name or class name)."
        ),
        examples=[
            ["kubernetes", "prometheus"],
            {"staging_prom": "prometheus", "gateway_prom": "prometheus"},
        ],
    )
    """
    An optional list of connector keys or a dict mapping of connector 
    key-paths to connector class names
    """

    @classmethod
    def generate(
        cls: Type["BaseServoConfiguration"], **kwargs
    ) -> "BaseServoConfiguration":
        """
        Generate configuration for the servo settings
        """
        for name, field in cls.__fields__.items():
            if (
                name not in kwargs
                and inspect.isclass(field.type_)
                and issubclass(field.type_, BaseConfiguration)
            ):
                kwargs[name] = field.type_.generate()
        return cls(**kwargs)

    @validator("connectors", pre=True)
    @classmethod
    def validate_connectors(
        cls, connectors
    ) -> Optional[Union[Dict[str, str], List[str]]]:
        if isinstance(connectors, str):
            # NOTE: Special case. When we are invoked with a string it is typically an env var
            try:
                decoded_value = BaseServoConfiguration.__config__.json_loads(connectors)  # type: ignore
            except ValueError as e:
                raise ValueError(f'error parsing JSON for "{connectors}"') from e

            # Prevent infinite recursion
            if isinstance(decoded_value, str):
                raise ValueError(
                    f'JSON string values for `connectors` cannot parse into strings: "{connectors}"'
                )

            connectors = decoded_value
        
        connectors = _normalize_connectors(connectors)
        # NOTE: Will raise if descriptor is invalid, failing validation
        _routes_for_connectors_descriptor(connectors)
        
        return connectors

    class Config:
        extra = Extra.forbid
        title = "Abstract Servo Configuration Schema"
        env_prefix = "SERVO_"


_servo_context_var = ContextVar('servo.servo', default=None)


@connector.metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
    version=servo.__version__,
)
class Servo(Connector):
    """
    A connector that interacts with the Opsani API to perform optimization.

    The `Servo` is a core object of the `servo` package. It manages a set of
    connectors that provide integration and interactivity to external services
    such as metrics collectors, orchestation systems, load generators, etc. The
    Servo acts primarily as an event gateway between the Opsani API and its child
    connectors.

    Servo objects are configured with a dynamically created class that is built by
    the `servo.Assembly` class. Servo objects are typically not created directly
    and are instead built through the `Assembly.assemble` method.
    """

    config: BaseServoConfiguration
    """Configuration for the Servo.

    Note that the Servo configuration is built dynamically at Servo assembly time.
    The concrete type is built in `Assembly.assemble()` and adds a field for each active 
    connector.
    """

    connectors: List[Connector]
    """
    The active connectors in the servo.
    """

    @staticmethod
    def current() -> 'Servo':
        """
        Returns the active servo for the current execution context.

        The value is managed by a contextvar and is concurrency safe.
        """
        return _servo_context_var.get()

    def __init__(self, *args, connectors: List[Connector], **kwargs) -> None:
        super().__init__(*args, connectors=[], **kwargs)

        # Ensure the connectors refer to the same objects by identity (required for eventing)
        self.connectors.extend(connectors)  

    def startup(self):
        """
        Notifies all connectors that the servo is starting up.
        """
        return self.broadcast_event(Events.STARTUP, prepositions=Preposition.ON)

    def shutdown(self):
        """
        Notifies all connectors that the servo is shutting down.
        """
        return self.broadcast_event(Events.SHUTDOWN, prepositions=Preposition.ON)

    def get_connector(self, name: Union[str, Sequence[str]]) -> Optional[Union[Connector, List[Connector]]]:
        """
        Returns one or more connectors by name.

        This is a convenience method equivalent to iterating `connectors` and comparing by name.

        When given a single name, returns the connector or `None` if not found.
        When given a sequence of names, returns a list of Connectors for all connectors found.
        """
        if isinstance(name, str):
            for connector in self.connectors:
                if connector.name == name:
                    return connector
            return None
        else:
            connectors = []
            for connector in self.connectors:
                if connector.name == name:
                    connectors.append(connector)
            return connectors

    ##
    # Event handlers

    @on_event()
    async def check(self) -> List[Check]:
        async with self.api_client() as client:
            event_request = api.Request(event=api.Event.HELLO)
            response = await client.post("servo", data=event_request.json())
            success = (response.status_code == httpx.codes.OK)
            return [Check(
                name="Opsani API connectivity",
                success=success,
                comment=f"Response status code: {response.status_code}",
            )]


def _normalize_connectors(connectors: Optional[Iterable]) -> Optional[Iterable]:
    if connectors is None:
        return connectors
    elif isinstance(connectors, str):
        if _connector_class_from_string(connectors) is None:
            raise ValueError(f"Invalid connectors value: {connectors}")
        return connectors
    elif isinstance(connectors, type) and issubclass(connectors, Connector):
        return connectors.__name__
    elif isinstance(connectors, (list, tuple, set)):
        connectors_list: List[str] = []
        for connector in connectors:
            connectors_list.append(_normalize_connectors(connector))
        return connectors_list
    elif isinstance(connectors, dict):
        normalized_dict: Dict[str, str] = {}
        for key, value in connectors.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Connector descriptor keys must be strings (invalid value '{key}'"
                )
            normalized_dict[key] = _normalize_connectors(value)

        return normalized_dict
    else:
        raise ValueError(f"Invalid connectors value: {connectors}")


RESERVED_KEYS = ["connectors", "control", "measure", "adjust", "optimization"]


def _reserved_keys() -> List[str]:
    reserved_keys = list(_default_routes().keys())
    reserved_keys.extend(RESERVED_KEYS)
    return reserved_keys


def _default_routes() -> Dict[str, Type[Connector]]:
    routes = {}
    for connector in _connector_subclasses:
        if connector is not Servo:
            routes[connector.__default_name__] = connector
    return routes


def _routes_for_connectors_descriptor(connectors) -> Dict[str, Connector]:
    if connectors is None:
        # None indicates that all available connectors should be activated
        return None

    elif isinstance(connectors, str):
        # NOTE: Special case. When we are invoked with a string it is typically an env var
        try:
            decoded_value = BaseServoConfiguration.__config__.json_loads(connectors)  # type: ignore
        except ValueError as e:
            raise ValueError(f'error parsing JSON for "{connectors}"') from e

        # Prevent infinite recursion
        if isinstance(decoded_value, str):
            raise ValueError(
                f'JSON string values for `connectors` cannot parse into strings: "{connectors}"'
            )

        return _routes_for_connectors_descriptor(decoded_value)

    elif isinstance(connectors, (list, tuple, set)):
        connector_routes: Dict[str, str] = {}
        for connector in connectors:
            if _validate_class(connector):
                connector_routes[connector.__default_name__] = connector
            elif connector_class := _connector_class_from_string(connector):
                connector_routes[connector_class.__default_name__] = connector_class
            else:
                raise ValueError(f"Missing validation for value {connector}")

        return connector_routes

    elif isinstance(connectors, dict):
        connector_routes = {}
        for name, value in connectors.items():
            if not isinstance(name, str):
                raise TypeError(f'Connector names must be strings: "{key}"')

            # Validate the name
            try:
                Connector.validate_name(name)
            except AssertionError as e:
                raise ValueError(f'"{name}" is not a valid connector name: {e}') from e

            # Resolve the connector class
            if isinstance(value, type):
                connector_class = value
            elif isinstance(value, str):
                connector_class = _connector_class_from_string(value)

            # Check for key reservations
            if name in _reserved_keys():
                if c := _default_routes().get(name, None):
                    if connector_class != c:
                        raise ValueError(
                            f'Name "{name}" is reserved by `{c.__name__}`'
                        )
                else:
                    raise ValueError(f'Name "{name}" is reserved')

            connector_routes[name] = connector_class

        return connector_routes

    else:
        raise ValueError(
            f"Unexpected type `{type(connectors).__qualname__}`` encountered (connectors: {connectors})"
        )
