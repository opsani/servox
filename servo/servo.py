from __future__ import annotations
import asyncio
from contextvars import ContextVar
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, Sequence

import httpx

import servo
from servo import api, connector
from servo.connector import (
    BaseConnector,
    License,
    Maturity,
)
from servo.checks import BaseChecks, Check, Filter, Severity
from servo.configuration import BaseAssemblyConfiguration
from servo.events import Preposition, event, on_event
from servo.types import Adjustment, Component, Control, Description, Duration, Measurement, Metric
from servo.utilities.pydantic import extra


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


class _EventDefinitions(Protocol):
    """
    Defines the default events. This class is declarative and is never directly referenced.

    The event signature is inferred from the decorated function.
    """

    # Lifecycle events
    @event(Events.STARTUP)
    async def startup(self) -> None:
        ...

    @event(Events.SHUTDOWN)
    async def shutdown(self) -> None:
        ...

    # Informational events
    @event(Events.METRICS)
    async def metrics(self) -> List[Metric]:
        ...

    @event(Events.COMPONENTS)
    async def components(self) -> List[Component]:
        ...

    # Operational events
    @event(Events.MEASURE)
    async def measure(
        self, *, metrics: List[str] = None, control: Control = Control()
    ) -> Measurement:
        if control.delay:
            await asyncio.sleep(control.delay.total_seconds())
        yield

    @event(Events.CHECK)
    async def check(self, filter_: Optional[Filter], halt_on: Optional[Severity] = Severity.critical) -> List[Check]:
        ...

    @event(Events.DESCRIBE)
    async def describe(self) -> Description:
        ...

    @event(Events.ADJUST)
    async def adjust(self, adjustments: List[Adjustment], control: Control = Control()) -> Description:
        ...

    @event(Events.PROMOTE)
    async def promote(self) -> None:
        ...


_servo_context_var = ContextVar('servo.servo', default=None)


class ServoChecks(BaseChecks):
    async def check_connectivity(self) -> Tuple[bool, str]:
        async with self.api_client() as client:
            event_request = api.Request(event=api.Event.HELLO)
            response = await client.post("servo", data=event_request.json())
            success = (response.status_code == httpx.codes.OK)
            return (success, f"Response status code: {response.status_code}")


@connector.metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
    version=servo.__version__,
)
class Servo(BaseConnector):
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

    config: BaseAssemblyConfiguration
    """Configuration of the Servo assembly.

    Note that the configuration is built dynamically at Servo assembly time.
    The concrete type is created in `Assembly.assemble()` and adds a field for each active
    connector.
    """

    connectors: List[BaseConnector]
    """
    The active connectors in the Servo.
    """

    @staticmethod
    def current() -> 'Servo':
        """
        Returns the active servo for the current execution context.

        The value is managed by a contextvar and is concurrency safe.
        """
        return _servo_context_var.get()

    def __init__(self, *args, connectors: List[BaseConnector], **kwargs) -> None:
        super().__init__(*args, connectors=[], **kwargs)

        # Ensure the connectors refer to the same objects by identity (required for eventing)
        self.connectors.extend(connectors)

        # associate our config with our children
        self._set_association("servo_config", self.config.servo)
        for connector in connectors:
            connector._set_association("servo_config", self.config.servo)

            with extra(connector):
                connector.api_client_options = self.api_client_options

    @property
    def api_client_options(self) -> Dict[str, Any]:
        options = super().api_client_options
        if self.config.servo:
            options["proxies"] = self.config.servo.proxies
            options["timeout"] = self.config.servo.timeouts
            options["verify"] = self.config.servo.ssl_verify

        return options

    async def startup(self):
        """
        Notifies all connectors that the servo is starting up.
        """
        await self.dispatch_event(Events.STARTUP, prepositions=Preposition.ON)

    async def shutdown(self):
        """
        Notifies all connectors that the servo is shutting down.
        """
        await self.dispatch_event(Events.SHUTDOWN, prepositions=Preposition.ON)

    def get_connector(self, name: Union[str, Sequence[str]]) -> Optional[Union[BaseConnector, List[BaseConnector]]]:
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
    async def check(self, filter_: Optional[Filter], halt_on: Optional[Severity] = Severity.critical) -> List[Check]:
        try:
            async with self.api_client() as client:
                event_request = api.Request(event=api.Event.HELLO)
                response = await client.post("servo", data=event_request.json())
                success = (response.status_code == httpx.codes.OK)
                return [Check(
                    name="Opsani API connectivity",
                    success=success,
                    message=f"Response status code: {response.status_code}",
                )]
        except Exception as error:
            return [Check(
                name="Opsani API connectivity",
                success=False,
                message=str(error),
            )]