from __future__ import annotations

import asyncio
import contextvars
import enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import httpx

import servo.api
import servo.checks
import servo.configuration
import servo.connector
import servo.events
import servo.types
import servo.utilities
import servo.utilities.pydantic

_servo_context_var = contextvars.ContextVar("servo.Servo.current", default=None)


class Events(str, enum.Enum):
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
    @servo.events.event(Events.STARTUP)
    async def startup(self) -> None:
        ...

    @servo.events.event(Events.SHUTDOWN)
    async def shutdown(self) -> None:
        ...

    # Informational events
    @servo.events.event(Events.METRICS)
    async def metrics(self) -> List[servo.types.Metric]:
        ...

    @servo.events.event(Events.COMPONENTS)
    async def components(self) -> List[servo.types.Component]:
        ...

    # Operational events
    @servo.events.event(Events.MEASURE)
    async def measure(
        self,
        *,
        metrics: List[str] = None,
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Measurement:
        if control.delay:
            await asyncio.sleep(control.delay.total_seconds())
        yield

    @servo.events.event(Events.CHECK)
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[
            servo.types.ErrorSeverity
        ] = servo.types.ErrorSeverity.CRITICAL,
    ) -> List[servo.checks.Check]:
        ...

    @servo.events.event(Events.DESCRIBE)
    async def describe(self) -> servo.types.Description:
        ...

    @servo.events.event(Events.ADJUST)
    async def adjust(
        self,
        adjustments: List[servo.types.Adjustment],
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Description:
        ...

    @servo.events.event(Events.PROMOTE)
    async def promote(self) -> None:
        ...


_servo_context_var = contextvars.ContextVar("servo.servo", default=None)


class ServoChecks(servo.checks.BaseChecks):
    async def check_connectivity(self) -> Tuple[bool, str]:
        async with self.api_client() as client:
            event_request = servo.api.Request(event=servo.api.Event.HELLO)
            response = await client.post("servo", data=event_request.json())
            success = response.status_code == httpx.codes.OK
            return (success, f"Response status code: {response.status_code}")


@servo.connector.metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=servo.types.Maturity.ROBUST,
    license=servo.types.License.APACHE2,
    version=servo.__version__,
)
class Servo(servo.connector.BaseConnector):
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

    config: servo.configuration.BaseAssemblyConfiguration
    """Configuration of the Servo assembly.

    Note that the configuration is built dynamically at Servo assembly time.
    The concrete type is created in `Assembly.assemble()` and adds a field for each active
    connector.
    """

    connectors: List[servo.connector.BaseConnector]
    """
    The active connectors in the Servo.
    """

    @staticmethod
    def current() -> "Servo":
        """
        Returns the active servo for the current execution context.

        The value is managed by a contextvar and is concurrency safe.
        """
        return _servo_context_var.get()

    @staticmethod
    def set_current(servo_: "Servo") -> None:
        """Set the current servo execution context.
        """
        _servo_context_var.set(servo_)

    def __init__(
        self, *args, connectors: List[servo.connector.BaseConnector], **kwargs
    ) -> None:
        super().__init__(*args, connectors=[], **kwargs)

        # Ensure the connectors refer to the same objects by identity (required for eventing)
        self.connectors.extend(connectors)

        # associate our config with our children
        self._set_association("servo_config", self.config.servo)
        for connector in connectors:
            connector._set_association("servo_config", self.config.servo)

            with servo.utilities.pydantic.extra(connector):
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
        await self.dispatch_event(Events.STARTUP, prepositions=servo.events.Preposition.ON)

    async def shutdown(self):
        """
        Notifies all connectors that the servo is shutting down.
        """
        await self.dispatch_event(Events.SHUTDOWN, prepositions=servo.events.Preposition.ON)

    def get_connector(
        self, name: Union[str, Sequence[str]]
    ) -> Optional[Union[servo.connector.BaseConnector, List[servo.connector.BaseConnector]]]:
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

    async def add_connector(
        self, name: str, connector: servo.connector.BaseConnector
    ) -> None:
        """Adds a connector to the servo.

        The connector is added to the servo event bus and is initialized with
        the startup event to prepare for execution.

        Args:
            name: A unique name for the connector in the servo.
            connector: The connector to be added to the servo.

        Raises:
            ValueError: Raised if the name is not unique in the servo.
        """
        if self.get_connector(name):
            raise ValueError(
                f"invalid name: a connector named '{name}' already exists in the servo"
            )

        connector.name = name
        self.connectors.append(connector)
        self.__connectors__.append(connector)

        with servo.utilities.pydantic.extra(self.config):
            setattr(self.config, name, connector.config)

        await self.dispatch_event(
            Events.STARTUP, prepositions=servo.events.Preposition.ON, include=[connector]
        )

    async def remove_connector(
        self, connector: Union[str, servo.connector.BaseConnector]
    ) -> None:
        """Removes a connector from the servo.

        The connector is removed from the servo event bus and is finalized with
        the shutdown event to prepare for eviction.

        Args:
            connector: The connector or name to remove from the servo.

        Raises:
            ValueError: Raised if the connector does not exist in the servo.
        """
        connector_ = (
            connector
            if isinstance(connector, servo.connector.BaseConnector)
            else self.get_connector(connector)
        )
        if not connector_ in self.connectors:
            name = connector_.name if connector_ else connector
            raise ValueError(
                f"invalid connector: a connector named '{name}' does not exist in the servo"
            )

        await self.dispatch_event(
            Events.SHUTDOWN, prepositions=servo.events.Preposition.ON, include=[connector_]
        )

        self.connectors.remove(connector_)
        self.__connectors__.remove(connector_)

        with servo.utilities.pydantic.extra(self.config):
            delattr(self.config, connector_.name)

    ##
    # Event handlers

    @servo.events.on_event()
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[servo.types.ErrorSeverity] = servo.types.ErrorSeverity.CRITICAL,
    ) -> List[servo.checks.Check]:
        try:
            async with self.api_client() as client:
                event_request = servo.api.Request(event=servo.api.Event.HELLO)
                response = await client.post("servo", data=event_request.json())
                success = response.status_code == httpx.codes.OK
                return [
                    servo.checks.Check(
                        name="Opsani API connectivity",
                        success=success,
                        message=f"Response status code: {response.status_code}",
                    )
                ]
        except Exception as error:
            return [
                servo.checks.Check(
                    name="Opsani API connectivity",
                    success=False,
                    message=str(error),
                )
            ]
