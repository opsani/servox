"""The servo module is responsible for interacting with the Opsani optimizer API."""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import json
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import httpx
import pydantic

import servo.api
import servo.checks
import servo.configuration
import servo.connector
import servo.events
import servo.pubsub
import servo.types
import servo.utilities
import servo.utilities.pydantic

__all__ = ['Servo', 'Events', 'current_servo']


_current_context_var = contextvars.ContextVar("servox.current_servo", default=None)

def current_servo() -> Optional["Servo"]:
    """Return the active servo for the current execution context.

    The value is managed by a contextvar and is concurrency safe.
    """
    return _current_context_var.get(None)

def _set_current_servo(servo_: Optional["Servo"]) -> None:
    """Set the active servo for the current execution context.

    The value is managed by a contextvar and is concurrency safe.
    """
    _current_context_var.set(servo_)


class Events(str, enum.Enum):
    """An enumeration of the names of standard events defined by the servo."""

    # Lifecycle events
    attach = "attach"
    detach = "detach"
    startup = "startup"
    shutdown = "shutdown"

    # Informational events
    metrics = "metrics"
    components = "components"

    # Operational events
    check = "check"
    describe = "describe"
    measure = "measure"
    adjust = "adjust"
    promote = "promote"


class _EventDefinitions(Protocol):
    """Defines the default events. This class is declarative and is never directly referenced.

    The event signature is inferred from the decorated function.
    """

    # Lifecycle events
    @servo.events.event(Events.attach)
    async def attach(self, servo_: Servo) -> None:
        ...

    @servo.events.event(Events.detach)
    async def detach(self, servo_: Servo) -> None:
        ...

    @servo.events.event(Events.startup)
    async def startup(self) -> None:
        ...

    @servo.events.event(Events.shutdown)
    async def shutdown(self) -> None:
        ...

    # Informational events
    @servo.events.event(Events.metrics)
    async def metrics(self) -> List[servo.types.Metric]:
        ...

    @servo.events.event(Events.components)
    async def components(self) -> List[servo.types.Component]:
        ...

    # Operational events
    @servo.events.event(Events.measure)
    async def measure(
        self,
        *,
        metrics: List[str] = None,
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Measurement:
        if control.delay:
            await asyncio.sleep(control.delay.total_seconds())
        yield

    @servo.events.event(Events.check)
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[
            servo.types.ErrorSeverity
        ] = servo.types.ErrorSeverity.critical,
    ) -> List[servo.checks.Check]:
        ...

    @servo.events.event(Events.describe)
    async def describe(self, control: servo.types.Control = servo.types.Control()) -> servo.types.Description:
        ...

    @servo.events.event(Events.adjust)
    async def adjust(
        self,
        adjustments: List[servo.types.Adjustment],
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Description:
        ...

    @servo.events.event(Events.promote)
    async def promote(self) -> None:
        ...


class ServoChecks(servo.checks.BaseChecks):
    """Check that a servo is ready to perform optimization.

    Args:
        servo: The servo to be checked.
    """

    async def check_connectivity(self) -> Tuple[bool, str]:
        """Check that the servo has connectivity to the Opsani API.

        Returns:
            A tuple value containing a boolean that indicates if connectivity is
            available and an advisory string describing the status encountered.
        """
        async with self.api_client() as client:
            event_request = servo.api.Request(event=servo.api.Events.hello)
            response = await client.post("servo", data=event_request.json())
            success = response.status_code == httpx.codes.OK
            return (success, f"Response status code: {response.status_code}")


@servo.connector.metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=servo.types.Maturity.robust,
    license=servo.types.License.apache2,
    version=servo.__version__,
    cryptonym=servo.__cryptonym__,
)
class Servo(servo.connector.BaseConnector):
    """A connector that interacts with the Opsani API to perform optimization.

    The `Servo` is a core object of the `servo` package. It manages a set of
    connectors that provide integration and interactivity to external services
    such as metrics collectors, orchestation systems, load generators, etc. The
    Servo acts primarily as an event gateway between the Opsani API and its child
    connectors.

    Servo objects are configured with a dynamically created class that is built by
    the `servo.Assembly` class. Servo objects are typically not created directly
    and are instead built through the `Assembly.assemble` method.

    Attributes:
        connectors...
        config...
    """

    config: servo.configuration.BaseServoConfiguration
    """Configuration of the Servo assembly.

    Note that the configuration is built dynamically at Servo assembly time.
    The concrete type is created in `Assembly.assemble()` and adds a field for each active
    connector.
    """

    connectors: List[servo.connector.BaseConnector]
    """The active connectors in the Servo.
    """

    _running: bool = pydantic.PrivateAttr(False)

    async def dispatch_event(self, *args, **kwargs) -> Union[Optional[servo.events.EventResult], List[servo.events.EventResult]]:
        with self.current():
            return await super().dispatch_event(*args, **kwargs)

    def __init__(
        self, *args, connectors: List[servo.connector.BaseConnector], **kwargs
    ) -> None: # noqa: D107
        super().__init__(*args, connectors=[], **kwargs)

        # Ensure the connectors refer to the same objects by identity (required for eventing)
        self.connectors.extend(connectors)

        # associate shared config with our children
        for connector in (connectors + [self]):
            connector._global_config = self.config.settings

    @pydantic.root_validator()
    def _initialize_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["name"] == "servo" and values.get('config'):
            values["name"] = (
                values["config"].name
                or getattr(values["config"].optimizer, 'id', 'servo')
            )

        return values

    async def attach(self, servo_: servo.assembly.Assembly) -> None:
        """Notify the servo that it has been attached to an Assembly."""
        await self.dispatch_event(Events.attach, self)

    async def detach(self, servo_: servo.assembly.Assembly) -> None:
        """Notify the servo that it has been detached from an Assembly."""
        await self.dispatch_event(Events.detach, self)

    @property
    def is_running(self) -> bool:
        """Return True if the servo is running."""
        return self._running

    async def startup(self):
        """Notify all active connectors that the servo is starting up."""
        if self.is_running:
            raise RuntimeError("Cannot start up a servo that is already running")

        self._running = True

        await self.dispatch_event(Events.startup, _prepositions=servo.events.Preposition.on)

        # Start up the pub/sub exchange
        if not self.pubsub_exchange.running:
            self.pubsub_exchange.start()

    async def shutdown(self):
        """Notify all active connectors that the servo is shutting down."""
        if not self.is_running:
            raise RuntimeError("Cannot shut down a servo that is not running")

        # Remove all the connectors (dispatches shutdown event)
        await asyncio.gather(*list(map(self.remove_connector, self.connectors)))

        # Shut down the pub/sub exchange
        if self.pubsub_exchange.running:
            await self.pubsub_exchange.shutdown()

        self._running = False

    @property
    def all_connectors(self) -> List[servo.connector.BaseConnector]:
        """Return a list of all active connectors including the Servo."""
        return [self, *self.connectors]

    def get_connector(
        self, name: Union[str, Sequence[str]]
    ) -> Optional[Union[servo.connector.BaseConnector, List[servo.connector.BaseConnector]]]:
        """Return one or more connectors by name.

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
        """Add a connector to the servo.

        The connector is added to the servo event bus and is initialized with
        the `attach` event to prepare for execution. If the servo is currently
        running, the connector is sent the `startup` event as well.

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
        connector._global_config = self.config.settings

        # Add to the event bus
        self.connectors.append(connector)
        self.__connectors__.append(connector)

        # Add to the pub/sub exchange
        connector.pubsub_exchange = self.pubsub_exchange

        # Register our name into the config class
        with servo.utilities.pydantic.extra(self.config):
            setattr(self.config, name, connector.config)

        await self.dispatch_event(Events.attach, self, include=[connector])

        # Start the connector if we are running
        if self.is_running:
            await self.dispatch_event(
                Events.startup, include=[connector], _prepositions=servo.events.Preposition.on
            )

    async def remove_connector(
        self, connector: Union[str, servo.connector.BaseConnector]
    ) -> None:
        """Remove a connector from the servo.

        The connector is removed from the servo event bus and is finalized with
        the detach event to prepare for eviction. If the servo is currently running,
        the connector is sent the shutdown event as well.

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

        # Shut the connector down if we are running
        if self.is_running:
            await self.dispatch_event(
                Events.shutdown, include=[connector_], _prepositions=servo.events.Preposition.on
            )

        await self.dispatch_event(Events.detach, self, include=[connector])

        # Remove from the event bus
        self.connectors.remove(connector_)
        self.__connectors__.remove(connector_)

        # Remove from the pub/sub exchange
        connector_.cancel_subscribers()
        connector_.cancel_publishers()
        connector_.pubsub_exchange = servo.pubsub.Exchange()

        with servo.utilities.pydantic.extra(self.config):
            delattr(self.config, connector_.name)

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        """Return a schema that only includes connector model definitions"""
        connectors = servo.Assembly.all_connector_types() if all else self.connectors
        config_models = list(map(lambda c: c.config_model(), connectors))
        return pydantic.schema.schema(config_models, title="Servo Schema")

    def top_level_schema_json(self, *, all: bool = False) -> str:
        """Return a JSON string representation of the top level schema"""
        return json.dumps(
            self.top_level_schema(all=all),
            indent=2,
            default=pydantic.json.pydantic_encoder,
        )

    ##
    # Event handlers

    @servo.events.on_event()
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[servo.types.ErrorSeverity] = servo.types.ErrorSeverity.critical,
    ) -> List[servo.checks.Check]:
        """Check that the servo is ready to perform optimization.

        Args:
            matching: An optional filter to limit the checks that are executed.
            halt_on: The severity level of errors that should halt execution of checks.

        Returns:
            A list of check objects that describe the outcomes of the checks that were run.
        """
        try:
            async with self.api_client() as client:
                event_request = servo.api.Request(event=servo.api.Events.hello)
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

    @contextlib.contextmanager
    def current(self):
        """A context manager that sets the current servo context."""
        try:
          token = _current_context_var.set(self)
          yield self

        finally:
            _current_context_var.reset(token)
