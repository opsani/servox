# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The servo module is responsible for interacting with the Opsani optimizer API."""
from __future__ import annotations

import asyncio
import backoff
import contextlib
import contextvars
from datetime import datetime, timedelta
import devtools
import enum
import json
from typing import (
    cast,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from authlib.integrations.httpx_client import AsyncOAuth2Client
import httpx
import pydantic
import watchfiles

import servo.api
import servo.checks
import servo.configuration
import servo.connector
import servo.events
import servo.errors
import servo.pubsub
import servo.types
import servo.utilities
import servo.utilities.pydantic

__all__ = ["Servo", "Events", "current_servo", "current_command_uid"]


_current_context_var = contextvars.ContextVar("servox.current_servo", default=None)
_current_command_uid_context_var = contextvars.ContextVar(
    "servox.current_command_uid", default=None
)


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


def current_command_uid() -> Union[str, None]:
    """Return the command ID that the current asyncio task was invoked under.

    A new copy is automatically generated on creation of tasks and functions as a closure of the ID even after it is
    updated by the main loop task
    """
    return _current_command_uid_context_var.get()


def set_current_command_uid(value: Union[str, None]) -> None:
    _current_command_uid_context_var.set(value)


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
    async def metrics(self) -> list[servo.types.Metric]:
        ...

    @servo.events.event(Events.components)
    async def components(self) -> list[servo.types.Component]:
        ...

    # Operational events
    @servo.events.event(Events.measure)
    async def measure(
        self,
        *,
        metrics: list[str] = None,
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Measurement:
        ...

    @servo.events.event(Events.check)
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[
            servo.types.ErrorSeverity
        ] = servo.types.ErrorSeverity.critical,
    ) -> list[servo.checks.Check]:
        ...

    @servo.events.event(Events.describe)
    async def describe(
        self, control: servo.types.Control = servo.types.Control()
    ) -> servo.types.Description:
        ...

    @servo.events.event(Events.adjust)
    async def adjust(
        self,
        adjustments: list[servo.types.Adjustment],
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.types.Description:
        ...

    @servo.events.event(Events.promote)
    async def promote(self) -> None:
        ...


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

    connectors: list[servo.connector.BaseConnector]
    """The active connectors in the Servo.
    """

    _api_client: Union[httpx.AsyncClient, AsyncOAuth2Client] = pydantic.PrivateAttr(
        None
    )
    """An asynchronous client for interacting with the Opsani API."""

    _running: bool = pydantic.PrivateAttr(False)

    async def dispatch_event(
        self, *args, **kwargs
    ) -> Union[Optional[servo.events.EventResult], list[servo.events.EventResult]]:
        with self.current():
            return await super().dispatch_event(*args, **kwargs)

    def __init__(
        self, *args, connectors: list[servo.connector.BaseConnector], **kwargs
    ) -> None:  # noqa: D107
        super().__init__(*args, connectors=[], **kwargs)

        self._api_client = servo.api.get_api_client_for_optimizer(
            self.config.optimizer, self.config.settings
        )

        # Ensure the connectors refer to the same objects by identity (required for eventing)
        self.connectors.extend(connectors)

        # associate shared config with our children
        for connector in connectors + [self]:
            connector._global_config = self.config.settings
            connector._optimizer = self.config.optimizer

    @pydantic.root_validator()
    def _initialize_name(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values["name"] == "servo" and values.get("config"):
            values["name"] = values["config"].name or getattr(
                values["config"].optimizer, "id", "servo"
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

    async def startup(self) -> None:
        """Notify all active connectors that the servo is starting up."""
        if self.is_running:
            raise RuntimeError("Cannot start up a servo that is already running")

        self._running = True

        await self.dispatch_event(
            Events.startup, _prepositions=servo.events.Preposition.on
        )

        # Start up the pub/sub exchange
        if not self.pubsub_exchange.running:
            self.pubsub_exchange.start()

    async def shutdown(self) -> None:
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
    def all_connectors(self) -> list[servo.connector.BaseConnector]:
        """Return a list of all active connectors including the Servo."""
        return [self, *self.connectors]

    def connectors_named(self, names: Sequence[str]) -> list[servo.BaseConnector]:
        return [
            connector for connector in self.all_connectors if connector.name in names
        ]

    def get_connector(
        self, name: Union[str, Sequence[str]]
    ) -> Optional[
        Union[servo.connector.BaseConnector, list[servo.connector.BaseConnector]]
    ]:
        """Return one or more connectors by name.

        This is a convenience method equivalent to iterating `connectors` and comparing by name.

        When given a single name, returns the connector or `None` if not found.
        When given a sequence of names, returns a list of Connectors for all connectors found.
        """
        if isinstance(name, str):
            return next(iter(self.connectors_named([name])), None)
        else:
            return self.connectors_named(name)

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
        connector._optimizer = self.config.optimizer

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
                Events.startup,
                include=[connector],
                _prepositions=servo.events.Preposition.on,
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
                Events.shutdown,
                include=[connector_],
                _prepositions=servo.events.Preposition.on,
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

    def top_level_schema(self, *, all: bool = False) -> dict[str, Any]:
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

    async def report_progress(self, **kwargs) -> None:
        """Post a progress report to the Opsani API."""
        request = self.progress_request(**kwargs)
        status = await self.post_event(*request)

        if status.status == servo.api.OptimizerStatuses.ok:
            pass
        elif status.status == servo.api.OptimizerStatuses.unexpected_event:
            # We have lost sync with the backend, raise an exception to halt broken execution
            raise servo.errors.UnexpectedEventError(status.reason)
        elif status.status == servo.api.OptimizerStatuses.cancelled:
            # Optimizer wants to cancel the operation
            raise servo.errors.EventCancelledError(status.reason or "Command cancelled")
        elif status.status == servo.api.OptimizerStatuses.invalid:
            self.logger.warning(
                f"progress report was rejected as invalid: {devtools.pformat(status.dict())}"
            )
            if status.reason == "unexpected cmd_uid":
                raise servo.errors.UnexpectedCommandIdError(status.reason)
        else:
            raise ValueError(f'unknown error status: "{status.status}"')

    def progress_request(
        self,
        operation: str,
        progress: servo.types.Numeric,
        started_at: datetime,
        message: Optional[str],
        *,
        command_uid: Union[str, None] = None,
        connector: Optional[str] = None,
        event_context: Optional["servo.events.EventContext"] = None,
        time_remaining: Optional[
            Union[servo.types.Numeric, servo.types.Duration]
        ] = None,
        logs: Optional[list[str]] = None,
    ) -> Tuple[str, dict[str, Any]]:
        def set_if(d: dict, k: str, v: Any):
            if v is not None:
                d[k] = v

        # Calculate runtime
        runtime = servo.types.Duration(datetime.now() - started_at)

        # Produce human readable and remaining time in seconds values (if given)
        if time_remaining:
            if isinstance(time_remaining, (int, float)):
                time_remaining_in_seconds = time_remaining
                time_remaining = servo.types.Duration(time_remaining_in_seconds)
            elif isinstance(time_remaining, timedelta):
                time_remaining_in_seconds = time_remaining.total_seconds()
            else:
                raise ValueError(
                    f"Unknown value of type '{time_remaining.__class__.__name__}' for parameter 'time_remaining'"
                )
        else:
            time_remaining_in_seconds = None

        params = dict(
            progress=float(progress),
            runtime=float(runtime.total_seconds()),
            cmd_uid=command_uid,
        )
        set_if(params, "message", message)

        return (operation, params)

    async def post_event(
        self, event: Events, param
    ) -> Union[servo.api.CommandResponse, servo.api.Status]:
        @backoff.on_exception(
            backoff.expo,
            httpx.HTTPError,
            max_time=lambda: self.config.settings.backoff.max_time(),
            max_tries=lambda: self.config.settings.backoff.max_tries(),
            giveup=servo.api.is_fatal_status_code,
        )
        async def _post_event(
            event: Events, param
        ) -> Union[servo.api.CommandResponse, servo.api.Status]:
            event_request = servo.api.Request(
                event=event, param=param, servo_uid=self.config.servo_uid
            )
            self.logger.trace(
                f"POST event request: {devtools.pformat(event_request.json())}"
            )

            try:
                response = await self._api_client.post(
                    "servo", data=event_request.json()
                )
                response.raise_for_status()
                response_json = response.json()
                self.logger.trace(
                    f"POST event response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )
                self.logger.trace(servo.api.redacted_to_curl(response.request))

                return pydantic.parse_obj_as(
                    Union[servo.api.CommandResponse, servo.api.Status], response_json
                )

            except httpx.HTTPError as error:
                if hasattr(error, "response"):
                    response_text = devtools.pformat(error.response.text)
                else:
                    response_text = "(No response on error)"
                self.logger.error(
                    f'HTTP error "{error.__class__.__name__}" encountered while posting "{event}" event: {error}, for '
                    f"url {error.request.url} \n\n Response: {response_text}"
                )
                self.logger.trace(servo.api.redacted_to_curl(error.request))
                raise

        return await _post_event(event, param)

    async def watch_connection_file(self) -> None:
        connection_file = cast(
            servo.configuration.AppdynamicsOptimizer, self.config.optimizer
        ).connection_file
        async for changes in watchfiles.awatch(connection_file):
            self.logger.info(f"Loading change to connection file {changes}")
            self.config.optimizer.load_connection_file()
            self._api_client = servo.api.get_api_client_for_optimizer(
                self.config.optimizer, self.config.settings
            )

    async def check_servo(self, print_callback: Callable[[str], None] = None) -> bool:
        connectors = self.config.checks.connectors
        name = self.config.checks.name
        id = self.config.checks.id
        tag = self.config.checks.tag

        quiet = self.config.checks.quiet
        progressive = self.config.checks.progressive
        wait = self.config.checks.wait
        delay = self.config.checks.delay
        delay_generator = servo.checks.CheckHelpers.delay_generator(delay=delay)
        halt_on = self.config.checks.halt_on

        # Validate that explicit args support check events
        connector_objs = (
            self.connectors_named(connectors)
            if connectors
            else list(
                filter(
                    lambda c: c.responds_to_event(servo.Events.check),
                    self.all_connectors,
                )
            )
        )
        if not connector_objs:
            if connectors:
                raise servo.ConnectorNotFoundError(
                    f"no connector found with name(s) '{connectors}'"
                )
            else:
                raise servo.EventHandlersNotFoundError(
                    f"no currently assembled connectors respond to the check event"
                )
        validate_connectors_respond_to_event(connector_objs, servo.Events.check)

        if wait:
            summary = "Running checks"
            summary += " progressively" if progressive else ""
            summary += f" for up to {wait} with a delay of {delay} between iterations"
            servo.logger.info(summary)

        passing = set()
        progress = servo.DurationProgress(servo.Duration(wait or 0))
        ready = False

        while not progress.finished:
            if not progress.started:
                # run at least one time
                progress.start()

            args = dict(
                name=servo.utilities.parse_re(name),
                id=servo.utilities.parse_id(id),
                tags=servo.utilities.parse_csv(tag),
            )
            constraints = dict(filter(lambda i: bool(i[1]), args.items()))

            results: list[servo.EventResult] = (
                await self.dispatch_event(
                    servo.Events.check,
                    servo.CheckFilter(**constraints),
                    include=self.all_connectors,
                    halt_on=halt_on,
                )
                or []
            )

            ready = await servo.checks.CheckHelpers.process_checks(
                checks_config=self.config.checks,
                results=results,
                passing=passing,
            )
            if not progressive and not quiet:
                output = await servo.checks.CheckHelpers.checks_to_table(
                    checks_config=self.config.checks, results=results
                )
                print_callback(output)

            if ready:
                return ready
            else:
                if wait:
                    next_delay = next(delay_generator)
                    servo.logger.info(
                        f"waiting for {next_delay} seconds before rerunning failing checks"
                    )
                    await asyncio.sleep(next_delay)

                if progress.finished:
                    # Don't log a timeout if we aren't running in wait mode
                    if progress.duration:
                        servo.logger.error(
                            f"timed out waiting for checks to pass {progress.duration}"
                        )
                    return ready

    ##
    # Event handlers

    @servo.events.on_event()
    async def check(
        self,
        matching: Optional[servo.checks.CheckFilter],
        halt_on: Optional[
            servo.types.ErrorSeverity
        ] = servo.types.ErrorSeverity.critical,
    ) -> list[servo.checks.Check]:
        """Check that the servo is ready to perform optimization.

        Args:
            matching: An optional filter to limit the checks that are executed.
            halt_on: The severity level of errors that should halt execution of checks.

        Returns:
            A list of check objects that describe the outcomes of the checks that were run.
        """
        try:
            event_request = servo.api.Request(
                event=servo.api.Events.hello, servo_uid=self.config.servo_uid
            )
            response = await self._api_client.post("servo", data=event_request.json())
            if (
                response.status_code == 410
                and response.json().get("detail") == "unexpected servo_uid"
            ):
                self.logger.warning(
                    f"servo UID {self.config.servo_uid} is no longer valid. Waiting for deprovisioning (will sleep for 1 hour)"
                )
                await asyncio.sleep(3600)

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
                    exception=error,
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


def validate_connectors_respond_to_event(
    connectors: Iterable[servo.BaseConnector], event: str
) -> None:
    for connector in connectors:
        if not connector.responds_to_event(event):
            raise servo.EventHandlersNotFoundError(
                f"no currently assembled connectors respond to the check event"
            )
