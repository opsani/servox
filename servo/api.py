from __future__ import annotations

import abc
import datetime
import enum
from typing import Any, Dict, List, Optional, Union

import backoff
import devtools
import httpx
import pydantic

import servo.utilities
import servo.types

USER_AGENT = "github.com/opsani/servox"

class OptimizerStatuses(str, enum.Enum):
    """An enumeration of status types sent by the optimizer."""
    ok = "ok"
    unexpected_event = "unexpected-event"
    cancelled = "cancel"

class ServoStatuses(str, enum.Enum):
    """An enumeration of status types sent from the servo."""
    ok = "ok"
    failed = "failed"
    rejected = "rejected"
    aborted = "aborted"


Statuses = Union[OptimizerStatuses, ServoStatuses]


class Reasons(str, enum.Enum):
    success = "success"
    unknown = "unknown"
    unstable = "unstable"


class Command(str, enum.Enum):
    DESCRIBE = "DESCRIBE"
    MEASURE = "MEASURE"
    ADJUST = "ADJUST"
    SLEEP = "SLEEP"

    @property
    def response_event(self) -> str:
        if self == Command.DESCRIBE:
            return Event.DESCRIPTION
        elif self == Command.MEASURE:
            return Event.MEASUREMENT
        elif self == Command.ADJUST:
            return Event.ADJUSTMENT
        else:
            return None


class Event(str, enum.Enum):
    HELLO = "HELLO"
    GOODBYE = "GOODBYE"
    DESCRIPTION = "DESCRIPTION"
    WHATS_NEXT = "WHATS_NEXT"
    ADJUSTMENT = "ADJUSTMENT"
    MEASUREMENT = "MEASUREMENT"


class Request(pydantic.BaseModel):
    event: Union[Event, str]  # TODO: Needs to be rethought -- used adhoc in some cases
    param: Optional[Dict[str, Any]]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Event: lambda v: str(v),
        }


class Status(pydantic.BaseModel):
    status: Statuses
    message: Optional[str] = None
    reason: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    descriptor: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, message: Optional[str] = None, reason: str = Reasons.success, **kwargs) -> "Status":
        """Return a success (status="ok") status object."""
        return cls(status=Statuses.ok, message=message, reason=reason, **kwargs)

    @classmethod
    def from_error(cls, error: servo.errors.BaseError) -> "Status":
        """Return a status object representation from the given error."""
        if isinstance(error, servo.errors.AdjustmentRejectedError):
            status = Statuses.rejected
        else:
            status = Statuses.failed

        return cls(status=status, message=str(error), reason=error.reason)

    def dict(
        self,
        *,
        exclude_unset: bool = True,
        **kwargs,
    ) -> pydantic.DictStrAny:
        return super().dict(exclude_unset=exclude_unset, **kwargs)

class SleepResponse(pydantic.BaseModel):
    pass
# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'

# Instructions from servo on what to measure
class MeasureParams(pydantic.BaseModel):
    metrics: List[str]
    control: servo.types.Control

    @pydantic.validator("metrics", always=True, pre=True)
    @classmethod
    def coerce_metrics(cls, value) -> List[str]:
        if isinstance(value, dict):
            return list(value.keys())

        return value


class CommandResponse(pydantic.BaseModel):
    command: Command = pydantic.Field(alias="cmd")
    param: Optional[
        Union[MeasureParams, Dict[str, Any]]
    ]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Command: lambda v: str(v),
        }


class Mixin(abc.ABC):
    """Provides functionality for interacting with the Opsani API via httpx.

    The mixin requires the implementation of the `api_client_options` method
    which is responsible for providing details around base URL, HTTP headers,
    timeouts, proxies, SSL configuration, etc. for initializing
    `httpx.AsyncClient` and `httpx.Client` instances.
    """

    @property
    @abc.abstractmethod
    def api_client_options(self) -> Dict[str, Any]:
        """Return a dict of options for initializing httpx API client objects.

        An implementation must be provided in subclasses derived from the mixin
        and is responsible for appropriately configuring the base URL, HTTP
        headers, timeouts, proxies, SSL configuration, transport flags, etc.

        The dict returned is passed directly to the initializer of
        `httpx.AsyncClient` and `httpx.Client` objects constructed by the
        `api_client` and `api_client_sync` methods.
        """
        ...
        # TODO: Overloaded for performance...
        # timeout = httpx.Timeout(60.0)
        # limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
        # return dict(base_url=self.optimizer.api_url, headers=self.api_headers, timeout=timeout, limits=limits)

    def api_client(self, **kwargs) -> httpx.AsyncClient:
        """Return an asynchronous client for interacting with the Opsani API."""
        return httpx.AsyncClient(**{**self.api_client_options, **kwargs})

    def api_client_sync(self, **kwargs) -> httpx.Client:
        """Return a synchronous client for interacting with the Opsani API."""
        return httpx.Client(**{**self.api_client_options, **kwargs})

    async def report_progress(self, **kwargs) -> None:
        """Post a progress report to the Opsani API."""
        request = self.progress_request(**kwargs)
        status = await self._post_event(*request)

        if status.status == Statuses.ok:
            pass
        elif status.status == Statuses.unexpected_event:
            # We have lost sync with the backend, raise an exception to halt broken execution
            raise servo.errors.UnexpectedEventError(status.reason)
        elif status.status == Statuses.cancelled:
            # Optimizer wants to cancel the operation
            raise servo.errors.EventCancelledError(status.reason)
        else:
            raise ValueError(f"unknown error status: \"{status.status}\"")

    def progress_request(
        self,
        operation: str,
        progress: servo.types.Numeric,
        started_at: datetime,
        message: Optional[str],
        *,
        connector: Optional[str] = None,
        event_context: Optional["servo.events.EventContext"] = None,
        time_remaining: Optional[
            Union[servo.types.Numeric, servo.types.Duration]
        ] = None,
        logs: Optional[List[str]] = None,
    ) -> None:
        def set_if(d: Dict, k: str, v: Any):
            if v is not None:
                d[k] = v

        # Normalize progress to positive percentage
        if progress < 1.0:
            progress = progress * 100

        # Calculate runtime
        runtime = servo.types.Duration(datetime.datetime.now() - started_at)

        # Produce human readable and remaining time in seconds values (if given)
        if time_remaining:
            if isinstance(time_remaining, (int, float)):
                time_remaining_in_seconds = time_remaining
                time_remaining = servo.types.Duration(time_remaining_in_seconds)
            elif isinstance(time_remaining, datetime.timedelta):
                time_remaining_in_seconds = time_remaining.total_seconds()
            else:
                raise ValueError(
                    f"Unknown value of type '{time_remaining.__class__.__name__}' for parameter 'time_remaining'"
                )
        else:
            time_remaining_in_seconds = None

        # FIXME: Tied off until reconciled with new oco-e enforcement (see https://github.com/opsani/oco/blob/wfc-devel/transport/protocol.py#L62)
        params = dict(
            # connector=self.name,
            # operation=operation,
            progress=float(progress),
            runtime=float(runtime.total_seconds()),
            # runtime=str(runtime),
            # runtime_in_seconds=runtime.total_seconds(),
        )
        # set_if(params, "connector", connector)
        # set_if(params, "event", str(event_context))
        # set_if(
        #     params, "time_remaining", str(time_remaining) if time_remaining else None
        # )
        # set_if(
        #     params,
        #     "time_remaining_in_seconds",
        #     str(time_remaining_in_seconds) if time_remaining_in_seconds else None,
        # )
        set_if(params, "message", message)
        # set_if(params, "logs", logs)

        return (operation, params)

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.Servo.current().config.servo.backoff.max_time(),
        max_tries=lambda: servo.Servo.current().config.servo.backoff.max_tries(),
    )
    async def _post_event(self, event: Event, param) -> Union[CommandResponse, Status]:
        async with self.api_client() as client:
            event_request = Request(event=event, param=param)
            self.logger.trace(f"POST event request: {devtools.pformat(event_request)}")

            try:
                response = await client.post("servo", data=event_request.json())
                response.raise_for_status()
                response_json = response.json()
                self.logger.trace(
                    f"POST event response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )

                return pydantic.parse_obj_as(
                    Union[CommandResponse, Status], response_json
                )
            except httpx.HTTPError as error:
                self.logger.error(f"HTTP error \"{error.__class__.__name__}\" encountered while posting \"{event}\" event: {error}")
                self.logger.trace(devtools.pformat(event_request))
                raise

    def _post_event_sync(self, event: Event, param) -> Union[CommandResponse, Status]:
        event_request = Request(event=event, param=param)
        with self.servo.api_client_sync() as client:
            try:
                response = client.post("servo", data=event_request.json())
                response.raise_for_status()
            except httpx.HTTPError as error:
                self.logger.error(
                    f"HTTP error \"{error.__class__.__name__}\" encountered while posting {event.value} event: {error}"
                )
                self.logger.trace(devtools.pformat(event_request))
                raise

        return pydantic.parse_obj_as(Union[CommandResponse, Status], response.json())


def descriptor_to_adjustments(descriptor: dict) -> List[servo.types.Adjustment]:
    """Return a list of adjustment objects from an Opsani API app descriptor."""
    adjustments = []
    for component_name, component in descriptor["application"]["components"].items():
        for setting_name, attrs in component["settings"].items():
            adjustment = servo.types.Adjustment(
                component_name=component_name,
                setting_name=setting_name,
                value=attrs["value"],
            )
            adjustments.append(adjustment)
    return adjustments
