from __future__ import annotations

import abc
import copy
import datetime
import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import curlify2
import devtools
import httpx
import pydantic

import servo.types
import servo.utilities

USER_AGENT = "github.com/opsani/servox"


class OptimizerStatuses(str, enum.Enum):
    """An enumeration of status types sent by the optimizer."""
    ok = "ok"
    invalid = "invalid"
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

class Events(str, enum.Enum):
    hello = "HELLO"
    whats_next = "WHATS_NEXT"
    describe = "DESCRIPTION"
    measure = "MEASUREMENT"
    adjust = "ADJUSTMENT"
    goodbye = "GOODBYE"

class Commands(str, enum.Enum):
    describe = "DESCRIBE"
    measure = "MEASURE"
    adjust = "ADJUST"
    sleep = "SLEEP"

    @property
    def response_event(self) -> Events:
        if self == Commands.describe:
            return Events.describe
        elif self == Commands.measure:
            return Events.measure
        elif self == Commands.adjust:
            return Events.adjust
        else:
            raise ValueError(f"unknoen command: {self}")


class Request(pydantic.BaseModel):
    event: Union[Events, str]  # TODO: Needs to be rethought -- used adhoc in some cases
    param: Optional[Dict[str, Any]]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Events: lambda v: str(v),
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
        return cls(status=ServoStatuses.ok, message=message, reason=reason, **kwargs)

    @classmethod
    def from_error(cls, error: servo.errors.BaseError) -> "Status":
        """Return a status object representation from the given error."""
        if isinstance(error, servo.errors.AdjustmentRejectedError):
            status = ServoStatuses.rejected
        else:
            status = ServoStatuses.failed

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

    @pydantic.validator('metrics', each_item=True, pre=True)
    def _map_metrics(cls, v) -> str:
        if isinstance(v, servo.Metric):
            return v.name

        return v


class CommandResponse(pydantic.BaseModel):
    command: Commands = pydantic.Field(alias="cmd")
    param: Optional[
        Union[MeasureParams, Dict[str, Any]]
    ]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Commands: lambda v: str(v),
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

        if status.status == OptimizerStatuses.ok:
            pass
        elif status.status == OptimizerStatuses.unexpected_event:
            # We have lost sync with the backend, raise an exception to halt broken execution
            raise servo.errors.UnexpectedEventError(status.reason)
        elif status.status == OptimizerStatuses.cancelled:
            # Optimizer wants to cancel the operation
            raise servo.errors.EventCancelledError(status.reason)
        elif status.status == OptimizerStatuses.invalid:
            servo.logger.warning(f"progress report was rejected as invalid")
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
    ) -> Tuple[str, Dict[str, Any]]:
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

        params = dict(
            progress=float(progress),
            runtime=float(runtime.total_seconds()),
        )
        set_if(params, "message", message)

        return (operation, params)

    def _is_fatal_status_code(error: Exception) -> bool:
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code < 500:
                servo.logger.warning(f"Giving up on non-retryable HTTP status code {error.response.status_code} ({error.response.reason_phrase}) for url: {error.request.url}")
                return True

        return False

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.current_servo() and servo.current_servo().config.settings.backoff.max_time(),
        max_tries=lambda: servo.current_servo() and servo.current_servo().config.settings.backoff.max_tries(),
        giveup=_is_fatal_status_code
    )
    async def _post_event(self, event: Events, param) -> Union[CommandResponse, Status]:
        async with self.api_client() as client:
            event_request = Request(event=event, param=param)
            self.logger.trace(f"POST event request: {devtools.pformat(event_request.json())}")

            try:
                response = await client.post("servo", data=event_request.json())
                response.raise_for_status()
                response_json = response.json()
                self.logger.trace(
                    f"POST event response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )
                self.logger.trace(_redacted_to_curl(response.request))

                return pydantic.parse_obj_as(
                    Union[CommandResponse, Status], response_json
                )

            except httpx.HTTPError as error:
                self.logger.error(f"HTTP error \"{error.__class__.__name__}\" encountered while posting \"{event}\" event: {error}")
                self.logger.trace(_redacted_to_curl(error.request))
                raise


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


def adjustments_to_descriptor(adjustments: List[servo.types.Adjustment]) -> Dict[str, Any]:
    components = {}
    descriptor = { "state": { "application": { "components": components }}}

    for adjustment in adjustments:
        if not adjustment.component_name in components:
            components[adjustment.component_name] = { "settings": {} }

        components[adjustment.component_name]["settings"][adjustment.setting_name] = { "value": adjustment.value }

    return descriptor


def user_agent() -> str:
    return f"{USER_AGENT} v{servo.__version__}"


def _redacted_to_curl(request: httpx.Request) -> str:
    """Pass through to curlify2.to_curl that redacts the authorization in the headers
    """
    if (auth_header := request.headers.get('authorization')) is None:
        return curlify2.to_curl(request)

    req_copy = copy.copy(request)
    req_copy.headers = copy.deepcopy(request.headers)
    if "Bearer" in auth_header:
        req_copy.headers['authorization'] = "Bearer [REDACTED]"
    else:
        req_copy.headers['authorization'] = "[REDACTED]"

    return curlify2.to_curl(req_copy)
