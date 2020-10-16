import datetime
import enum
from typing import Any, Dict, List, Optional, Union

import devtools
import httpx
import pydantic

import servo.types

USER_AGENT = "github.com/opsani/servox"


class UnexpectedEventError(RuntimeError):
    pass


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
    status: str
    message: Optional[str]
    reason: Optional[str]


UNEXPECTED_EVENT = "unexpected-event"


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


class StatusMessage(pydantic.BaseModel):
    status: str
    message: Optional[str]


class Mixin:
    @property
    def api_headers(self) -> Dict[str, str]:
        if not self.optimizer:
            raise RuntimeError(
                f"cannot construct API headers: optimizer is not configured"
            )
        return {
            "Authorization": f"Bearer {self.optimizer.token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }

    @property
    def api_client_options(self) -> Dict[str, Any]:
        return dict(base_url=self.optimizer.api_url, headers=self.api_headers)

    def api_client(self, **kwargs) -> httpx.AsyncClient:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        if not self.optimizer:
            raise RuntimeError(
                f"cannot construct API client: optimizer is not configured"
            )
        return httpx.AsyncClient(**{**self.api_client_options, **kwargs})

    def api_client_sync(self, **kwargs) -> httpx.Client:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        if not self.optimizer:
            raise RuntimeError(
                f"cannot construct API client: optimizer is not configured"
            )
        return httpx.Client(**{**self.api_client_options, **kwargs})

    async def report_progress(self, **kwargs):
        request = self.progress_request(**kwargs)
        status = await self._post_event(*request)

        if status.status == UNEXPECTED_EVENT:
            # We have lost sync with the backend, raise an exception to halt broken execution
            raise UnexpectedEventError(status.reason)

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

    # NOTE: Opsani API primitive
    # @backoff.on_exception(backoff.expo, (httpx.HTTPError), max_time=180, max_tries=12)
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
            except httpx.HTTPError:
                self.logger.error(f"HTTP error encountered while posting {event} event")
                self.logger.trace(devtools.pformat(event_request))
                raise

    def _post_event_sync(self, event: Event, param) -> Union[CommandResponse, Status]:
        event_request = Request(event=event, param=param)
        with self.servo.api_client_sync() as client:
            try:
                response = client.post("servo", data=event_request.json())
                response.raise_for_status()
            except httpx.HTTPError:
                self.logger.error(
                    f"HTTP error encountered while posting {event.value} event"
                )
                self.logger.trace(devtools.pformat(event_request))
                raise

        return pydantic.parse_obj_as(Union[CommandResponse, Status], response.json())


def descriptor_to_adjustments(descriptor: dict) -> List[servo.types.Adjustment]:
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
