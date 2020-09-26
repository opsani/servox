from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from devtools import pformat
import httpx
from pydantic import BaseModel, Field, parse_obj_as, validator

from servo.types import Adjustment, Control, Duration, Numeric

USER_AGENT = "github.com/opsani/servox"

class UnexpectedEventError(RuntimeError):
    pass

class Command(str, Enum):
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


class Event(str, Enum):
    HELLO = "HELLO"
    GOODBYE = "GOODBYE"
    DESCRIPTION = "DESCRIPTION"
    WHATS_NEXT = "WHATS_NEXT"
    ADJUSTMENT = "ADJUSTMENT"
    MEASUREMENT = "MEASUREMENT"


class Request(BaseModel):
    event: Union[Event, str] # TODO: Needs to be rethought -- used adhoc in some cases
    param: Optional[Dict[str, Any]]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Event: lambda v: str(v),
        }


class Status(BaseModel):
    status: str
    message: Optional[str]
    reason: Optional[str]

UNEXPECTED_EVENT = 'unexpected-event'

class SleepResponse(BaseModel):
    pass


# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'

# Instructions from servo on what to measure
class MeasureParams(BaseModel):
    metrics: List[str]
    control: Control

    @validator('metrics', always=True, pre=True)
    @classmethod
    def coerce_metrics(cls, value) -> List[str]:
        if isinstance(value, dict):
            return list(value.keys())

        return value

class CommandResponse(BaseModel):
    command: Command = Field(alias="cmd",)
    param: Optional[
        Union[MeasureParams, Dict[str, Any]]
    ]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Command: lambda v: str(v),
        }


class StatusMessage(BaseModel):
    status: str
    message: Optional[str]


class Mixin:
    @property
    def api_headers(self) -> Dict[str, str]:
        if not self.optimizer:
            raise RuntimeError(f"cannot construct API headers: optimizer is not configured")
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
            raise RuntimeError(f"cannot construct API client: optimizer is not configured")
        return httpx.AsyncClient(**{ **self.api_client_options, **kwargs })

    def api_client_sync(self, **kwargs) -> httpx.Client:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        if not self.optimizer:
            raise RuntimeError(f"cannot construct API client: optimizer is not configured")
        return httpx.Client(**{ **self.api_client_options, **kwargs })

    async def report_progress(self, **kwargs):
        request = self.progress_request(**kwargs)
        status = await self._post_event(*request)

        if status.status == UNEXPECTED_EVENT:
            # We have lost sync with the backend, raise an exception to halt broken execution
            raise UnexpectedEventError(status.reason)

    def progress_request(self,
        operation: str,
        progress: Numeric,
        started_at: datetime,
        message: Optional[str],
        *,
        connector: Optional[str] = None,
        event_context: Optional['EventContext'] = None,
        time_remaining: Optional[Union[Numeric, Duration]] = None,
        logs: Optional[List[str]] = None,
    ) -> None:
        def set_if(d: Dict, k: str, v: Any):
            if v is not None: d[k] = v

        # Normalize progress to positive percentage
        if progress < 1.0:
            progress = progress * 100

        # Calculate runtime
        runtime = Duration(datetime.now() - started_at)

        # Produce human readable and remaining time in seconds values (if given)
        if time_remaining:
            if isinstance(time_remaining, (int, float)):
                time_remaining_in_seconds = time_remaining
                time_remaining = Duration(time_remaining_in_seconds)
            elif isinstance(time_remaining, timedelta):
                time_remaining_in_seconds = time_remaining.total_seconds()
            else:
                raise ValueError(f"Unknown value of type '{time_remaining.__class__.__name__}' for parameter 'time_remaining'")
        else:
            time_remaining_in_seconds = None

        params = dict(
            connector=self.name,
            operation=operation,
            progress=progress,
            runtime=str(runtime),
            runtime_in_seconds=runtime.total_seconds()
        )
        set_if(params, 'connector', connector)
        set_if(params, 'event', str(event_context))
        set_if(params, 'time_remaining', str(time_remaining) if time_remaining else None)
        set_if(params, 'time_remaining_in_seconds', str(time_remaining_in_seconds) if time_remaining_in_seconds else None)
        set_if(params, 'message', message)
        set_if(params, 'logs', logs)

        return (operation, params)


    # NOTE: Opsani API primitive
    # @backoff.on_exception(backoff.expo, (httpx.HTTPError), max_time=180, max_tries=12)
    async def _post_event(self, event: Event, param) -> Union[CommandResponse, Status]:        
        async with self.api_client() as client:
            event_request = Request(event=event, param=param)
            self.logger.trace(f"POST event request: {pformat(event_request)}")

            try:
                response = await client.post("servo", data=event_request.json())
                response.raise_for_status()
                response_json = response.json()
                self.logger.trace(f"POST event response ({response.status_code} {response.reason_phrase}): {pformat(response_json)}")

                return parse_obj_as(Union[CommandResponse, Status], response_json)
            except httpx.HTTPError:
                self.logger.error(
                    f"HTTP error encountered while posting {event} event"
                )
                self.logger.trace(pformat(event_request))
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
                self.logger.trace(pformat(event_request))
                raise

        return parse_obj_as(Union[CommandResponse, Status], response.json())

def descriptor_to_adjustments(descriptor: dict) -> List[Adjustment]:
    adjustments = []
    for component_name, component in descriptor["application"]["components"].items():
        for setting_name, attrs in component["settings"].items():
            adjustment = Adjustment(
                component_name=component_name,
                setting_name=setting_name,
                value=attrs["value"]
            )
            adjustments.append(adjustment)
    return adjustments
