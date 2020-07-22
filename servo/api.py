import abc
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import backoff
import httpx
from pydantic import BaseModel, Field, parse_obj_as

from servo.configuration import Optimizer
from servo.types import Control, Description, Duration, Measurement, Numeric


USER_AGENT = "github.com/opsani/servox"


class Command(str, Enum):
    DESCRIBE = "DESCRIBE"
    MEASURE = "MEASURE"
    ADJUST = "ADJUST"
    SLEEP = "SLEEP"


class Event(str, Enum):
    HELLO = "HELLO"
    GOODBYE = "GOODBYE"
    DESCRIPTION = "DESCRIPTION"
    WHATS_NEXT = "WHATS_NEXT"
    ADJUSTMENT = "ADJUSTMENT"
    MEASUREMENT = "MEASUREMENT"


class Request(BaseModel):
    event: Event
    param: Optional[Dict[str, Any]]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Event: lambda v: str(v),
        }


class Status(BaseModel):
    status: str
    message: Optional[str]


class SleepResponse(BaseModel):
    pass


# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'

# Instructions from servo on what to measure
class MeasureParams(BaseModel):
    metrics: List[str]
    control: Control


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
        return {
            "Authorization": f"Bearer {self.optimizer.token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }

    def api_client(self) -> httpx.AsyncClient:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        headers = {
            "Authorization": f"Bearer {self.optimizer.token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
        return httpx.AsyncClient(base_url=self.optimizer.api_url, headers=self.api_headers)
    
    def api_client_sync(self) -> httpx.Client:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        return httpx.Client(base_url=self.optimizer.api_url, headers=self.api_headers)

    ##
    # TODO: This probably becomes progress_request
    def report_progress(self,
        operation: str, 
        progress: Numeric, 
        started_at: datetime,
        message: Optional[str],
        *,
        time_remaining: Optional[Union[Numeric, Duration]],
        logs: List[str],
    ) -> None:
        def set_if(d: Dict, k: str, v: Any):
            if v is not None: d[k] = v
        
        # Normalize progress to positive percentage
        if progress < 1.0:
            progress = progress * 100
        
        # Calculate runtime
        runtime = Duration(started_at - time.now())

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

        # SERVER: {"progress": 66, "operation": "MEASURE", "nfy": "progress", "tstamp": "2020-07-22 03:38:18"}
        params = dict(
            connector=self.name, 
            operation=operation,
            progress=progress,
            runtime=runtime, 
            runtime_in_seconds=runtime.total_seconds()
        )
        set_if(params, 'time_remaining', str(time_remaining))
        set_if(params, 'time_remaining_in_seconds', str(time_remaining_in_seconds))
        set_if(params, 'message', message)
        set_if(params, 'logs', logs)

        # rsp = request(operation, param, retries=1, backoff=False)