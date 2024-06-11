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

from __future__ import annotations

import copy
import enum
import time
from typing import Annotated, Any, Dict, List, Optional, Union, TYPE_CHECKING

from authlib.integrations.httpx_client import AsyncOAuth2Client
import curlify2
import httpx
import pydantic

import servo
import servo.configuration
import servo.errors
import servo.types
import servo.utilities

if TYPE_CHECKING:
    from pydantic.typing import DictStrAny

USER_AGENT = "github.com/opsani/servox"


class OptimizerStatuses(enum.StrEnum):
    """An enumeration of status types sent by the optimizer."""

    ok = "ok"
    invalid = "invalid"
    unexpected_event = "unexpected-event"
    cancelled = "cancel"


class ServoStatuses(enum.StrEnum):
    """An enumeration of status types sent from the servo."""

    ok = "ok"
    failed = "failed"
    rejected = "rejected"
    aborted = "aborted"
    cancelled = "cancelled"

    def from_error(error: Exception) -> ServoStatuses:
        if isinstance(error, servo.errors.AdjustmentRejectedError):
            return ServoStatuses.rejected
        elif isinstance(error, servo.errors.EventAbortedError):
            return ServoStatuses.aborted
        elif isinstance(error, servo.errors.EventCancelledError):
            return ServoStatuses.cancelled
        else:
            return ServoStatuses.failed


Statuses = Union[OptimizerStatuses, ServoStatuses]


class Reasons(enum.StrEnum):
    success = "success"
    unknown = "unknown"
    unstable = "unstable"


class Events(enum.StrEnum):
    hello = "HELLO"
    whats_next = "WHATS_NEXT"
    describe = "DESCRIPTION"
    measure = "MEASUREMENT"
    adjust = "ADJUSTMENT"
    goodbye = "GOODBYE"


class Commands(enum.StrEnum):
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
            raise ValueError(f"unknown command: {self}")


class Request(pydantic.BaseModel):
    event: Union[Events, str]  # TODO: Needs to be rethought -- used adhoc in some cases
    param: Optional[Dict[str, Any]] = None  # TODO: Switch to a union of supported types
    servo_uid: Union[str, None] = None

    @pydantic.field_serializer("event")
    def event_str(self, event: Events | str) -> str:
        if isinstance(event, Events):
            return str(event)
        return event


class Status(pydantic.BaseModel):
    status: Statuses
    message: Optional[str] = None
    additional_messages: Optional[list[str]] = (
        None  # other lower priority error in exception group
    )
    reason: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    descriptor: Optional[Dict[str, Any]] = None
    metrics: Union[Dict[str, Any], None] = None
    annotations: Union[Dict[str, str], None] = None
    command_uid: Union[str, None] = pydantic.Field(default=None, alias="cmd_uid")

    @classmethod
    def ok(
        cls, message: Optional[str] = None, reason: str = Reasons.success, **kwargs
    ) -> "Status":
        """Return a success (status="ok") status object."""
        return cls(status=ServoStatuses.ok, message=message, reason=reason, **kwargs)

    @classmethod
    def from_error(
        cls, error: servo.errors.BaseError | ExceptionGroup, **kwargs
    ) -> "Status":
        """Return a status object representation from the given error (first if multiple in group)."""
        if isinstance(error, ExceptionGroup):
            servo.logger.warning(
                f"from_error executed on exceptiongroup {error}. May produce undefined behavior"
            )
            status = ServoStatuses.failed
            try:
                error = servo.errors.ServoError.servo_error_from_group(
                    exception_group=error
                )
                if error._additional_errors:
                    additional_messages = [str(e) for e in error._additional_errors]
                    kwargs["additional_messages"] = (
                        kwargs.get("additional_messages", []) + additional_messages
                    )
            except Exception:
                servo.logger.exception(
                    "Failed to derive exceptiongroup reason for api response"
                )
                pass

        reason = getattr(error, "reason", ...)
        status = ServoStatuses.from_error(error)

        return cls(status=status, message=str(error), reason=reason, **kwargs)

    def model_dump(
        self,
        *,
        exclude_unset: bool = True,
        by_alias: bool = True,
        **kwargs,
    ) -> DictStrAny:
        return super().model_dump(
            exclude_unset=exclude_unset, by_alias=by_alias, **kwargs
        )

    model_config = pydantic.ConfigDict(populate_by_name=True)


class SleepResponse(pydantic.BaseModel):
    pass


# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'


def metric_name(v: servo.Metric | str) -> str:
    if isinstance(v, servo.Metric):
        return v.name

    return v


# Instructions from servo on what to measure
class MeasureParams(pydantic.BaseModel):
    metrics: List[
        Annotated[
            str,
            pydantic.Field(validate_default=True),
            pydantic.BeforeValidator(metric_name),
        ]
    ]
    control: servo.types.Control

    @pydantic.field_validator("metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, value) -> List[str]:
        if isinstance(value, dict):
            return list(value.keys())

        return value


class CommandResponse(pydantic.BaseModel):
    command: Commands = pydantic.Field(alias="cmd")
    command_uid: Union[str, None] = pydantic.Field(alias="cmd_uid")
    param: Optional[Union[MeasureParams, Dict[str, Any]]] = (
        None  # TODO: Switch to a union of supported types, remove isinstance check from ServoRunner.measure when done
    )

    @pydantic.field_serializer("command")
    def cmd_str(self, cmd: Commands) -> str:
        return cmd.value


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


def adjustments_to_descriptor(
    adjustments: List[servo.types.Adjustment],
) -> Dict[str, Any]:
    components = {}
    descriptor = {"state": {"application": {"components": components}}}

    for adjustment in adjustments:
        if not adjustment.component_name in components:
            components[adjustment.component_name] = {"settings": {}}

        components[adjustment.component_name]["settings"][adjustment.setting_name] = {
            "value": adjustment.value
        }

    return descriptor


def is_fatal_status_code(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        # Include 404 in status codes to backoff on to reduce noise on CO when workload is not onboarded (OPTSERV-606)
        if error.response.status_code == 404:
            return False
        if error.response.status_code < 500:
            servo.logger.error(
                f"Giving up on non-retryable HTTP status code {error.response.status_code} ({error.response.reason_phrase}) "
            )
            return True
    return False


def user_agent() -> str:
    return f"{USER_AGENT} v{servo.__version__}"


def redacted_to_curl(request: httpx.Request) -> str:
    """Pass through to curlify2.to_curl that redacts the authorization in the headers"""
    if (auth_header := request.headers.get("authorization")) is None:
        return curlify2.to_curl(request)

    req_copy = copy.copy(request)
    req_copy.headers = copy.deepcopy(request.headers)
    if "Bearer" in auth_header:
        req_copy.headers["authorization"] = "Bearer [REDACTED]"
    else:
        req_copy.headers["authorization"] = "[REDACTED]"

    return curlify2.to_curl(req_copy)


def get_api_client_for_optimizer(
    optimizer: servo.configuration.OptimizerTypes,
    settings: servo.configuration.CommonConfiguration,
) -> Union[httpx.AsyncClient, AsyncOAuth2Client]:
    if optimizer.token:
        # NOTE httpx useage docs indicate context manager but author states singleton is fine...
        #   https://github.com/encode/httpx/issues/1042#issuecomment-652951591
        auth_header_value = optimizer.token.get_secret_value()
        if "Bearer" not in auth_header_value:
            auth_header_value = f"Bearer {auth_header_value}"
        return httpx.AsyncClient(
            base_url=str(optimizer.url),
            headers={
                "Authorization": auth_header_value,
                "User-Agent": user_agent(),
                "Content-Type": "application/json",
            },
            proxies=settings.proxies,
            timeout=settings.timeouts,
            verify=settings.ssl_verify,
        )
    elif isinstance(optimizer, servo.configuration.AppdynamicsOptimizer):
        api_client = AsyncOAuth2Client(
            base_url=str(optimizer.url),
            headers={
                "User-Agent": user_agent(),
                "Content-Type": "application/json",
            },
            client_id=optimizer.client_id,
            client_secret=optimizer.client_secret.get_secret_value(),
            token_endpoint=optimizer.token_url,
            grant_type="client_credentials",
            proxies=settings.proxies,
            timeout=settings.timeouts,
            verify=settings.ssl_verify,
        )

        # authlib doesn't check status of token request so we have to do it ourselves
        def raise_for_resp_status(response: httpx.Response):
            response.raise_for_status()
            return response

        api_client.register_compliance_hook(
            "access_token_response", raise_for_resp_status
        )

        # Ideally we would call the following but async is not allowed in __init__
        #   await self.api_client.fetch_token(self.config.optimizer.token_url)
        # Instead we use an ugly hack to trigger the client's autorefresh capabilities
        api_client.token = {
            "expires_at": int(time.time()) - 1,
            "access_token": "_",
        }
        return api_client

    else:
        raise RuntimeError(
            f"Unable to construct api_client from Optimizer type {optimizer.__class__.__name__}"
        )
