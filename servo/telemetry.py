from __future__ import annotations

import enum
import json
import logging
import os
import platform
import pydantic
from typing import Any, Optional, Union

import aiofiles
import asyncio
import backoff
import devtools
import httpx
import pydantic

import servo
import servo.api
from servo.logging import InterceptHandler, logs_path

ONE_MiB = 1048576
DIAGNOSTICS_MAX_RETRIES = 20

DIAGNOSTICS_CHECK_ENDPOINT = "assets/opsani.com/diagnostics-check"
DIAGNOSTICS_OUTPUT_ENDPOINT = "assets/opsani.com/diagnostics-output"

# Intercept backoff decorator logs, only log on giveup
logging.getLogger("diagnostics-backoff").setLevel(logging.ERROR)
logging.getLogger("diagnostics-backoff").addHandler(InterceptHandler())


class DiagnosticStates(str, enum.Enum):
    withhold = "WITHHOLD"
    send = "SEND"
    stop = "STOP"


class Diagnostics(pydantic.BaseModel):
    configmap: Optional[dict[str, Any]]
    logs: Optional[dict[str, Any]]


class Telemetry(pydantic.BaseModel):
    """Class and convenience methods for storage of arbitrary servo metadata"""

    _values: dict[str, str] = pydantic.PrivateAttr(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self["servox.version"] = str(servo.__version__)
        self["servox.platform"] = platform.platform()

        if servo_ns := os.environ.get("POD_NAMESPACE"):
            self["servox.namespace"] = servo_ns

    def __getitem__(self, k: str) -> str:
        return self._values.__getitem__(k)

    def __setitem__(self, k: str, v: str) -> None:
        self._values.__setitem__(k, v)

    def remove(self, key: str) -> None:
        """Safely remove an arbitrary key from telemetry metadata"""
        self._values.pop(key, None)

    @property
    def values(self) -> dict[str, dict[str, str]]:
        # TODO return copy to ensure read only?
        return self._values


class DiagnosticsHandler(servo.logging.Mixin, servo.api.Mixin):

    servo: servo.Servo = None
    _running: bool = False

    def __init__(self, servo: servo.Servo) -> None:  # noqa: D10
        self.servo = servo

    @property
    def api_client_options(self) -> dict[str, Any]:
        # Adopt the servo config for driving the API mixin
        return self.servo.api_client_options

    async def diagnostics_check(self) -> None:

        self._running = True

        while self._running:
            try:
                self.logger.trace("Polling for diagnostics request")
                request = await self._diagnostics_api(
                    method="GET",
                    endpoint=DIAGNOSTICS_CHECK_ENDPOINT,
                    output_model=DiagnosticStates,
                )

                if request == DiagnosticStates.withhold:
                    self.logger.trace("Withholding diagnostics")

                elif request == DiagnosticStates.send:
                    self.logger.info(f"Diagnostics requested, gathering and sending")
                    diagnostic_data = await self._get_diagnostics()

                    await self._diagnostics_api(
                        method="PUT",
                        endpoint=DIAGNOSTICS_OUTPUT_ENDPOINT,
                        output_model=servo.api.Status,
                        json=diagnostic_data.dict(),
                    )

                    # Reset diagnostics check state to withhold
                    reset_state = DiagnosticStates.withhold
                    await self._diagnostics_api(
                        method="PUT",
                        endpoint=DIAGNOSTICS_CHECK_ENDPOINT,
                        output_model=servo.api.Status,
                        json=reset_state,
                    )

                elif request == DiagnosticStates.stop:
                    self.logger.info(
                        f"Received request to disable polling for diagnostics"
                    )
                    self.servo.config.no_diagnostics = True
                    self._running = False
                else:
                    raise

                await asyncio.sleep(60)

            except Exception:
                self.logger.exception(
                    f"Diagnostics check failed with unrecoverable error"
                )  # exception logger logs the exception object
                self._running = False

    async def _get_diagnostics(self) -> Diagnostics:

        async with aiofiles.open(servo.logging.logs_path, "r") as log_file:
            logs = await log_file.read()

        # Strip emoji from logs :(
        raw_logs = logs.encode("ascii", "ignore").decode()

        # Limit + truncate per 1MiB /assets limit, allowing ample room for configmap
        log_data_lines = filter(None, raw_logs[-ONE_MiB - 10000 :].split("\n")[1:])
        log_dict = {}

        for line in log_data_lines:
            # Handle rare multi-line logs e.g. from self.tuning_container.resources
            try:
                time, msg = line.split("|", 1)
                log_dict[time.strip()] = msg.strip()
            except:
                log_dict[list(log_dict.keys())[-1]] += line

        # TODO: Re-evaluate roundtripping through JSON required to produce primitives
        config_dict = self.servo.config.json(exclude_unset=True, exclude_none=True)
        config_data = json.loads(config_dict)

        return Diagnostics(configmap=config_data, logs=log_dict)

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.current_servo()
        and servo.current_servo().config.settings.backoff.max_time(),
        max_tries=lambda: DIAGNOSTICS_MAX_RETRIES,
        logger="diagnostics-backoff",
        on_giveup=lambda x: asyncio.current_task().cancel(),
    )
    async def _diagnostics_api(
        self,
        method: str,
        endpoint: str,
        output_model: pydantic.BaseModel,
        json: Optional[dict] = None,
    ) -> Union[DiagnosticStates, servo.api.Status]:

        async with self.api_client() as client:
            self.logger.trace(f"{method} diagnostic request")
            try:
                response = await client.request(
                    method=method, url=endpoint, json=dict(data=json)
                )
                response.raise_for_status()
                response_json = response.json()

                # Handle /diagnostics-check retrieval
                if "data" in response_json:
                    response_json = response_json["data"]

                self.logger.trace(
                    f"{method} diagnostic request response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )
                self.logger.trace(servo.api._redacted_to_curl(response.request))
                try:
                    return pydantic.parse_obj_as(output_model, response_json)
                except pydantic.ValidationError as error:
                    # Should not raise due to improperly set diagnostic states
                    self.logger.exception(
                        f"Malformed diagnostic {method} response", level_id="DEBUG"
                    )
                    return DiagnosticStates.withhold

            except httpx.HTTPError as error:
                if error.response.status_code < 500:
                    self.logger.debug(
                        f"Giving up on non-retryable HTTP status code {error.response.status_code} ({error.response.reason_phrase}) for url: {error.request.url}"
                    )
                    return DiagnosticStates.withhold
                else:
                    self.logger.trace(servo.api._redacted_to_curl(error.request))
                    raise
