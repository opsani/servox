from __future__ import annotations

import abc
import os
import platform
from typing import Dict
import pydantic
import enum
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncio
import aiofiles
import backoff
import devtools
import httpx
import pydantic
from semver import b

import servo
import servo.api
from servo.logging import logs_path
from servo.types import JSON_FORMAT


ONE_MiB = 1048576
DIAGNOSTICS_MAX_RETRIES = 20

# Intercept backoff decorator logs
logging.getLogger('diagnostics-backoff').setLevel(logging.ERROR)

class DiagnosticStates(str, enum.Enum):
    withhold = "WITHHOLD"
    send = "SEND"
    stop = "STOP"

class Diagnostics(pydantic.BaseModel):
    configmap: Optional[Dict[str, Any]]
    logs: Optional[Dict[str, Any]]

class Telemetry(pydantic.BaseModel):
    """Class and convenience methods for storage of arbitrary servo metadata
    """

    _values: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)

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
    def values(self) -> Dict[str, Dict[str, str]]:
        # TODO return copy to ensure read only?
        return self._values

class DiagnosticsHandler(servo.logging.Mixin, servo.api.Mixin):

    _servo: servo.Servo = pydantic.PrivateAttr(None)
    _running: bool = pydantic.PrivateAttr(False)

    def __init__(self, servo: servo.Servo) -> None: # noqa: D10
        self.servo = servo

    @property
    def api_client_options(self) -> Dict[str, Any]:
        # Adopt the servo config for driving the API mixin
        return self.servo.api_client_options


    async def diagnostics_check(self):

        self._running = True

        while self._running:
            try:
                self.logger.trace("Polling for diagnostics request")

                request = await self._diagnostics_request()

                if request == DiagnosticStates.withhold:
                    self.logger.trace("Withholding diagnostics")

                elif request == DiagnosticStates.send:
                    self.logger.info(f"Diagnostics requested, gathering and sending")
                    diagnostic_data = await self._get_diagnostics()

                    await self._put_diagnostics(diagnostic_data)
                    await self._reset_diagnostics()

                elif request == DiagnosticStates.stop:
                    self.logger.info(f"Received request to disable polling for diagnostics")
                    asyncio.current_task().cancel()
                else:
                    raise

                await asyncio.sleep(60)

            except Exception:
                self.logger.exception(f"Diagnostics check failed with unrecoverable error") # exception logger logs the exception object
                self._running = False

    async def _get_diagnostics(self) -> Diagnostics:

        async with aiofiles.open(logs_path, 'r') as log_file:
            logs = await log_file.read()

        # Strip emoji from logs :(
        raw_logs = logs.encode("ascii", "ignore").decode()

        # Limit + truncate per 1MiB /assets limit
        log_data_lines = list(filter(None, raw_logs[-ONE_MiB:].split("\n")[1:]))

        log_dict = {}

        for line in log_data_lines:
            # Handle rare multi-line logs e.g. from self.tuning_container.resources
            try:
                time, msg = line.split('|', 1)
                log_dict[time.strip()] = msg.strip()
            except:
                log_dict[list(log_dict.keys())[-1]] += line

        config_dict = self.servo.config.json(exclude_unset=True, exclude_none=True)
        config_data = json.loads(config_dict)

        return Diagnostics(configmap=config_data, logs=log_dict)

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.current_servo() and servo.current_servo().config.settings.backoff.max_time(),
        max_tries=DIAGNOSTICS_MAX_RETRIES,
        logger='diagnostics-backoff',
        on_giveup=lambda x: asyncio.current_task().cancel()
    )
    async def _diagnostics_request(self) -> DiagnosticStates:
        async with self.api_client() as client:
            self.logger.trace(f"GET diagnostic request")
            try:
                # response = await client.get("https://www.fsjdiofjdsiofjisdofjiosdjfosd.com")
                response = await client.get("assets/opsani.com/diagnostics-check")
                response.raise_for_status()
                response_json = response.json()['data']
                self.logger.trace(
                    f"GET diagnostic request response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )
                self.logger.trace(servo.api._redacted_to_curl(response.request))
                try:
                    return pydantic.parse_obj_as(
                        DiagnosticStates, response_json
                    )
                except pydantic.ValidationError as error:
                    # Should not raise due to improperly set diagnostic states
                    self.logger.trace(f"Improperly set diagnostic state {error}")
                    return DiagnosticStates.withhold

            except httpx.HTTPError as error:
                self.logger.trace(servo.api._redacted_to_curl(error.request))
                raise

    async def _put_diagnostics(self, diagnostic_data: Diagnostics) -> servo.api.Status:

        async with self.api_client() as client:
            self.logger.trace(f"POST diagnostic data: {devtools.pformat(diagnostic_data.json())}")

            # Push into data key
            data = json.dumps(
                dict(data=diagnostic_data.dict())
                )
            return await self._diagnostics_api(client, "assets/opsani.com/diagnostics-output", data)


    async def _reset_diagnostics(self) -> servo.api.Status:

        async with self.api_client() as client:

            # Push into data key
            reset_request = json.dumps(
                dict(data=DiagnosticStates.withhold)
                )
            return await self._diagnostics_api(client, "assets/opsani.com/diagnostics-check", reset_request)


    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.current_servo() and servo.current_servo().config.settings.backoff.max_time(),
        max_tries=lambda: DIAGNOSTICS_MAX_RETRIES,
        logger='diagnostics-backoff',
        on_giveup=lambda x: asyncio.current_task().cancel()
    )
    async def _diagnostics_api(self, client: httpx.AsyncClient, endpoint: str, data: Optional[JSON_FORMAT]=None) -> servo.api.Status:

            self.logger.trace(f"Diagnostics : {devtools.pformat(data)}")
            try:
                response = await client.put(endpoint, data=data)
                response.raise_for_status()
                response_json = response.json()
                self.logger.trace(
                    f"Diagnostics API response ({response.status_code} {response.reason_phrase}): {devtools.pformat(response_json)}"
                )
                self.logger.trace(servo.api._redacted_to_curl(response.request))

                return pydantic.parse_obj_as(
                    servo.api.Status, response_json
                )

            except pydantic.ValidationError as error:
                # Should not raise due to improperly set diagnostic states
                self.logger.trace(f"Improperly set diagnostic state {error}")

            except httpx.HTTPError as error:
                self.logger.trace(servo.api._redacted_to_curl(error.request))
                raise
