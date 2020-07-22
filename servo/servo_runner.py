import asyncio
import signal
import sys
import time
from enum import Enum
from logging import Logger
from typing import Any, Dict, List, Optional, Union

import backoff
import httpx
import loguru

from devtools import pformat
from pydantic import BaseModel, Field, parse_obj_as

from servo import api
from servo.assembly import BaseServoConfiguration
from servo.configuration import Optimizer
from servo.logging import ProgressHandler
from servo.servo import Events, Servo
from servo.types import Control, Description, Measurement


# TODO: Review and expand all the error classes
class ConnectorError(Exception):
    """Exception indicating that a connector failed
    """

    def __init__(self, *args, status="failed", reason="unknown"):
        self.status = status
        self.reason = reason
        super().__init__(*args)


class ServoRunner:
    servo: Servo

    def __init__(self, servo: Servo) -> None:
        self.servo = servo
        super().__init__()

    @property
    def optimizer(self) -> Optimizer:
        return self.servo.optimizer

    @property
    def configuration(self) -> BaseServoConfiguration:
        return self.servo.configuration

    @property
    def logger(self) -> Logger:
        return self.servo.logger

    async def describe(self) -> Description:
        self.logger.info("Describing...")

        aggregate_description = Description.construct()
        results: List[EventResult] = await self.servo.dispatch_event(Events.DESCRIBE)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        return aggregate_description

    async def measure(self, param: api.MeasureParams) -> Measurement:
        self.logger.info(f"Measuring... [metrics={', '.join(param.metrics)}]")
        self.logger.trace(pformat(param))

        aggregate_measurement = Measurement.construct()
        results: List[EventResult] = await self.servo.dispatch_event(
            Events.MEASURE, metrics=param.metrics, control=param.control
        )
        for result in results:
            measurement = result.value
            aggregate_measurement.readings.extend(measurement.readings)
            aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    async def adjust(self, param) -> dict:
        self.logger.info("Adjusting...")
        self.logger.trace(pformat(param))

        results: List[EventResult] = await self.servo.dispatch_event(Events.ADJUST, param)
        for result in results:
            # TODO: Should be modeled
            adjustment = result.value
            status = adjustment.get("status", "undefined")

            if status == "ok":
                self.logger.info(f"{result.connector.name} - Adjustment completed")
                return adjustment
            else:
                raise ConnectorError(
                    'Adjustment driver failed with status "{}" and message:\n{}'.format(
                        status, str(adjustment.get("message", "undefined"))
                    ),
                    status=status,
                    reason=adjustment.get("reason", "undefined"),
                )

        # TODO: Model a response class
        return {}

    @backoff.on_exception(backoff.expo, (httpx.HTTPError), max_time=180, max_tries=12)
    async def post_event(self, event: api.Event, param) -> Union[api.CommandResponse, api.Status]:
        event_request = api.Request(event=event, param=param)
        async with self.servo.api_client() as client:
            try:
                response = await client.post("servo", data=event_request.json())
                response.raise_for_status()
            except httpx.HTTPError as error:
                self.logger.exception(
                    f"HTTP error encountered while posting {event.value} event"
                )
                self.logger.trace(pformat(event_request))
                raise error

        return parse_obj_as(Union[api.CommandResponse, api.Status], response.json())

    async def exec_command(self):
        cmd_response = await self.post_event(api.Event.WHATS_NEXT, None)
        self.logger.debug(f"What's Next? => {cmd_response.command}")
        self.logger.trace(pformat(cmd_response))

        try:
            if cmd_response.command == api.Command.DESCRIBE:
                description = await self.describe()
                self.logger.info(
                    f"Described: {len(description.components)} components, {len(description.metrics)} metrics"
                )
                self.logger.trace(pformat(description))
                param = dict(descriptor=description.opsani_dict(), status="ok")
                await self.post_event(api.Event.DESCRIPTION, param)

            elif cmd_response.command == api.Command.MEASURE:
                measurement = await self.measure(cmd_response.param)
                self.logger.info(
                    f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
                )
                self.logger.trace(pformat(measurement))
                param = measurement.opsani_dict()
                await self.post_event(api.Event.MEASUREMENT, param)

            elif cmd_response.command == api.Command.ADJUST:
                # # TODO: This needs to be modeled
                # oc"{'cmd': 'ADJUST', 'param': {'state': {'application': {'components': {'web': {'settings': {'cpu': {'value': 0.225}, 'mem': {'value': 0.1}}}}}}, 'control': {}}}"

                # TODO: Why do we do this nonsense??
                # create a new dict based on p['state'] (with its top level key
                # 'application') which also includes a top-level 'control' key, and
                # pass this to adjust()
                new_dict = cmd_response.param["state"].copy()
                new_dict["control"] = cmd_response.param.get("control", {})
                adjustment = await self.adjust(new_dict)

                # TODO: What works like this and why?
                if (
                    "state" not in adjustment
                ):  # if driver didn't return state, assume it is what was requested
                    adjustment["state"] = cmd_response.param["state"]

                components_dict = adjustment["state"]["application"]["components"]
                components_count = len(components_dict)
                settings_count = sum(
                    len(components_dict[component]["settings"])
                    for component in components_dict
                )
                self.logger.info(
                    f"Adjusted: {components_count} components, {settings_count} settings"
                )

                await self.post_event(api.Event.ADJUSTMENT, adjustment)

            elif cmd_response.command == api.Command.SLEEP:
                if (
                    not self.interactive
                ):  # ignore sleep request when interactive - let user decide
                    # TODO: Model this
                    duration = int(cmd_response.param.get("duration", 120))
                    self.logger.info(f"Sleeping {duration} sec.")
                    await asyncio.sleep(duration)

            else:
                raise ValueError(f"Unknown command '{cmd_response.command.value}'")

        except Exception as error:
            self.logger.exception(f"{cmd_response.command} command failed!")
            param = dict(status="failed", message=_exc_format(error))
            self.shutdown()
            await self.post_event(_event_for_command(cmd_response.command), param)

    async def main(self) -> None:
        self.logger.info(
            f"Servo started with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.base_url}]"
        )
        self.logger.info("Broadcasting startup event...")
        self.servo.startup()

        self.logger.info("Saying HELLO.", end=" ")
        await self.post_event(api.Event.HELLO, dict(agent=api.USER_AGENT))

        while True:
            try:
                await self.exec_command()
            except Exception:
                self.logger.exception("Exception encountered while executing command")
    
    async def shutdown(self, loop, signal=None):
        if signal:
            self.logger.info(f"Received exit signal {signal.name}...")

        try:
            reason = signal.name if signal else 'shutdown'
            await self.post_event(api.Event.GOODBYE, dict(reason=reason))
        except Exception as e:
            self.logger.exception(
                f"Exception occurred during GOODBYE request: {e}"
            )
        
        self.logger.info("Dispatching shutdown event...")
        await self.servo.shutdown()
        
        tasks = [t for t in asyncio.all_tasks() if t is not
                asyncio.current_task()]

        [task.cancel() for task in tasks]

        self.logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

        loop.stop()
    
    def handle_exception(self, loop, context):
        # context["message"] will always be there; but context["exception"] may not
        msg = context.get("exception", context["message"])
        self.logger.exception(f"Caught exception: {msg}")
        self.logger.info("Shutting down...")
        asyncio.create_task(self.shutdown(loop))

    def run(self) -> None:
        # Setup logging
        handler = ProgressHandler(self.servo)
        loguru.logger.add(handler)

        # Setup async event loop
        loop = asyncio.get_event_loop()
        
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(self.shutdown(loop, signal=s)))
        
        loop.set_exception_handler(self.handle_exception)

        try:
            loop.create_task(self.main())
            loop.run_forever()
        finally:
            loop.close()
            self.logger.info("Servo shutdown complete.")


def _event_for_command(command: api.Command) -> Optional[api.Event]:
    if cmd_response.command == api.Command.DESCRIBE:
        return api.Event.DESCRIPTION
    elif cmd_response.command == api.Command.MEASURE:
        return api.Event.MEASUREMENT
    elif cmd_response.command == api.Command.ADJUST:
        return api.Event.ADJUSTMENT
    else:
        return None


def _exc_format(e):
    if type(e) is Exception:  # if it's just an Exception
        return str(e)  # print only the message but not the type
    return "{type(e).__name__}: {e}"
