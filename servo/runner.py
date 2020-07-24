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
import typer

from devtools import pformat
from pydantic import BaseModel, Field, parse_obj_as

from servo import api
from servo.assembly import Assembly, BaseServoConfiguration
from servo.configuration import Optimizer
from servo.errors import ConnectorError
from servo.logging import ProgressHandler
from servo.servo import Events, Servo
from servo.types import Control, Description, Measurement
from servo.utilities import commandify


class Runner(api.Mixin):
    assembly: Assembly

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        super().__init__()

    @property
    def optimizer(self) -> Optimizer:
        return self.servo.optimizer
    
    @property
    def servo(self) -> Servo:
        return self.assembly.servo

    @property
    def configuration(self) -> BaseServoConfiguration:
        return self.servo.configuration

    @property
    def logger(self) -> Logger:
        return self.servo.logger
    
    def display_banner(self) -> None:
        banner = (
            "   _____                      _  __\n"
            "  / ___/___  ______   ______ | |/ /\n"
            "  \__ \/ _ \/ ___/ | / / __ \|   /\n"
            " ___/ /  __/ /   | |/ / /_/ /   |\n"
            "/____/\___/_/    |___/\____/_/|_|"
        )
        typer.secho(banner, fg=typer.colors.BRIGHT_BLUE, bold=True)
                
        name_st = typer.style("name", fg=typer.colors.CYAN, bold=False)
        version_st = typer.style("version", fg=typer.colors.WHITE, bold=True)
        types = Assembly.all_connector_types()
        types.remove(Servo)
        
        names = []
        for c in types:
            name = typer.style(commandify(c.__default_name__), fg=typer.colors.CYAN, bold=False)
            version = typer.style(str(c.version), fg=typer.colors.WHITE, bold=True)
            names.append(f"{name}-{version}")
        version = typer.style(f"v{Servo.version}", fg=typer.colors.WHITE, bold=True)
        codename = typer.style("the awakening", fg=typer.colors.MAGENTA, bold=False)
        initialized = typer.style("initialized", fg=typer.colors.BRIGHT_GREEN, bold=True)        
        
        typer.secho(f"{version} \"{codename}\" {initialized}")
        typer.secho()
        typer.secho(f"connectors:  {', '.join(sorted(names))}")
        typer.secho(f"config file: {typer.style(str(self.assembly.config_file), bold=True, fg=typer.colors.YELLOW)}")
        id = typer.style(self.optimizer.id, bold=True, fg=typer.colors.WHITE)        
        typer.secho(f"optimizer:   {id}")
        if self.optimizer.base_url != "https://api.opsani.com/":
            base_url = typer.style(f"{self.optimizer.base_url}", bold=True, fg=typer.colors.RED)
            typer.secho(f"base url: {base_url}")
        typer.secho()

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

    async def exec_command(self):
        cmd_response = await self._post_event(api.Event.WHATS_NEXT, None)
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
                await self._post_event(api.Event.DESCRIPTION, param)

            elif cmd_response.command == api.Command.MEASURE:
                measurement = await self.measure(cmd_response.param)
                self.logger.info(
                    f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
                )
                self.logger.trace(pformat(measurement))
                param = measurement.opsani_dict()
                await self._post_event(api.Event.MEASUREMENT, param)

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

                await self._post_event(api.Event.ADJUSTMENT, adjustment)

            elif cmd_response.command == api.Command.SLEEP:
                    # TODO: Model this
                    duration = int(cmd_response.param.get("duration", 120))
                    self.logger.info(f"Sleeping {duration} sec.")
                    await asyncio.sleep(duration)

            else:
                raise ValueError(f"Unknown command '{cmd_response.command.value}'")

        except Exception as error:
            self.logger.exception(f"{cmd_response.command} command failed!")
            param = dict(status="failed", message=_exc_format(error))
            await self.shutdown(asyncio.get_event_loop())
            await self._post_event(cmd_response.command.response_event, param)

    async def main(self) -> None:
        self.logger.info(
            f"Servo started with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.base_url}]"
        )
        self.logger.info("Broadcasting startup event...")
        self.servo.startup()

        self.logger.info("Saying HELLO.", end=" ")
        await self._post_event(api.Event.HELLO, dict(agent=api.USER_AGENT))

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
            await self._post_event(api.Event.GOODBYE, dict(reason=reason))
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
        self.display_banner()

        # Setup async event loop
        loop = asyncio.get_event_loop()

        # Setup logging
        handler = ProgressHandler(self.servo)
        loop.create_task(handler.run())
        loguru.logger.add(handler)        
        
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

def _exc_format(e):
    if type(e) is Exception:  # if it's just an Exception
        return str(e)  # print only the message but not the type
    return "{type(e).__name__}: {e}"
