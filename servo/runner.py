from __future__ import annotations
import asyncio
import signal
from typing import Any, Dict, List, Optional, Union

import backoff
import httpx
import typer

from devtools import pformat
from pydantic import BaseModel, Field, parse_obj_as

import servo
from servo import api
from servo.api import descriptor_to_adjustments
from servo.assembly import Assembly
from servo.configuration import BaseAssemblyConfiguration, Optimizer, ServoConfiguration
from servo.errors import ConnectorError
from servo.events import EventResult
from servo.logging import ProgressHandler, logger
from servo.servo import Events, Servo
from servo.types import Adjustment, Control, Duration, Description, Measurement
from servo.utilities import commandify, value_for_key_path


DEFAULT_CONTEXT = "__default__"


class BackoffConfig:
    """BackoffConfig provides callables for runtime configuration of backoff decorators.

    This implementation relies upon the framework managed context variables to determine
    which servo is running and retrieve the configuration.
    """

    @staticmethod
    def max_time(context: str = DEFAULT_CONTEXT) -> Optional[int]:
        if servo_config := Servo.current().config.servo:
            if backoff_config := servo_config.backoff.get(context, None):
                if max_time := backoff_config.max_time:
                    return max_time.total_seconds()

        # fallback to default
        if max_time := BackoffConfig.max_time():
            return max_time

        raise AssertionError("max_time default should never be None")


    @staticmethod
    def max_tries(context: str = DEFAULT_CONTEXT) -> Optional[int]:
        if servo_config := Servo.current().config.servo:
            if backoff_config := servo_config.backoff.get(context, None):
                return backoff_config.max_tries

        # fallback to default
        return BackoffConfig.max_tries()


class Runner(servo.logging.Mixin, servo.api.Mixin):
    assembly: Assembly
    connected: bool = False

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly

        # initialize default servo options if not configured
        if self.config.servo is None:
            self.config.servo = ServoConfiguration()

        super().__init__()

    @property
    def optimizer(self) -> Optimizer:
        return self.servo.optimizer

    @property
    def servo(self) -> Servo:
        return self.assembly.servo

    @property
    def config(self) -> BaseAssemblyConfiguration:
        return self.servo.config

    @property
    def api_client_options(self) -> Dict[str, Any]:
        # FIXME: Support for proxies. This is messy. Needs to be cleaned up.
        # We have unnatural layering because proxies is on config but api is standalone.
        return self.servo.api_client_options

    def display_banner(self) -> None:
        banner = (
            "   _____                      _  __\n"
            "  / ___/___  ______   ______ | |/ /\n"
            "  \__ \/ _ \/ ___/ | / / __ \|   /\n"
            " ___/ /  __/ /   | |/ / /_/ /   |\n"
            "/____/\___/_/    |___/\____/_/|_|"
        )
        typer.secho(banner, fg=typer.colors.BRIGHT_BLUE, bold=True)

        types = Assembly.all_connector_types()
        types.remove(Servo)

        names = []
        for c in types:
            name = typer.style(commandify(c.__default_name__), fg=typer.colors.CYAN, bold=False)
            version = typer.style(str(c.version), fg=typer.colors.WHITE, bold=True)
            names.append(f"{name}-{version}")

        version = typer.style(f"v{servo.__version__}", fg=typer.colors.WHITE, bold=True)
        codename = typer.style(servo.__codename__, fg=typer.colors.MAGENTA, bold=False)
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
        if self.config.servo.proxies:
            proxies = typer.style(f"{pformat(self.config.servo.proxies)}", bold=True, fg=typer.colors.CYAN)
            typer.secho(f"proxies: {proxies}")
        typer.secho()

    async def describe(self) -> Description:
        logger.info("Describing...")

        aggregate_description = Description.construct()
        results: List[EventResult] = await self.servo.dispatch_event(Events.DESCRIBE)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        return aggregate_description

    async def measure(self, param: api.MeasureParams) -> Measurement:
        logger.info(f"Measuring... [metrics={', '.join(param.metrics)}]")
        logger.trace(pformat(param))

        aggregate_measurement = Measurement.construct()
        results: List[EventResult] = await self.servo.dispatch_event(
            Events.MEASURE, metrics=param.metrics, control=param.control
        )
        for result in results:
            measurement = result.value
            aggregate_measurement.readings.extend(measurement.readings)
            aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    async def adjust(self, adjustments: List[Adjustment], control: Control) -> Description:
        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        logger.info(f"Adjusting... {summary}")
        logger.trace(pformat(adjustments))
        
        aggregate_description = Description.construct()
        results = await self.servo.dispatch_event(Events.ADJUST, adjustments)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        logger.info(f"Adjustment completed {summary}")
        return aggregate_description

    # backoff and retry for an hour on transient request failures
    @backoff.on_exception(backoff.expo, httpx.HTTPError, max_time=BackoffConfig.max_time)
    async def exec_command(self) -> servo.api.Status:
        cmd_response = await self._post_event(api.Event.WHATS_NEXT, None)
        logger.info(f"What's Next? => {cmd_response.command}")
        logger.trace(pformat(cmd_response))

        if cmd_response.command == api.Command.DESCRIBE:
            description = await self.describe()
            logger.info(
                f"Described: {len(description.components)} components, {len(description.metrics)} metrics"
            )
            logger.trace(pformat(description))

            param = dict(descriptor=description.__opsani_repr__(), status="ok")
            return await self._post_event(api.Event.DESCRIPTION, param)

        elif cmd_response.command == api.Command.MEASURE:
            measurement = await self.measure(cmd_response.param)
            logger.info(
                f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
            )
            logger.trace(pformat(measurement))
            param = measurement.__opsani_repr__()
            return await self._post_event(api.Event.MEASUREMENT, param)

        elif cmd_response.command == api.Command.ADJUST:
            adjustments = descriptor_to_adjustments(cmd_response.param["state"])
            control = Control(**cmd_response.param.get("control", {}))
            description = await self.adjust(adjustments, control)

            reply = { "status": "ok", "state": description.__opsani_repr__() }

            components_count = len(description.components)
            settings_count = sum(
                len(component.settings)
                for component in description.components
            )
            logger.info(
                f"Adjusted: {components_count} components, {settings_count} settings"
            )

            return await self._post_event(api.Event.ADJUSTMENT, reply)

        elif cmd_response.command == api.Command.SLEEP:
                # TODO: Model this
                duration = Duration(cmd_response.param.get("duration", 120))
                status = value_for_key_path(cmd_response.param, "data.status", None)
                reason = value_for_key_path(cmd_response.param, "data.reason", "unknown reason")
                msg = f"{status}: {reason}" if status else f"{reason}"
                logger.info(f"Sleeping for {duration} ({msg}).")
                await asyncio.sleep(duration.total_seconds())

                # Return a status so we have a simple API contract
                return servo.api.Status(status="slept", message=msg)
        else:
            raise ValueError(f"Unknown command '{cmd_response.command.value}'")

    async def main(self) -> None:
        # Main run loop for processing commands from the optimizer
        async def main_loop() -> None:
            while True:
                try:
                    status = await self.exec_command()
                    if status.status == servo.api.UNEXPECTED_EVENT:
                        logger.warning(f"server reported unexpected event: {status.reason}")
                except Exception as error:
                    logger.exception(f"failed with unrecoverable error: {error}")
                    raise error

        def handle_progress_exception(error: Exception) -> None:
            # Restart the main event loop if we get out of sync with the server
            if isinstance(error, servo.api.UnexpectedEventError):
                logger.error("servo has lost synchronization with the optimizer: restarting operations")

                tasks = [t for t in asyncio.all_tasks() if t is not
                    asyncio.current_task()]
                logger.info(f"Cancelling {len(tasks)} outstanding tasks")
                [task.cancel() for task in tasks]

                # Restart a fresh main loop
                asyncio.create_task(main_loop(), name="main loop")

        # Setup logging
        self.progress_handler = servo.logging.ProgressHandler(self.servo.report_progress, self.logger.warning, handle_progress_exception)
        self.logger.add(self.progress_handler.sink, catch=True)

        logger.info(
            f"Servo started with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.url or self.optimizer.base_url}]"
        )

        async def giveup(details) -> None:
            loop = asyncio.get_event_loop()
            logger.critical("retries exhausted, giving up")
            asyncio.create_task(self.shutdown(loop))

        try:
            @backoff.on_exception(
                backoff.expo,
                httpx.HTTPError,
                max_time=lambda: BackoffConfig.max_time("connect"),
                on_giveup=giveup
            )
            async def connect() -> None:
                logger.info("Saying HELLO.", end=" ")
                await self._post_event(api.Event.HELLO, dict(agent=api.USER_AGENT))
                self.connected = True

            logger.info("Dispatching startup event...")
            await self.servo.startup()

            logger.info(f"Connecting to Opsani Optimizer @ {self.optimizer.api_url}...")
            await connect()
        except:
            pass

        await asyncio.create_task(main_loop(), name="main loop")

    async def shutdown(self, loop, signal=None):
        if signal:
            logger.info(f"Received exit signal {signal.name}...")

        try:
            if self.connected:
                reason = signal.name if signal else 'shutdown'
                await self._post_event(api.Event.GOODBYE, dict(reason=reason))
        except Exception:
            logger.exception(f"Exception occurred during GOODBYE request")

        logger.info("Dispatching shutdown event...")
        await self.servo.shutdown()

        tasks = [t for t in asyncio.all_tasks() if t is not
                asyncio.current_task()]

        [task.cancel() for task in tasks]

        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Servo shutdown complete.")
        await logger.complete()

        loop.stop()

    def handle_exception(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        logger.error(f"asyncio exception handler triggered with context: {context}")

        logger.critical("Shutting down due to unhandled exception in asyncio event loop...")
        asyncio.create_task(self.shutdown(loop))

    def run(self) -> None:
        self.display_banner()

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

