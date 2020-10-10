from __future__ import annotations

import asyncio
import signal
from typing import Any, Dict, List, Optional

import backoff
import devtools
import httpx
import typer

import servo as servox
import servo.configuration
import servo.api
import servo.utilities.key_paths
import servo.utilities.strings
from servo.types import Adjustment, Control, Description, Duration, Measurement

class Runner(servo.logging.Mixin, servo.api.Mixin):
    assembly: servo.Assembly
    connected: bool = False

    def __init__(self, assembly: servo.Assembly) -> None: # noqa: D107
        self.assembly = assembly

        # initialize default servo options if not configured
        # if self.config.servo is None:
        #     self.config.servo = servo.ServoConfiguration()

        super().__init__()

    @property
    def optimizer(self) -> servo.Optimizer:
        return self.servo.config.optimizer

    @property
    def servo(self) -> servo.Servo:
        # return self.assembly.servo
        return servo.servo.Servo.current()

    @property
    def config(self) -> servo.BaseServoConfiguration:
        return self.servo.config

    @property
    def api_client_options(self) -> Dict[str, Any]:
        # Adopt the servo config for driving the API mixin
        return self.servo.api_client_options

    def display_banner(self) -> None:
        banner = "\n".join([
            r"   _____                      _  __",
            r"  / ___/___  ______   ______ | |/ /",
            r"  \__ \/ _ \/ ___/ | / / __ \|   /",
            r" ___/ /  __/ /   | |/ / /_/ /   |",
            r"/____/\___/_/    |___/\____/_/|_|",
        ])
        typer.secho(banner, fg=typer.colors.BRIGHT_BLUE, bold=True)
        types = servo.Assembly.all_connector_types()
        types.remove(servo.Servo)

        names = []
        for c in types:
            name = typer.style(
                servo.utilities.strings.commandify(c.__default_name__), fg=typer.colors.CYAN, bold=False
            )
            version = typer.style(str(c.version), fg=typer.colors.WHITE, bold=True)
            names.append(f"{name}-{version}")

        version = typer.style(f"v{servo.__version__}", fg=typer.colors.WHITE, bold=True)
        codename = typer.style(servo.__cryptonym__, fg=typer.colors.MAGENTA, bold=False)
        initialized = typer.style(
            "initialized", fg=typer.colors.BRIGHT_GREEN, bold=True
        )

        typer.secho(f'{version} "{codename}" {initialized}')
        typer.secho()
        typer.secho(f"connectors:  {', '.join(sorted(names))}")
        typer.secho(
            f"config file: {typer.style(str(self.assembly.config_file), bold=True, fg=typer.colors.YELLOW)}"
        )
        id = typer.style(self.optimizer.id, bold=True, fg=typer.colors.WHITE)
        typer.secho(f"optimizer:   {id}")
        if self.optimizer.base_url != "https://api.opsani.com/":
            base_url = typer.style(
                f"{self.optimizer.base_url}", bold=True, fg=typer.colors.RED
            )
            typer.secho(f"base url: {base_url}")
        # if self.config.servo.proxies:
        #     proxies = typer.style(
        #         f"{devtools.pformat(self.config.servo.proxies)}",
        #         bold=True,
        #         fg=typer.colors.CYAN,
        #     )
        #     typer.secho(f"proxies: {proxies}")
        typer.secho()

    async def describe(self) -> Description:
        self.logger.info("Describing...")

        aggregate_description = Description.construct()
        results: List[servo.EventResult] = await self.servo.dispatch_event(servo.Events.DESCRIBE)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        return aggregate_description

    async def measure(self, param: servo.MeasureParams) -> Measurement:
        servo.logger.info(f"Measuring... [metrics={', '.join(param.metrics)}]")
        servo.logger.trace(devtools.pformat(param))

        aggregate_measurement = Measurement.construct()
        results: List[servo.EventResult] = await self.servo.dispatch_event(
            servo.Events.MEASURE, metrics=param.metrics, control=param.control
        )
        for result in results:
            measurement = result.value
            aggregate_measurement.readings.extend(measurement.readings)
            aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    async def adjust(
        self, adjustments: List[Adjustment], control: Control
    ) -> Description:
        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        self.logger.info(f"Adjusting... {summary}")
        self.logger.trace(devtools.pformat(adjustments))

        aggregate_description = Description.construct()
        results = await self.servo.dispatch_event(servo.Events.ADJUST, adjustments)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        self.logger.info(f"Adjustment completed {summary}")
        return aggregate_description

    @backoff.on_exception(
        backoff.expo,
        httpx.HTTPError,
        max_time=lambda: servo.Servo.current().config.servo.backoff.max_time(),
        max_tries=lambda: servo.Servo.current().config.servo.backoff.max_tries(),
    )
    async def exec_command(self) -> servo.api.Status:
        cmd_response = await self._post_event(servo.api.Event.WHATS_NEXT, None)
        self.logger.info(f"What's Next? => {cmd_response.command}")
        self.logger.trace(devtools.pformat(cmd_response))

        if cmd_response.command == servo.api.Command.DESCRIBE:
            description = await self.describe()
            self.logger.info(
                f"Described: {len(description.components)} components, {len(description.metrics)} metrics"
            )
            self.logger.trace(devtools.pformat(description))

            status = servo.api.Status.ok(descriptor=description.__opsani_repr__())
            return await self._post_event(servo.api.Event.DESCRIPTION, status.dict())

        elif cmd_response.command == servo.api.Command.MEASURE:
            measurement = await self.measure(cmd_response.param)
            self.logger.info(
                f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
            )
            self.logger.trace(devtools.pformat(measurement))
            param = measurement.__opsani_repr__()
            return await self._post_event(servo.api.Event.MEASUREMENT, param)

        elif cmd_response.command == servo.api.Command.ADJUST:
            adjustments = servo.api.descriptor_to_adjustments(cmd_response.param["state"])
            control = Control(**cmd_response.param.get("control", {}))

            try:
                description = await self.adjust(adjustments, control)
                status = servo.api.Status.ok(state=description.__opsani_repr__())

                components_count = len(description.components)
                settings_count = sum(
                    len(component.settings) for component in description.components
                )
                self.logger.info(
                    f"Adjusted: {components_count} components, {settings_count} settings"
                )
            except servo.AdjustmentFailedError as error:
                status = servo.api.Status.from_error(error)
                self.logger.error(
                    f"Adjustment failed: {error}"
                )

            return await self._post_event(servo.api.Event.ADJUSTMENT, status.dict())

        elif cmd_response.command == servo.api.Command.SLEEP:
            # TODO: Model this
            duration = Duration(cmd_response.param.get("duration", 120))
            status = servo.utilities.key_paths.value_for_key_path(cmd_response.param, "data.status", None)
            reason = servo.utilities.key_paths.value_for_key_path(
                cmd_response.param, "data.reason", "unknown reason"
            )
            msg = f"{status}: {reason}" if status else f"{reason}"
            self.logger.info(f"Sleeping for {duration} ({msg}).")
            await asyncio.sleep(duration.total_seconds())

            # Return a status so we have a simple API contract
            return servo.api.Status(status="slept", message=msg)
        else:
            raise ValueError(f"Unknown command '{cmd_response.command.value}'")

    async def main(self, servo_: servo.servo.Servo) -> None:
        # Main run loop for processing commands from the optimizer
        async def main_loop() -> None:
            while True:
                try:
                    servo.servo.Servo.set_current(servo_)
                    status = await self.exec_command()
                    if status.status == servo.api.UNEXPECTED_EVENT:
                        self.logger.warning(
                            f"server reported unexpected event: {status.reason}"
                        )
                except Exception as error:
                    self.logger.exception(f"failed with unrecoverable error: {error}")
                    raise error

        def handle_progress_exception(error: Exception) -> None:
            # Restart the main event loop if we get out of sync with the server
            if isinstance(error, (servo.api.UnexpectedEventError, servo.api.EventCancelledError)):
                if isinstance(error, servo.api.UnexpectedEventError):
                    self.logger.error(
                        "servo has lost synchronization with the optimizer: restarting"
                    )
                elif isinstance(error, servo.api.EventCancelledError):
                    self.logger.error(
                        "optimizer has cancelled operation in progress: restarting"
                    )

                tasks = [
                    t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                ]
                self.logger.info(f"Cancelling {len(tasks)} outstanding tasks")
                [task.cancel() for task in tasks]

                # Restart a fresh main loop
                asyncio.create_task(main_loop(), name="main loop")

        # Setup logging
        servo.servo.Servo.set_current(servo_)
        self.progress_handler = servo.logging.ProgressHandler(
            self.servo.report_progress, self.logger.warning, handle_progress_exception
        )
        self.logger.add(self.progress_handler.sink, catch=True)

        self.display_banner()
        self.logger.info(
            f"Servo started with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.url or self.optimizer.base_url}]"
        )

        async def giveup(details) -> None:
            loop = asyncio.get_event_loop()
            self.logger.critical("retries exhausted, giving up")
            asyncio.create_task(self.shutdown(loop))

        try:

            @backoff.on_exception(
                backoff.expo,
                httpx.HTTPError,
                max_time=lambda: servox.Servo.current().config.backoff.max_time(),
                max_tries=lambda: servox.Servo.current().config.backoff.max_tries(),
                on_giveup=giveup,
            )
            async def connect() -> None:
                self.logger.info("Saying HELLO.", end=" ")
                await self._post_event(servo.api.Event.HELLO, dict(agent=self.USER_AGENT))
                self.connected = True

            self.logger.info("Dispatching startup event...")
            await servo_.startup()

            self.logger.info(f"Connecting to Opsani Optimizer @ {self.optimizer.api_url}...")
            await connect()
        except:
            pass

        await asyncio.create_task(main_loop(), name="main loop")

    async def shutdown(self, loop, signal=None):
        if signal:
            self.logger.info(f"Received exit signal {signal.name}...")

        try:
            if self.connected:
                reason = signal.name if signal else "shutdown"
                await self._post_event(servo.api.Event.GOODBYE, dict(reason=reason))
        except Exception:
            self.logger.exception(f"Exception occurred during GOODBYE request")

        self.logger.info("Dispatching shutdown event...")
        await self.assembly.shutdown()
        await self.progress_handler.shutdown()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        self.logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.info("Servo shutdown complete.")
        await self.logger.complete()

        loop.stop()

    def handle_exception(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        self.logger.error(f"asyncio exception handler triggered with context: {context}")

        self.logger.critical(
            "Shutting down due to unhandled exception in asyncio event loop..."
        )
        asyncio.create_task(self.shutdown(loop))

    def run(self) -> None:
        # Setup async event loop
        loop = asyncio.get_event_loop()

        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(self.shutdown(loop, signal=s))
            )

        loop.set_exception_handler(self.handle_exception)

        try:
            for servo_ in self.assembly.servos:
                loop.create_task(self.main(servo_))
            loop.run_forever()
        finally:
            loop.close()
