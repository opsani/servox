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

import asyncio
import functools
import os
import random
import shutil
import signal
from typing import Any, Optional, Union

import colorama
import devtools
import httpx
import pydantic
import pyfiglet
import typer

import servo
import servo.api
import servo.telemetry
import servo.configuration
import servo.utilities.key_paths
import servo.utilities.strings
from servo.servo import _set_current_servo, set_current_command_uid
from servo.types import Adjustment, Control, Description, Duration, Measurement


class ServoRunner(pydantic.BaseModel, servo.logging.Mixin):
    interactive: bool = False
    _assembly_runner: AssemblyRunner = pydantic.PrivateAttr(None)
    _servo: servo.Servo = pydantic.PrivateAttr(None)
    _running: bool = pydantic.PrivateAttr(False)
    _file_watcher_task: Optional[asyncio.Task] = pydantic.PrivateAttr(None)
    _main_loop_task: Optional[asyncio.Task] = pydantic.PrivateAttr(None)
    _task_group: Optional[asyncio.TaskGroup] = pydantic.PrivateAttr(None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, servo_: servo, _assembly_runner: AssemblyRunner = None, **kwargs
    ) -> None:  # noqa: D10
        super().__init__(**kwargs)
        self._servo = servo_
        self._assembly_runner = _assembly_runner

        # initialize default servo options if not configured
        if self.config.settings is None:
            self.config.settings = servo.CommonConfiguration()

    @property
    def servo(self) -> servo.Servo:
        return self._servo

    @property
    def running(self) -> bool:
        return self._running

    @property
    def connected(self) -> bool:
        return self.servo._connected

    @property
    def optimizer(
        self,
    ) -> servo.configuration.OptimizerTypes:
        return self.servo.optimizer

    @property
    def config(self) -> servo.BaseServoConfiguration:
        return self.servo.config

    async def describe(self, control: Control) -> Description:
        self.logger.info("Describing...")

        aggregate_description = Description.construct()
        results: list[servo.EventResult] = await self.servo.dispatch_event(
            servo.Events.describe, control=control
        )
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        return aggregate_description

    async def measure(self, param: servo.api.MeasureParams) -> Measurement:
        if isinstance(param, dict):
            # required parsing has failed in servo.post_event(), run parse_obj to surface the validation errors
            servo.api.MeasureParams.parse_obj(param)
        servo.logger.info(f"Measuring... [metrics={', '.join(param.metrics)}]")
        servo.logger.trace(devtools.pformat(param))

        aggregate_measurement = Measurement.construct()
        results: list[servo.EventResult] = await self.servo.dispatch_event(
            servo.Events.measure, metrics=param.metrics, control=param.control
        )
        for result in results:
            measurement = result.value
            aggregate_measurement.readings.extend(measurement.readings)
            aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    async def adjust(
        self, adjustments: list[Adjustment], control: Control
    ) -> Description:
        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        self.logger.info(f"Adjusting... {summary}")
        self.logger.trace(devtools.pformat(adjustments))
        self.logger.trace(devtools.pformat(control))

        aggregate_description = Description.construct()
        results = await self.servo.dispatch_event(
            servo.Events.adjust, adjustments=adjustments, control=control
        )
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        self.logger.success(f"Adjustment completed {summary}")
        return aggregate_description

    async def exec_command(self) -> servo.api.Status:
        cmd_response: Union[
            servo.api.CommandResponse, servo.api.Status
        ] = await self.servo.post_event(servo.api.Events.whats_next, None)
        self.logger.trace(devtools.pformat(cmd_response))
        self.logger.info(f"What's Next? => {cmd_response.command}")
        set_current_command_uid(cmd_response.command_uid)

        if cmd_response.command == servo.api.Commands.describe:
            try:
                description = await self.describe(
                    Control(**cmd_response.param.get("control", {}))
                )
                self.logger.success(
                    f"Described: {len(description.components)} components, {len(description.metrics)} metrics"
                )
                self.logger.debug(devtools.pformat(description))

                status = servo.api.Status.ok(
                    descriptor=description.__opsani_repr__(),
                    command_uid=cmd_response.command_uid,
                )
            except* servo.errors.EventError as error_group:
                self.logger.error(f"Describe failed: {error_group.exceptions}")
                top_error = servo.errors.ServoError.servo_error_from_group(error_group)
                status = servo.api.Status.from_error(
                    error=top_error,
                    command_uid=cmd_response.command_uid,
                )
                self.logger.error(f"Responding with {status.dict()}")
                self.logger.opt(exception=error_group).debug("Describe failure details")

            self.clear_progress_queue()
            return await self.servo.post_event(servo.api.Events.describe, status.dict())

        elif cmd_response.command == servo.api.Commands.measure:
            try:
                measurement = await self.measure(cmd_response.param)
                self.logger.success(
                    f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
                )
                self.logger.trace(devtools.pformat(measurement))
                status = servo.api.Status.ok(
                    command_uid=cmd_response.command_uid,
                    **measurement.__opsani_repr__(),
                )
            except* servo.errors.EventError as error_group:
                self.logger.error(f"Measurement failed: {error_group.exceptions}")
                top_error = servo.errors.ServoError.servo_error_from_group(error_group)
                status = servo.api.Status.from_error(
                    error=top_error,
                    command_uid=cmd_response.command_uid,
                )
                self.logger.error(f"Responding with {status.dict()}")
                self.logger.opt(exception=error_group).debug("Measure failure details")

            self.clear_progress_queue()
            return await self.servo.post_event(servo.api.Events.measure, status.dict())

        elif cmd_response.command == servo.api.Commands.adjust:
            adjustments = servo.api.descriptor_to_adjustments(
                cmd_response.param["state"]
            )
            control = Control(**cmd_response.param.get("control", {}))

            try:
                description = await self.adjust(adjustments, control)
                status = servo.api.Status.ok(
                    state=description.__opsani_repr__(),
                    command_uid=cmd_response.command_uid,
                )

                components_count = len(description.components)
                settings_count = sum(
                    len(component.settings) for component in description.components
                )
                self.logger.success(
                    f"Adjusted: {components_count} components, {settings_count} settings"
                )
            except* servo.errors.EventError as error_group:
                self.logger.error(f"Adjustment failed: {error_group.exceptions}")
                self.logger.error(f"Describe failed: {error_group.exceptions}")
                top_error = servo.errors.ServoError.servo_error_from_group(error_group)
                status = servo.api.Status.from_error(
                    error=top_error,
                    command_uid=cmd_response.command_uid,
                )
                self.logger.error(f"Responding with {status.dict()}")
                self.logger.opt(exception=error_group).debug("Adjust failure details")

            self.clear_progress_queue()
            return await self.servo.post_event(servo.api.Events.adjust, status.dict())

        elif cmd_response.command == servo.api.Commands.sleep:
            # TODO: Model this
            duration = Duration(cmd_response.param.get("duration", 120))
            status = servo.utilities.key_paths.value_for_key_path(
                cmd_response.param, "data.status", None
            )
            reason = servo.utilities.key_paths.value_for_key_path(
                cmd_response.param, "data.reason", "unknown reason"
            )
            msg = f"{status}: {reason}" if status else f"{reason}"
            self.logger.info(f"Sleeping for {duration} ({msg}).")
            await asyncio.sleep(duration.total_seconds())

            # Return a status so we have a simple API contract
            return servo.api.Status(status="ok", message=msg)
        else:
            raise ValueError(f"Unknown command '{cmd_response.command.value}'")

    # Main run loop for processing commands from the optimizer
    # FIXME: We have seen exceptions from using `with self.servo.current()` crossing contexts
    async def run_main_loop(self) -> None:
        async with asyncio.TaskGroup() as tg:
            self._task_group = tg

            if not servo.current_servo().config.no_diagnostics:
                diagnostics_handler = servo.telemetry.DiagnosticsHandler(self.servo)
                self._diagnostics_loop_task = self._task_group.create_task(
                    diagnostics_handler.diagnostics_check(),
                    name=f"diagnostics for servo {self.optimizer.id}",
                )
            else:
                self.logger.info(
                    f"Servo runner initialized with diagnostics polling disabled"
                )

            if getattr(self.optimizer, "connection_file", None):
                self._file_watcher_task = self._task_group.create_task(
                    self.servo.watch_connection_file(),
                    name=f"connection file watcher for servo {self.optimizer.id}",
                )

            while self.running:
                try:
                    if self.interactive:
                        if not typer.confirm("Poll for next command?"):
                            typer.echo("Sleeping for 1m")
                            await asyncio.sleep(60)
                            continue

                    self.logger.info("getting next command")
                    status = await self.exec_command()
                    if status.status == servo.api.OptimizerStatuses.unexpected_event:
                        self.logger.warning(
                            f"server reported unexpected event: {status.reason}"
                        )

                except httpx.TimeoutException as error:
                    self.logger.warning(
                        f"command execution failed HTTP client timeout error: {error}"
                    )

                except httpx.HTTPStatusError as error:
                    if (
                        error.response.status_code == 410
                        and error.response.json().get("detail")
                        == "unexpected servo_uid"
                    ):
                        self.logger.warning(
                            f"servo UID {self.servo.config.servo_uid} is no longer valid. Waiting for deprovisioning (will sleep for 1 hour)"
                        )
                        await asyncio.sleep(3600)
                    else:
                        self.logger.warning(
                            f"command execution failed HTTP client status error: {error}"
                        )

                except pydantic.ValidationError as error:
                    self.logger.warning(
                        f"command execution failed with model validation error: {error}"
                    )
                    self.logger.opt(exception=error).debug(
                        "Pydantic model failed validation"
                    )

                except Exception as error:
                    self.logger.exception(
                        f"Exiting from servo main loop due to unrecoverable error: {error}"
                    )
                    raise error

            self.logger.info("Main loop exited, cancelling task group")
            raise asyncio.CancelledError("Main loop exited, cancelling task group")

    async def run(self, *, poll: bool = True, startup: bool = True) -> None:
        self._running = True
        _set_current_servo(self.servo)

        if startup:
            try:
                self.logger.info(
                    f"Connecting to Opsani Optimizer @ {self.optimizer.url}..."
                )
                if self.interactive:
                    typer.confirm("Connect to the optimizer?", abort=True)

                await self.servo.startup()
                self.logger.info(
                    f"Servo started with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.url or self.optimizer.base_url}]"
                )
            except typer.Abort:
                # Rescue abort and notify user
                servo.logger.warning("Operation aborted. Use Control-C to exit")
            except asyncio.CancelledError as error:
                self.logger.trace("task cancelled, aborting servo runner")
                raise error
            except:
                self.logger.exception("exception encountered during startup")
                await self.shutdown()
                return

        if poll:
            # block until exit or cancellation internally or externally
            self._main_loop_task = asyncio.create_task(self.run_main_loop())
            await self._main_loop_task
        else:
            self.logger.warning(
                f"Servo runner initialized with polling disabled -- command loop is not running"
            )

    async def shutdown(self, *, delay: float = 0.0) -> None:
        """Shutdown the running servo."""
        try:
            self.logger.info("Shutting down servo runner")
            self.logger.debug(
                f"self._main_loop_task {self._main_loop_task}\nself._task_group {self._task_group}"
            )
            self._running = False

            if delay > 0:
                self.logger.info(f"waiting {delay} seconds for graceful shutdown")
                await asyncio.sleep(delay=delay)

            if not self._main_loop_task.done():
                self.logger.info("ServoRunner _main_loop_task exited gracefully")
            else:
                self.logger.info(
                    f"Cancelling ServoRunner _main_loop_task and running task_group: {self._task_group} _exiting {getattr(self._task_group, '_exiting')}"
                )
                self.logger.debug(
                    f"Current ServoRunner task group: {devtools.pformat(self._task_group)}"
                )
                self._main_loop_task.cancel()
                await asyncio.gather(self._main_loop_task, return_exceptions=True)

        except Exception:
            self.logger.exception(f"Exception occurred during servo runner shutdown")

    def clear_progress_queue(self) -> None:
        if self._assembly_runner and self._assembly_runner.progress_handler:
            self.logger.debug("Clearing progress handler queue")
            self._assembly_runner.progress_handler.clear_progress_queue()


class AssemblyRunner(pydantic.BaseModel, servo.logging.Mixin):
    assembly: servo.Assembly
    runners: list[ServoRunner] = []
    progress_handler: Optional[servo.logging.ProgressHandler] = None
    progress_handler_id: Optional[int] = None
    poll: bool = True
    interactive: bool = False
    _running: bool = pydantic.PrivateAttr(False)
    _shutting_down: bool = pydantic.PrivateAttr(False)
    _root_task: asyncio.Task | None = pydantic.PrivateAttr(None)
    _task_group: asyncio.TaskGroup | None = pydantic.PrivateAttr(None)
    _runners_task_group: asyncio.TaskGroup | None = pydantic.PrivateAttr(None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, assembly: servo.Assembly, **kwargs) -> None:
        super().__init__(assembly=assembly, **kwargs)

    def _runner_for_servo(self, servo: servo.Servo) -> ServoRunner:
        for runner in self.runners:
            if runner.servo == servo:
                return runner

        raise KeyError(f'no runner was found for the servo: "{servo}"')

    @property
    def running(self) -> bool:
        return self._running

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def run(self, *, debug: bool = False) -> None:
        """Asynchronously run all servos active within the assembly.

        Running the assembly takes over the current event loop and schedules a `ServoRunner` instance for each servo active in the assembly.
        """
        if self.running:
            raise RuntimeError("Cannot run an assembly that is already running")

        self._running = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            if str(e) == "no running event loop":
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise

        # Setup signal handling
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: loop.create_task(self.shutdown(loop, signal=s))
            )

        if not debug:
            loop.set_exception_handler(self._handle_exception)
        else:
            loop.set_exception_handler(None)

        try:
            self._root_task = loop.create_task(self._run())
            # get result of maint task so excpetions will be raised
            _ = loop.run_until_complete(self._root_task)
            loop.run_forever()

        finally:
            loop.close()

    async def _run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            self._task_group = tg
            self.progress_handler = servo.logging.ProgressHandler(
                self._report_progress,
                self.logger.warning,
                self._handle_progress_exception,
                self._task_group,
            )
            self.progress_handler_id = self.logger.add(self.progress_handler.sink)

            self._display_banner()

            # Allow additional runners to be added to task group but stop running when all runners have finished
            # TODO may need tweaks to work with progres exception
            async with asyncio.TaskGroup() as runners_tg:
                self._runners_task_group = runners_tg
                for servo_ in self.assembly.servos:
                    servo_runner = ServoRunner(
                        servo_, interactive=self.interactive, _assembly_runner=self
                    )

                    _ = self._runners_task_group.create_task(
                        servo_runner.run(poll=self.poll),
                        name=f"runner for servo {servo_.optimizer.id}",
                    )
                    self.runners.append(servo_runner)

            self.logger.info("All servo runners have completed, exiting")
            raise asyncio.CancelledError("All servo runners have completed")

    # Setup logging
    async def _report_progress(self, **kwargs) -> None:
        # Forward to the active servo...
        if servo_ := servo.current_servo():
            await servo_.report_progress(**kwargs)
        else:
            self.logger.warning(
                f"failed progress reporting -- no current servo context is established (kwargs={devtools.pformat(kwargs)})"
            )

    def _display_banner(self) -> None:
        fonts = [
            "slant",
            "banner3",
            "bigchief",
            "cosmic",
            "speed",
            "nancyj",
            "fourtops",
            "contessa",
            "doom",
            "broadway",
            "acrobatic",
            "trek",
            "eftirobot",
            "roman",
        ]
        color_map = {
            "RED": colorama.Fore.RED,
            "GREEN": colorama.Fore.GREEN,
            "YELLOW": colorama.Fore.YELLOW,
            "BLUE": colorama.Fore.BLUE,
            "MAGENTA": colorama.Fore.MAGENTA,
            "CYAN": colorama.Fore.CYAN,
            "RAINBOW": colorama.Fore.MAGENTA,
        }
        terminal_size = shutil.get_terminal_size()

        # Generate an awesome banner for this launch
        font = os.getenv("SERVO_BANNER_FONT", random.choice(fonts))
        color_name = os.getenv("SERVO_BANNER_COLOR")
        # coinflip unless we have been directly configured from the env
        rainbow = (
            bool(random.getrandbits(1))
            if color_name is None
            else (color_name.upper() == "RAINBOW")
        )

        figlet = pyfiglet.Figlet(font=font, width=terminal_size.columns)
        banner = figlet.renderText("ServoX").rstrip()

        if rainbow:
            # Rainbow it
            colored_banner = [
                random.choice(list(color_map.values())) + char for char in banner
            ]
            typer.echo("".join(colored_banner), color=True)
        else:
            # Flat single color
            color = (
                color_map[color_name.upper()]
                if color_name
                else random.choice(list(color_map.values()))
            )
            typer.echo(f"{color}{banner}", color=True)

        secho = functools.partial(typer.secho, color=True)
        types = servo.Assembly.all_connector_types()
        types.remove(servo.Servo)

        names = []
        for c in types:
            name = typer.style(
                servo.utilities.strings.commandify(c.__default_name__),
                fg=typer.colors.CYAN,
                bold=False,
            )
            version = typer.style(str(c.version), fg=typer.colors.WHITE, bold=True)
            names.append(f"{name}-{version}")

        version = typer.style(f"v{servo.__version__}", fg=typer.colors.WHITE, bold=True)
        codename = typer.style(servo.__cryptonym__, fg=typer.colors.MAGENTA, bold=False)
        initialized = typer.style(
            "initialized", fg=typer.colors.BRIGHT_GREEN, bold=True
        )
        version = typer.style(f"v{servo.__version__}", fg=typer.colors.WHITE, bold=True)

        secho(f'{version} "{codename}" {initialized}')
        secho(reset=True)
        secho(f"connectors:  {', '.join(sorted(names))}")
        secho(
            f"config file: {typer.style(str(self.assembly.config_file), bold=True, fg=typer.colors.YELLOW)}"
        )

        if len(self.assembly.servos) == 1:
            servo_ = self.assembly.servos[0]

            servo_uid = typer.style(
                servo_.config.servo_uid, bold=True, fg=typer.colors.WHITE
            )
            secho(f"servo UID: {servo_uid}")

            optimizer = servo_.optimizer
            id = typer.style(optimizer.id, bold=True, fg=typer.colors.WHITE)
            secho(f"optimizer:   {id}")

            if optimizer.base_url != "https://api.opsani.com/":
                base_url = typer.style(
                    f"{optimizer.base_url}", bold=True, fg=typer.colors.RED
                )
                secho(f"base url: {base_url}")

            if servo_.config.settings and servo_.config.settings.proxies:
                proxies = typer.style(
                    f"{devtools.pformat(servo_.config.settings.proxies)}",
                    bold=True,
                    fg=typer.colors.CYAN,
                )
                secho(f"proxies: {proxies}")
        else:
            servo_count = typer.style(
                str(len(self.assembly.servos)), bold=True, fg=typer.colors.WHITE
            )
            secho(f"servos:   {servo_count}")

        secho(reset=True)

    async def shutdown(self, loop: asyncio.AbstractEventLoop, signal=None) -> None:
        if not self.running:
            raise RuntimeError("Cannot shutdown an assembly that is not running")

        if self._shutting_down:
            self.logger.warning(
                "AssemblyRunner already shutting down, ignoring redundant call"
            )
            return

        self._shutting_down = True

        if signal:
            self.logger.info(f"Received exit signal {signal.name}...")

        reason = signal.name if signal else "shutdown"

        # Shut down the servo runners, breaking active control loops
        if len(self.runners) == 1:
            self.logger.info(f"Shutting down servo...")
        else:
            self.logger.info(f"Shutting down {len(self.runners)} running servos...")

        self.logger.debug(
            f"self._root_task {self._root_task}\nself._task_group {self._task_group}"
        )
        for fut in asyncio.as_completed(
            list(map(lambda r: r.shutdown(), self.runners)), timeout=30.0
        ):
            try:
                await fut
            except Exception as error:
                self.logger.critical(
                    f"Failed servo runner shutdown with error: {error}"
                )

        # Shutdown the assembly and the servos it contains
        self.logger.debug("Dispatching shutdown event...")
        try:
            await self.assembly.shutdown(reason=reason)
        except Exception as error:
            self.logger.critical(f"Failed assembly shutdown with error: {error}")

        try:
            await self.progress_handler.shutdown()
        except Exception as error:
            self.logger.warning(f"Failed progress handler shutdown with error: {error}")

        self.logger.remove(self.progress_handler_id)

        # Cancel any outstanding tasks -- under a clean, graceful shutdown this list will be empty
        # The shutdown of the assembly and the servo should clean up its tasks
        if self._root_task is None:
            self.logger.warning(
                f"AssemblyRunner root_task set to None, skipping task cleanup (task_group: {self._task_group})"
            )
        elif self._root_task.done():
            self.logger.info(
                "AssemblyRunner root_task exited gracefully after shutdown"
            )
        else:
            self.logger.info(
                f"Cancelling AssemblyRunner root_task and running task_group: {self._task_group}"
            )
            self.logger.debug(
                f"Current AssemblyRunner task group: {devtools.pformat(self._task_group)}"
            )
            self._root_task.cancel()
            await asyncio.gather(self._root_task, return_exceptions=True)

        self.logger.info("Servo shutdown complete.")
        await asyncio.gather(self.logger.complete(), return_exceptions=True)

        self._running = False

        loop.stop()

    def _handle_exception(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        self.logger.debug(
            f"asyncio exception handler triggered with context: {context}"
        )

        exception = context.get("exception", None)
        logger = self.logger.opt(exception=exception)

        if isinstance(exception, asyncio.CancelledError):
            logger.warning(f"ignoring asyncio.CancelledError exception")
        elif loop.is_closed():
            logger.critical("Ignoring exception -- the event loop is closed.")
        elif self.running:
            logger.critical(
                f"Shutting down due to unhandled exception {exception} in asyncio event loop..."
            )
            loop.create_task(self.shutdown(loop))
        else:
            logger.critical("Ignoring exception -- the assembly is not running")

    async def _handle_progress_exception(
        self, progress: dict[str, Any], error: Exception
    ) -> None:
        # FIXME: This needs to be made multi-servo aware
        # Restart the main event loop if we get out of sync with the server
        ###### TODO get current servo and call its shutdown instead of task reaping
        if isinstance(
            error,
            (
                servo.errors.UnexpectedEventError,
                servo.errors.EventCancelledError,
                servo.errors.UnexpectedCommandIdError,
            ),
        ) or (
            isinstance(error, httpx.HTTPStatusError)
            and error.response.status_code == 410
            and error.response.json().get("detail")
        ):
            servo_ = progress["servo"]
            if isinstance(error, httpx.HTTPStatusError):
                self.logger.error(
                    f"servo UID {servo.current_servo().config.servo_uid} is no longer valid: shutting down tasks to commence sleep loop"
                )
            elif isinstance(error, servo.errors.UnexpectedEventError):
                self.logger.error(
                    "servo has lost synchronization with the optimizer: restarting"
                )
            elif isinstance(error, servo.errors.UnexpectedCommandIdError):
                self.logger.error("servo is processing outdated command: restarting")
            elif isinstance(error, servo.errors.EventCancelledError):
                self.logger.error(
                    "optimizer has cancelled operation in progress: cancelling and restarting loop"
                )

                # Post a status to resolve the operation
                operation = progress["operation"]
                command_uid = progress["command_uid"]
                status = servo.api.Status.from_error(
                    error=error, command_uid=command_uid
                )
                self.logger.error(f"Responding with {status.dict()}")
                await servo_.post_event(operation, status.dict())

            if self.shutting_down:
                self.logger.warning(
                    "restart attmempted during shutdown of servo. aborting restart"
                )
                return

            runner = self._runner_for_servo(servo_)
            # stop servo main loop
            await runner.shutdown()
            # call detach and shutdown
            await self.assembly.remove_servo(self.servo, reason="restarting")

            # Restart a fresh main loop
            if self.poll:
                servo_.load_connectors()
                # call attach and startup
                await self.assembly.add_servo(servo_)
                _ = self._runners_task_group.create_task(
                    runner.run(poll=self.poll),
                    name=f"runner for servo {servo_.optimizer.id}",
                )

        else:
            self.logger.error(
                f"unrecognized exception passed to progress exception handler: {error}"
            )
