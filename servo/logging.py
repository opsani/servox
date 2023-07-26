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

"""The `servo.logging` module provides logging capabilities to the servo package and its dependencies.

Logging is implemented on top of the
[loguru](https://loguru.readthedocs.io/en/stable/) library.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import pathlib
import sys
import time
import traceback
from typing import Any, Awaitable, Callable, Optional, Union

import loguru

import servo
import servo.assembly
import servo.events

__all__ = (
    "Mixin",
    "Filter",
    "ProgressHandler",
    "logger",
    "log_execution",
    "log_execution_time",
    "reset_to_defaults",
    "set_level",
)

# Alias the loguru default logger
logger = loguru.logger


class Mixin:
    """Provides a convenience interface for accessing the logger as a property.

    The `servo.logging.Mixin` class is a convenience class for adding
    logging capabilities to arbitrary classes through multiple inheritance.
    """

    @property
    def logger(self) -> loguru.Logger:
        """Return the servo package logger."""
        global logger
        return logger


class Filter:
    """The level of messages that are to be outputted via logging.

    NOTE: The level on the sink needs to be set to 0.
    """

    def __init__(self, level="INFO") -> None:  # noqa: D107
        self.level = level

    def __call__(self, record) -> bool:  # noqa: D102
        levelno = logger.level(self.level).no
        return record["level"].no >= levelno


class ProgressHandler:
    """A logging handler that provides automatic progress reporting to the Opsani API.

    The `ProgressHandler` class provides connectivity between logging events and API based
    reporting to Opsani. Log messages annotated with a "progress" attribute are
    automatically picked up by the handler and reported back to the API via a callback.

    NOTE: We call the logger re-entrantly for misconfigured progress logging attempts. The
        `progress` must be excluded on logger calls to avoid recursion.
    """

    def __init__(
        self,
        progress_reporter: Callable[[dict[Any, Any]], Optional[Awaitable[None]]],
        error_reporter: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
        exception_handler: Optional[
            Callable[[dict[str, Any], Exception], Optional[Awaitable[None]]]
        ] = None,
    ) -> None:  # noqa: D107
        self._progress_reporter = progress_reporter
        self._error_reporter = error_reporter
        self._exception_handler = exception_handler
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._queue_processor: Optional[asyncio.Task[Any]] = None

    async def sink(self, message: loguru.Message) -> None:
        """Enqueue asynchronous tasks for reporting status of operations in progress.

        An asynchronous loguru sink responsible for handling progress reporting.

        Implemented as a sink versus a `logging.Handler` because the Python stdlib logging package isn't async.
        """
        if self._queue_processor is None:
            self._queue_processor = asyncio.create_task(self._process_queue())

        record = message.record
        extra = record["extra"]
        progress = extra.get("progress", None)
        if not progress:
            return

        # Favor explicit connector in extra (see Mixin) else use the context var
        connector = extra.get("connector", servo.current_connector())
        if not connector:
            return await self._report_error(
                "declining request to report progress for record without a connector attribute",
                record,
            )

        event_context: Optional[servo.events.EventContext] = servo.current_event()
        operation = extra.get("operation", None)
        if not operation:
            if not event_context:
                return await self._report_error(
                    "declining request to report progress for record without an operation parameter or inferrable value from event context",
                    record,
                )
            operation = event_context.operation()

        command_uid: Union[str, None] = servo.current_command_uid()

        started_at = extra.get("started_at", None)
        if not started_at:
            if event_context:
                started_at = event_context.created_at
            else:
                return await self._report_error(
                    "declining request to report progress for record without a started_at parameter or inferrable value from event context",
                    record,
                )

        connector_name = connector.name if hasattr(connector, "name") else connector

        self._queue.put_nowait(
            dict(
                operation=operation,
                progress=progress,
                connector=connector_name,
                event_context=event_context,
                started_at=started_at,
                message=message,
                command_uid=command_uid,
            )
        )

    def clear_progress_queue(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    async def shutdown(self) -> None:
        """Shutdown the progress handler by emptying the queue and releasing the queue processor."""
        await self._queue.join()

        if self._queue_processor:
            self._queue_processor.cancel()
            await asyncio.gather(self._queue_processor, return_exceptions=True)

    async def _process_queue(self) -> None:
        while True:
            try:
                progress = await self._queue.get()
                if progress is None:
                    logger.info(
                        f"retrieved None from progress queue. halting progress reporting"
                    )
                    break

                if int(progress["progress"]) == 100:
                    logger.debug(f"eliding 100% progress event: {progress}")
                    continue

                if asyncio.iscoroutinefunction(self._progress_reporter):
                    await self._progress_reporter(**progress)
                else:
                    self._progress_reporter(**progress)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # pylint: disable=broad-except
                logger.warning(
                    f"encountered exception while processing progress logging: {repr(progress)} => {repr(error)}"
                )
                if self._exception_handler:
                    try:
                        if asyncio.iscoroutinefunction(self._exception_handler):
                            await self._exception_handler(progress, error)
                        else:
                            self._exception_handler(progress, error)
                    except Exception as inner_error:
                        logger.critical(
                            f"encountered an exception while attempting to handle a progress reporting exception: {repr(progress)} => {repr(inner_error)} from {repr(error)}"
                        )
                        raise inner_error from error
                else:
                    logger.warning(
                        f"ignoring exception raised during progress reporting due to lack of handler: {repr(error)}"
                    )
            finally:
                self._queue.task_done()

    async def _report_error(self, message: str, record: loguru.Record) -> None:
        """Report an error message about processing a log message annotated with a `progress` attribute."""
        message = f"!!! WARNING: {record['name']}:{record['file'].name}:{record['line']} | servo.logging.ProgressHandler - {message}"
        if self._error_reporter:
            if asyncio.iscoroutinefunction(self._error_reporter):
                await self._error_reporter(message)
            else:
                self._error_reporter(message)


class InterceptHandler(logging.Handler):
    """A logging handler that forwards messages from Python stdlib logging to loguru."""

    def emit(self, record) -> None:
        """Emit a log record from Python stdlib logging facilities into loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <magenta>{extra[component]}</magenta> - <level>{message}</level>"
    "{extra[traceback]}"
)


class Formatter:
    """A logging formatter that is aware of assemblies, servos, and connectors."""

    def __call__(self, record: dict) -> str:  # noqa: D107
        """Format a log message with contextual information about the servo assembly."""
        extra = record["extra"]

        # Add optional traceback
        if extra.get("with_traceback", False):
            extra["traceback"] = "\n" + "".join(traceback.format_stack())
        else:
            extra["traceback"] = ""

        # Respect an explicit component
        if not "component" in record["extra"]:
            # Favor explicit connector from the extra dict or use the context var
            if connector := extra.get("connector", servo.current_connector()):
                component = connector.name
            else:
                component = "servo"

            # Append event context if available
            event_context = servo.current_event()
            if event_context:
                component += f"[{event_context}]"

            # If we are running multiservo, annotate that as well
            assembly = servo.current_assembly()
            if assembly and len(assembly.servos) > 1 and servo.current_servo():
                component = f"{servo.current_servo().config.optimizer.id}({component})"

            extra["component"] = component

        return DEFAULT_FORMAT + "\n{exception}"


DEFAULT_FILTER = Filter("INFO")
DEFAULT_FORMATTER = Formatter()


DEFAULT_STDERR_HANDLER = {
    "sink": sys.stderr,
    "filter": DEFAULT_FILTER,
    "level": 0,
    "format": DEFAULT_FORMATTER,
    "backtrace": True,
    "diagnose": True,
}


# Persistent disk logging to logs/
root_path = pathlib.Path(__file__).parents[1]
logs_path = root_path / "logs" / f"servo.log"


DEFAULT_FILE_HANDLER = {
    "sink": logs_path,
    "colorize": False,
    "filter": DEFAULT_FILTER,
    "level": 0,
    "format": DEFAULT_FORMATTER,
    "backtrace": True,
    "diagnose": False,
}

DEFAULT_HANDLERS = [
    DEFAULT_STDERR_HANDLER,
    DEFAULT_FILE_HANDLER,
]


def set_level(level: str) -> None:
    """Set the logging threshold to the given level for all log handlers."""
    DEFAULT_FILTER.level = level


def set_colors(colors: bool) -> None:
    """Set whether or not log messages should be outputted in ANSI color.

    Args:
        colors: Whether or not to color log output.
    """
    DEFAULT_STDERR_HANDLER["colorize"] = colors
    loguru.logger.remove()
    loguru.logger.configure(handlers=DEFAULT_HANDLERS)


def reset_to_defaults() -> None:
    """Reset the logging subsystem to the default configuration."""
    DEFAULT_FILTER.level = "INFO"
    DEFAULT_STDERR_HANDLER["colorize"] = None

    loguru.logger.remove()
    loguru.logger.configure(handlers=DEFAULT_HANDLERS)

    # Intercept messages from backoff library
    logging.getLogger("backoff").addHandler(InterceptHandler())


def friendly_decorator(f):
    """Transform a decorated function into a decorator that can be called with or without parameters.

    The returned function wraps a decorator function such that it can be invoked
    with or without parentheses such as:

        @decorator(with, arguments, and=kwargs)
        or
        @decorator
    """

    @functools.wraps(f)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return decorator


@friendly_decorator
def log_execution(func, *, entry=True, exit=True, level="DEBUG"):
    """Log the execution of the decorated function."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        name = func.__name__
        logger_ = logger.opt(depth=1)
        if entry:
            logger_.log(level, f"Entering '{name}' (args={args}, kwargs={kwargs})")
        result = func(*args, **kwargs)
        if exit:
            logger_.log(level, f"Exiting '{name}' (result={result})")
        return result

    return wrapped


@friendly_decorator
def log_execution_time(func, *, level="DEBUG"):
    """Log the execution time upon exit from the decorated function."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        from servo.types import Duration

        name = func.__name__
        logger_ = logger.opt(depth=1)

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        duration = Duration(end - start)
        logger_.log(level, f"Function '{name}' executed in {duration}")
        return result

    return wrapped


# Alias the loguru logger to hide implementation details
reset_to_defaults()
