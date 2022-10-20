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

"""The `servo.utilities.subprocess` module provides support for asynchronous
execution of subprocesses with support for timeouts, streaming output, error
management, and logging.
"""
import asyncio
import contextlib
import datetime
import pathlib
import time
from typing import (
    IO,
    Any,
    Awaitable,
    Callable,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)

import loguru

__all__ = (
    "OutputStreamCallback",
    "SubprocessResult",
    "Timeout",
    "stream_subprocess_exec",
    "run_subprocess_exec",
    "stream_subprocess_shell",
    "run_subprocess_shell",
    "stream_subprocess_output",
)


_DEFAULT_LIMIT = 2**16  # 64 KiB


# Type definition for streaming output callbacks.
# Must accept a single string positional argument and returns nothing. Optionally asynchronous.
OutputStreamCallback = TypeVar(
    "OutputStreamCallback", bound=Callable[[str], Union[None, Awaitable[None]]]
)

# Timeouts can be expressed as nummeric values in seconds or timedelta/Duration values
Timeout = Union[int, float, datetime.timedelta, None]


class SubprocessResult(NamedTuple):
    """
    An object that encapsulates the results of a subprocess execution.

    The `stdout` and `stderr` attributes will have a value of `None` when the corresponding
    attribute of the parent subprocess is not a pipe.
    """

    return_code: int
    stdout: Optional[list[str]]
    stderr: Optional[list[str]]


async def stream_subprocess_exec(
    program: str,
    *args,
    cwd: Union[pathlib.Path, Callable[[], pathlib.Path]] = pathlib.Path.cwd,
    env: Optional[dict[str, str]] = None,
    timeout: Timeout = None,
    stdout_callback: Optional[OutputStreamCallback] = None,
    stderr_callback: Optional[OutputStreamCallback] = None,
    stdin: Union[int, IO[Any], None] = None,
    stdout: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    stderr: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    limit: int = _DEFAULT_LIMIT,
    **kwargs,
) -> int:
    """
    Run a program asynchronously in a subprocess and stream its output.

    :param program: The program to run.
    :param *args: A list of string arguments to supply to the executed program.
    :param cwd: The working directory to execute the subprocess in.
    :param env: An optional dictionary of environment variables to apply to the subprocess.
    :param timeout: An optional timeout in seconds for how long to read the streams before giving up.
    :param stdout_callback: An optional callable invoked with each line read from stdout. Must accept a single string positional argument and returns nothing.
    :param stderr_callback: An optional callable invoked with each line read from stderr. Must accept a single string positional argument and returns nothing.
    :param stdin: A file descriptor, IO stream, or None value to use as the standard input of the subprocess. Default is `None`.
    :param stdout: A file descriptor, IO stream, or None value to use as the standard output of the subprocess.
    :param stderr: A file descriptor, IO stream, or None value to use as the standard error of the subprocess.
    :param limit: The amount of memory to allocate for buffering subprocess data.

    :raises asyncio.TimeoutError: Raised if the timeout expires before the subprocess exits.
    :return: The exit status of the subprocess.
    """
    process = await asyncio.create_subprocess_exec(
        program,
        *args,
        cwd=(cwd() if callable(cwd) else cwd),
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        limit=limit,
        **kwargs,
    )
    return await stream_subprocess_output(
        process,
        timeout=timeout,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )


async def run_subprocess_exec(
    program: str,
    *args,
    cwd: pathlib.Path = pathlib.Path.cwd(),
    env: Optional[dict[str, str]] = None,
    timeout: Timeout = None,
    stdin: Union[int, IO[Any], None] = None,
    stdout: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    stderr: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    limit: int = _DEFAULT_LIMIT,
    **kwargs,
) -> SubprocessResult:
    """
    Run a program asynchronously in a subprocess and return the results.

    The standard input and output are configurable but generally do not need to be changed.
    Input via `stdin` is only necessary if dynamic content needs to be supplied via `stdin`.
    Output via `stdout` and `stderr` only need to be changed for unusual configurations like redirecting
    standard error onto the standard output stream.

    :param program: The program to run.
    :param *args: A list of string arguments to supply to the executed program.
    :param cwd: The working directory to execute the subprocess in.
    :param env: An optional dictionary of environment variables to apply to the subprocess.
    :param timeout: An optional timeout in seconds for how long to read the streams before giving up.
    :param stdin: A file descriptor, IO stream, or None value to use as the standard input of the subprocess. Default is `None`.
    :param stdout: A file descriptor, IO stream, or None value to use as the standard output of the subprocess.
    :param stderr: A file descriptor, IO stream, or None value to use as the standard error of the subprocess.
    :param limit: The amount of memory to allocate for buffering subprocess data.

    :raises asyncio.TimeoutError: Raised if the timeout expires before the subprocess exits.
    :return: A named tuple value of the exit status and two string lists of standard output and standard error.
    """
    stdout_list: list[str] = []
    stderr_list: list[str] = []
    return SubprocessResult(
        await stream_subprocess_exec(
            program,
            *args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            limit=limit,
            stdout_callback=lambda m: stdout_list.append(m),
            stderr_callback=lambda m: stderr_list.append(m),
            **kwargs,
        ),
        stdout_list,
        stderr_list,
    )


async def run_subprocess_shell(
    cmd: str,
    *,
    cwd: pathlib.Path = pathlib.Path.cwd(),
    env: Optional[dict[str, str]] = None,
    timeout: Timeout = None,
    stdin: Union[int, IO[Any], None] = None,
    stdout: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    stderr: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    limit: int = _DEFAULT_LIMIT,
    **kwargs,
) -> SubprocessResult:
    """
    Run a shell command asynchronously in a subprocess and return the results.

    The standard input and output are configurable but generally do not need to be changed.
    Input via `stdin` is only necessary if dynamic content needs to be supplied via `stdin`.
    Output via `stdout` and `stderr` only need to be changed for unusual configurations like redirecting
    standard error onto the standard output stream.

    :param cmd: The command to run.
    :param cwd: The working directory to execute the subprocess in.
    :param env: An optional dictionary of environment variables to apply to the subprocess.
    :param timeout: An optional timeout in seconds for how long to read the streams before giving up.
    :param stdin: A file descriptor, IO stream, or None value to use as the standard input of the subprocess. Default is `None`.
    :param stdout: A file descriptor, IO stream, or None value to use as the standard output of the subprocess.
    :param stderr: A file descriptor, IO stream, or None value to use as the standard error of the subprocess.
    :param limit: The amount of memory to allocate for buffering subprocess data.

    :raises asyncio.TimeoutError: Raised if the timeout expires before the subprocess exits.
    :return: A named tuple value of the exit status and two string lists of standard output and standard error.
    """
    stdout_list: list[str] = []
    stderr_list: list[str] = []
    return SubprocessResult(
        await stream_subprocess_shell(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            limit=limit,
            stdout_callback=lambda m: stdout_list.append(m),
            stderr_callback=lambda m: stderr_list.append(m),
            **kwargs,
        ),
        stdout_list,
        stderr_list,
    )


async def stream_subprocess_shell(
    cmd: str,
    *,
    cwd: Union[pathlib.Path, Callable[[], pathlib.Path]] = pathlib.Path.cwd,
    env: Optional[dict[str, str]] = None,
    timeout: Timeout = None,
    stdout_callback: Optional[OutputStreamCallback] = None,
    stderr_callback: Optional[OutputStreamCallback] = None,
    stdin: Union[int, IO[Any], None] = None,
    stdout: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    stderr: Union[int, IO[Any], None] = asyncio.subprocess.PIPE,
    limit: int = _DEFAULT_LIMIT,
    **kwargs,
) -> int:
    """
    Run a shell command asynchronously in a subprocess and stream its output.

    :param cmd: The command to run.
    :param cwd: The working directory to execute the subprocess in.
    :param env: An optional dictionary of environment variables to apply to the subprocess.
    :param timeout: An optional timeout in seconds for how long to read the streams before giving up.
    :param stdout_callback: An optional callable invoked with each line read from stdout. Must accept a single string positional argument and returns nothing.
    :param stderr_callback: An optional callable invoked with each line read from stderr. Must accept a single string positional argument and returns nothing.
    :param stdin: A file descriptor, IO stream, or None value to use as the standard input of the subprocess. Default is `None`.
    :param stdout: A file descriptor, IO stream, or None value to use as the standard output of the subprocess.
    :param stderr: A file descriptor, IO stream, or None value to use as the standard error of the subprocess.
    :param limit: The amount of memory to allocate for buffering subprocess data.

    :raises asyncio.TimeoutError: Raised if the timeout expires before the subprocess exits.
    :return: The exit status of the subprocess.
    """
    process = await asyncio.create_subprocess_shell(
        cmd,
        cwd=(cwd() if callable(cwd) else cwd),
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        limit=limit,
        **kwargs,
    )
    from servo.types import Duration

    start = time.time()
    timeout_note = f" ({Duration(timeout)} timeout)" if timeout else ""
    loguru.logger.info(f"Running subprocess command `{cmd}`{timeout_note}")
    result = await stream_subprocess_output(
        process,
        timeout=timeout,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )
    end = time.time()
    duration = Duration(end - start)
    if result == 0:
        loguru.logger.success(f"Subprocess succeeded in {duration} (`{cmd}`)")
    else:
        loguru.logger.error(
            f"Subprocess failed with return code {result} in {duration} (`{cmd}`)"
        )
    return result


async def stream_subprocess_output(
    process: asyncio.subprocess.Process,
    *,
    timeout: Timeout = None,
    stdout_callback: Optional[OutputStreamCallback] = None,
    stderr_callback: Optional[OutputStreamCallback] = None,
) -> int:
    """
    Asynchronously read the stdout and stderr output streams of a subprocess and
    and optionally invoke a callback with each line of text read.

    :param process: An asyncio subprocess created with `create_subprocess_exec` or `create_subprocess_shell`.
    :param timeout: An optional timeout in seconds for how long to read the streams before giving up.
    :param stdout_callback: An optional callable invoked with each line read from stdout. Must accept a single string positional argument and returns nothing.
    :param stderr_callback: An optional callable invoked with each line read from stderr. Must accept a single string positional argument and returns nothing.

    :raises asyncio.TimeoutError: Raised if the timeout expires before the subprocess exits.
    :return: The exit status of the subprocess.
    """
    tasks = []
    if process.stdout:
        tasks.append(
            asyncio.create_task(
                _read_lines_from_output_stream(process.stdout, stdout_callback),
                name="stdout",
            )
        )
    if process.stderr:
        tasks.append(
            asyncio.create_task(
                _read_lines_from_output_stream(process.stderr, stderr_callback),
                name="stderr",
            )
        )

    timeout_in_seconds = (
        timeout.total_seconds() if isinstance(timeout, datetime.timedelta) else timeout
    )
    try:
        # Gather the stream output tasks and the parent process
        gather_task = asyncio.gather(*tasks, process.wait())
        await asyncio.wait_for(gather_task, timeout=timeout_in_seconds)

    except (asyncio.TimeoutError, asyncio.CancelledError):
        with contextlib.suppress(ProcessLookupError):
            if process.returncode is None:
                process.terminate()

        with contextlib.suppress(asyncio.CancelledError):
            await gather_task

        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)

        raise

    return cast(int, process.returncode)


async def _read_lines_from_output_stream(
    stream: asyncio.streams.StreamReader,
    callback: Optional[OutputStreamCallback],
    *,
    encoding: str = "utf-8",
) -> None:
    """
    Asynchronously read a subprocess output stream line by line,
    optionally invoking a callback with each line as it is read.

    :param stream: An IO stream reader linked to the stdout or stderr of a subprocess.
    :param callback: An optionally async callable that accepts a single string positional argument and returns nothing.
    :param encoding: The encoding to use when decoding from bytes to string (default is utf-8).
    """
    while True:
        line = await stream.readline()
        if line:
            line = line.decode(encoding).rstrip()
            if callback:
                if asyncio.iscoroutinefunction(callback):
                    await callback(line)
                else:
                    callback(line)
        else:
            break
