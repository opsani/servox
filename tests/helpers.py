from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import pathlib
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
import typing

import fastapi
import httpx
import kubernetes_asyncio.client
import pytest
import respx
import uvicorn
import yaml
from pydantic.json import pydantic_encoder

import servo.events
import servo.types
from servo.configuration import BaseConfiguration
from servo.connector import BaseConnector
from servo.events import after_event, before_event, on_event
from servo.logging import logger
from servo.servo import Events
from servo.types import (
    Component,
    DataPoint,
    Description,
    Measurement,
    Metric,
    RangeSetting,
    Unit,
)
from servo.utilities import SubprocessResult, Timeout, stream_subprocess_shell


class MeasureConnector(BaseConnector):
    @on_event()
    async def metrics(self) -> List[Metric]:
        return [
            Metric(name="throughput", unit=Unit.requests_per_minute),
            Metric(name="error_rate", unit=Unit.requests_per_minute),
        ]

    @on_event()
    async def describe(self, control: servo.Control = servo.Control()) -> Description:
        metrics = await self.metrics()
        return Description(metrics=metrics)

    @before_event(Events.measure)
    def before_measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> None:
        pass

    @on_event()
    def measure(
        self,
        metrics: List[str] = None,
        control: servo.types.Control = servo.types.Control(),
    ) -> Measurement:
        return Measurement(
            readings=[
                DataPoint(
                    time=datetime.datetime.now(),
                    value=31337,
                    metric=Metric(
                        name="Some Metric",
                        unit=Unit.requests_per_minute,
                    ),
                )
            ]
        )

    @after_event(Events.measure)
    def after_measure(self, results: List[servo.events.EventResult]) -> None:
        pass


class AdjustConnector(BaseConnector):
    @on_event()
    async def describe(self, control: servo.Control = servo.Control()) -> Description:
        components = await self.components()
        return Description(components=components)

    @on_event()
    async def components(self) -> List[Component]:
        return [
            Component(
                name="main",
                settings=[
                    RangeSetting(
                        name="cpu", min=0, max=10, step=1, value=3, unit="cores"
                    )
                ],
            )
        ]

    @on_event()
    async def adjust(
        self,
        adjustments: List[servo.Adjustment],
        control: servo.Control = servo.Control(),
    ) -> servo.Description:
        return await self.describe()


@contextlib.contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env


def generate_config_yaml(
    config: Dict[str, Union[BaseConfiguration, dict]],
    cls: Type[BaseConfiguration] = BaseConfiguration,
    **dict_kwargs,
) -> str:
    """
    Generate configuration YAML from a dict of string to configuration objects or dicts.

    This is useful for testing aliased configurations.
    """
    config_dict = {}
    for k, v in config.items():
        if isinstance(v, BaseConfiguration):
            config_dict[k] = v.model_dump(**dict_kwargs)
        else:
            config_dict[k] = v
    config_json = cls.__config__.json_dumps(config, default=pydantic_encoder)
    return yaml.dump(json.loads(config_json), sort_keys=False)


def write_config_yaml(
    config: Dict[str, BaseConfiguration],
    file: Path,
    cls: Type[BaseConfiguration] = BaseConfiguration,
    **dict_kwargs,
) -> str:
    config_yaml = generate_config_yaml(config)
    file.write_text(config_yaml)
    return config_yaml


def dict_key_path(obj: dict, key_path: str) -> Any:
    components = key_path.split(".")
    for component in components:
        obj = obj[component]
    return obj


def yaml_key_path(yaml_str: str, key_path: str) -> Any:
    """
    Parse a YAML document and return a value at the target key-path.
    """
    obj = yaml.full_load(yaml_str)
    return dict_key_path(obj, key_path)


def json_key_path(json_str: str, key_path: str) -> Any:
    """
    Parse a JSON document and return a value at the target key-path.
    """
    obj = json.loads(json_str)
    return dict_key_path(obj, key_path)


E = typing.TypeVar("E")


def unwrap_exception_group(
    excg: ExceptionGroup, expected_type: type[E], expected_count: int | None = None
) -> E | list[E]:
    flattened_group = flatten_exception_group(exception_group=excg)
    excg_len = len(flattened_group)
    if expected_count is not None:
        assert (
            excg_len == expected_count
        ), f"Excpetion group count {excg_len} did not match expected count {expected_count}"

    assert excg_len > 0
    if not all(isinstance(e, expected_type) for e in flattened_group):
        raise excg

    servo.errors.ServoError.servo_error_from_group
    if excg_len > 1:
        return flattened_group
    else:
        return flattened_group[0]


def flatten_exception_group(exception_group: ExceptionGroup) -> list[Exception]:
    exc_list = []
    # traverse tree of exceptions depth first. Evaluate priority and raise as the "main" exception with others included as property additional exceptions
    visit_list = list(exception_group.exceptions)
    while visit_list:
        exc = visit_list.pop(0)
        if isinstance(exc, ExceptionGroup):
            visit_list = list(exc.exceptions) + visit_list
        else:
            exc_list.append(exc)

    return exc_list


class Subprocess:
    @staticmethod
    async def shell(
        cmd: str,
        *,
        timeout: Timeout = None,
        event: Optional[asyncio.Event] = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,
    ) -> SubprocessResult:
        stdout: List[str] = []
        stderr: List[str] = []

        def create_output_callback(
            name: str, output: List[str]
        ) -> Callable[[str], Awaitable[None]]:
            def output_callback(msg: str) -> None:
                output.append(msg)
                m = f"[{name}] {msg}"
                if print_output:
                    print(m)
                if log_output:
                    logger.debug(m)
                if event:
                    event.set()

            return output_callback

        print(f"\n❯ Executing `{cmd}`")
        return_code = await stream_subprocess_shell(
            cmd,
            timeout=timeout,
            stdout_callback=create_output_callback("stdout", stdout),
            stderr_callback=create_output_callback("stderr", stderr),
            **kwargs,
        )
        return SubprocessResult(return_code, stdout, stderr)

    async def __call__(
        self,
        cmd: str,
        *,
        timeout: Timeout = None,
        event: Optional[asyncio.Event] = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,
    ) -> SubprocessResult:
        return await self.shell(
            cmd,
            timeout=timeout,
            event=event,
            print_output=print_output,
            log_output=log_output,
            **kwargs,
        )


api_mock = respx.mock(base_url="https://api.opsani.com/", assert_all_called=True)
api_mock.post("/accounts/dev.opsani.com/applications/servox/servo", name="servo").mock(
    return_value=httpx.Response(200, json={"status": "ok", "command": "SLEEP"}),
)


class FakeAPI(uvicorn.Server):
    """Testing server for implementing API fakes on top of Uvicorn and FastAPI.

    The test server is meant to be paired with pytest fixtures that enable a
    simple mechanism for utilizing API fakes in testing.

    A fake is a protocol compliant stand-in for another system that aids in testing
    by providing stateless, deterministic, and isolated implementations of dependent
    services. Fakes tend to be easier to develop and less brittle than mocking, which
    tends to cut out entire subsystems such as network transport. A fake, in contrast,
    focuses on delivering a request/response compatible stand-in for the real system
    and supports high velocity development and testing by eliminating concerns such as
    stateful persistence, cross talk from other users/developers, and the drag of latency.

    Usage:
        @pytest.fixture
        async def fakeapi_url(fastapi_app: fastapi.FastAPI, unused_tcp_port: int) -> AsyncIterator[str]:
            server = FakeAPI(fastapi_app, port=unused_tcp_port)
            await server.start()
            yield server.base_url
            await server.stop()
    """

    def __init__(
        self,
        app: fastapi.FastAPI,
        host: str = "127.0.0.1",
        port: int = 8000,
        task_group: asyncio.TaskGroup | None = None,
    ) -> None:
        """Initialize a FakeAPI instance by mounting a FastAPI app and starting Uvicorn.

        Args:
            app (FastAPI, optional): the FastAPI app.
            host (str, optional): the host ip. Defaults to '127.0.0.1'.
            port (int, optional): the port. Defaults to 8000.
        """
        self._startup_done = asyncio.Event()
        self._task_group = task_group
        super().__init__(config=uvicorn.Config(app, host=host, port=port))

    async def startup(self, sockets: Optional[List] = None) -> None:
        """Override Uvicorn startup to signal any tasks blocking to await startup."""
        await super().startup(sockets=sockets)
        self._startup_done.set()

    async def start(self) -> None:
        """Start up the server and wait for it to initialize."""
        if self._task_group:
            self._serve_task = self._task_group.create_task(self.serve())
        else:
            self._serve_task = asyncio.create_task(self.serve())
        await self._startup_done.wait()

    async def stop(self) -> None:
        """Shut down server asynchronously."""
        self.should_exit = True
        await self._serve_task

    @property
    def base_url(self) -> str:
        """Return the base URL for accessing the FakeAPI server."""
        return f"http://{self.config.host}:{self.config.port}/"


@contextlib.asynccontextmanager
async def kubernetes_asyncio_client_overrides(
    **kwargs,
) -> AsyncIterator[kubernetes_asyncio.client.Configuration]:
    """Override fields on the default kubernetes_asyncio.client.Configuration within the context.

    Fields are set directly on a copy of the original configuration using `setattr`. Refer to documentation
    of the kubernetes_asyncio.client.Configuration to see what is available.

    Yields the updated configuration instance.
    """
    original_config = kubernetes_asyncio.client.Configuration.get_default_copy()
    new_config = kubernetes_asyncio.client.Configuration.get_default_copy()
    for attr, value in kwargs.items():
        setattr(new_config, attr, value)

    try:
        kubernetes_asyncio.client.Configuration.set_default(new_config)
        yield new_config

    finally:
        kubernetes_asyncio.client.Configuration.set_default(original_config)


async def build_docker_image(
    tag: str = "opsani/servox:edge",
    *,
    preamble: Optional[str] = None,
    print_output: bool = True,
    **kwargs,
) -> str:
    root_path = pathlib.Path(__file__).parents[1]
    subprocess = Subprocess()
    exit_code, stdout, stderr = await subprocess(
        f"{preamble or 'true'} && DOCKER_BUILDKIT=1 docker build -t {tag} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:edge {root_path}",
        print_output=print_output,
        **kwargs,
    )
    if exit_code != 0:
        error = "\n".join(stderr)
        raise RuntimeError(
            f"Docker build failed with exit code {exit_code}: error: {error}"
        )

    return tag
