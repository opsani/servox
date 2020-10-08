from __future__ import annotations
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union

import yaml
from pydantic.json import pydantic_encoder

from servo.configuration import BaseConfiguration
from servo.connector import BaseConnector
from servo.events import before_event, on_event, after_event
from servo.logging import logger
from servo.servo import Events, connector
from servo.types import Measurement
from servo.utilities import SubprocessResult, Timeout, stream_subprocess_shell
import servo.events
import servo.types
from servo.types import Component, Description, RangeSetting, EnumSetting

class StubBaseConfiguration(BaseConfiguration):
    name: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> "StubBaseConfiguration":
        return cls(**kwargs)


class MeasureConnector(BaseConnector):
    config: StubBaseConfiguration

    @before_event(Events.MEASURE)
    def before_measure(self) -> None:
        pass

    @on_event()
    def measure(self, metrics: List[str] = None, control: servo.types.Control = servo.types.Control()) -> Measurement:
        pass

    @after_event(Events.MEASURE)
    def after_measure(self, results: List[servo.events.EventResult]) -> None:
        pass

class AdjustConnector(BaseConnector):
    config: StubBaseConfiguration

    @on_event()
    async def describe(self) -> Description:
        components = await self.components()
        return Description(components=components)

    @on_event()
    async def components(self) -> List[Component]:
        return [
            Component(
                name="main",
                settings=[
                    RangeSetting(name="cpu", min=0, max=10, step=1, value=3)
                ]
            )
        ]

    @on_event()
    def adjust(self, *args, **kwargs) -> Description:
        pass


@contextmanager
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
    **dict_kwargs
) -> str:
    """
    Generate configuration YAML from a dict of string to configuration objects or dicts.

    This is useful for testing aliased configurations.
    """
    config_dict = {}
    for k, v in config.items():
        if isinstance(v, BaseConfiguration):
            config_dict[k] = v.dict(**dict_kwargs)
        else:
            config_dict[k] = v
    config_json = cls.__config__.json_dumps(config, default=pydantic_encoder)
    return yaml.dump(json.loads(config_json))


def write_config_yaml(
    config: Dict[str, BaseConfiguration],
    file: Path,
    cls: Type[BaseConfiguration] = BaseConfiguration,
    **dict_kwargs
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


class SubprocessTestHelper:    
    async def shell(
        self,
        cmd: str,
        *,
        timeout: Timeout = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,        
    ) -> SubprocessResult:
        stdout: List[str] = []
        stderr: List[str] = []

        def create_output_callback(name: str, output: List[str]) -> Callable[[str], Awaitable[None]]:
            async def output_callback(msg: str) -> None:
                output.append(msg)
                m = f"[{name}] {msg}"
                if print_output:
                    print(m)
                if log_output:
                    logger.debug(m)

            return output_callback
        
        print(f"\n❯ Executing `{cmd}`")
        return_code = await stream_subprocess_shell(
            cmd,
            timeout=timeout,
            stdout_callback=create_output_callback("stdout", stdout),
            stderr_callback=create_output_callback("stderr", stderr),
        )
        return SubprocessResult(return_code, stdout, stderr)

    async def __call__(
        self,
        cmd: str,
        *,
        timeout: Timeout = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,        
    ) -> SubprocessResult:
        return await self.shell(cmd, timeout=timeout, print_output=print_output, log_output=log_output, **kwargs)