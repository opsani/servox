import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import yaml
from pydantic.json import pydantic_encoder

from servo.connector import BaseConfiguration, Connector
from servo.servo import Events, connector
from servo.types import Measurement


class StubBaseConfiguration(BaseConfiguration):
    name: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> "StubBaseConfiguration":
        return cls(**kwargs)


class MeasureConnector(Connector):
    config: StubBaseConfiguration

    @connector.before_event(Events.MEASURE)
    def before_measure(self, *args, **kwargs) -> None:
        pass

    @connector.on_event()
    def measure(self, *args, **kwargs) -> Measurement:
        pass

    @connector.after_event(Events.MEASURE)
    def after_measure(self, *args, **kwargs) -> None:
        pass


class AdjustConnector(Connector):
    config: StubBaseConfiguration

    @connector.on_event()
    def adjust(self, *args, **kwargs) -> dict:
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
