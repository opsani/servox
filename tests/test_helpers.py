import os
from contextlib import contextmanager
from typing import Dict

from servo.connector import Connector, ConnectorSettings
from typing import Optional


class StubConnectorSettings(ConnectorSettings):
    name: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> 'StubConnectorSettings':
        return cls(**kwargs)

class MeasureConnector(Connector):
    settings: StubConnectorSettings


class AdjustConnector(Connector):
    settings: StubConnectorSettings


class LoadgenConnector(Connector):
    settings: StubConnectorSettings


@contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env
