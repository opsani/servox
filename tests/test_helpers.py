import os
from contextlib import contextmanager
from typing import Dict

from servo.connector import Connector, ConnectorSettings
from typing import Optional


class StubConnectorSettings(ConnectorSettings):
    name: Optional[str]


class MeasureConnector(Connector):
    settings: StubConnectorSettings
    pass


class AdjustConnector(Connector):
    settings: StubConnectorSettings
    pass


class LoadgenConnector(Connector):
    # settings: StubConnectorSettings
    pass


@contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env
