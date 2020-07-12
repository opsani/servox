import os
from contextlib import contextmanager
from typing import Dict

from servo import connector
from servo.connector import Connector, ConnectorSettings
from typing import Optional


class StubConnectorSettings(ConnectorSettings):
    name: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> 'StubConnectorSettings':
        return cls(**kwargs)
        

class MeasureConnector(Connector):
    settings: StubConnectorSettings

    @connector.before_event('measure')
    def before_measure(self, *args, **kwargs):
        pass

    @connector.on_event()
    def measure(self, *args, **kwargs):
        pass

    @connector.after_event('measure')
    def after_measure(self, *args, **kwargs):
        pass


class AdjustConnector(Connector):
    settings: StubConnectorSettings

    @connector.on_event()
    def adjust(self, *args, **kwargs):
        pass


class LoadgenConnector(Connector):
    settings: StubConnectorSettings

    @connector.on_event()
    def loadgen(self, *args, **kwargs):
        pass

@contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env
