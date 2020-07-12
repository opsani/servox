import os
from contextlib import contextmanager
from typing import Dict

from servo.servo import connector, Events
from servo.connector import Connector, ConnectorSettings
from servo.types import Measurement
from typing import Optional


class StubConnectorSettings(ConnectorSettings):
    name: Optional[str]

    @classmethod
    def generate(cls, **kwargs) -> 'StubConnectorSettings':
        return cls(**kwargs)
        

class MeasureConnector(Connector):
    settings: StubConnectorSettings

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
    settings: StubConnectorSettings

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
