import os
from contextlib import contextmanager
from typing import Dict, Optional

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
