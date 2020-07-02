import os
from contextlib import contextmanager
from typing import Dict

from servo.connector import Connector


class MeasureConnector(Connector):
    pass


class AdjustConnector(Connector):
    pass


class LoadgenConnector(Connector):
    pass


@contextmanager
def environment_overrides(env: Dict[str, str]) -> None:
    original_env = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ = original_env
