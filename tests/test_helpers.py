from contextlib import contextmanager
import os
from servo.connector import Connector
from typing import Dict

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
