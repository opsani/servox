import logging
import sys
from weakref import WeakKeyDictionary
from pathlib import Path

import httpx
import loguru

class Mixin:
    @property
    def logger(self) -> logging.Logger:
        """Returns the logger"""
        return loguru.logger.bind(connector=self.name)

# logging configuration
CONNECTOR_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <magenta>{extra[connector]}</magenta> - <level>{message}</level>"
)

servo_levels = {
    "": "DEBUG",
    "servo": "INFO",
    "servo.connectors": False,
}

handlers = [
    {
        "sink": sys.stdout,
        "colorize": True,
        "filter": "servo.connectors",
        "level": logging.INFO,
        "format": CONNECTOR_FORMAT
    },
    {"sink": sys.stdout, "colorize": True, "filter": servo_levels},
]

root_path = Path(__file__).parents[1]
logs_path = root_path / "logs" / f"servo.log"
handlers.append(
    {
        "sink": logs_path,
        "colorize": True,
        "filter": {
            "servo": logging.DEBUG,
            "servo.connectors": logging.DEBUG,
            "tests": logging.DEBUG,
        },
        "backtrace": True,
        "diagnose": True,
    }
)

loguru.logger.configure(handlers=handlers)

class ProgressHandler(logging.Handler):
    def __init__(self, connector: 'Connector', **kwargs) -> None:
        self.connector = connector
        super().__init__(**kwargs)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # TODO: Call back to the connector class
        if progress := record.extra.get("progress", None):
            debug(self.format(record))
            # TODO: Include timestamp, connector name, type, version, message, line
            try:
                with self.connector.api_client_sync() as client:
                    client.post(f"http://localhost:8080?progress={progress}")
            except Exception as e:
                debug(e)
