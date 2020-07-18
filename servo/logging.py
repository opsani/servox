import logging
import sys
from pathlib import Path

from loguru import logger

root_path = Path(__file__).parents[1]

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

logger.configure(handlers=handlers)
