import logging
import sys
from pathlib import Path

from loguru import logger

root_path = Path(__file__).parents[1]

# logging configuration
handlers = [
    {
        "sink": sys.stdout,
        "colorize": True,
        "filter": "connectors",
        "level": logging.INFO,
    },
    {"sink": sys.stdout, "colorize": True, "filter": "servo", "level": logging.INFO},
]

logs_path = root_path / "logs" / f"servo.log"
handlers.append(
    {
        "sink": logs_path,
        "colorize": True,
        "filter": {
            "servo": logging.DEBUG,
            "connectors": logging.DEBUG,
            "tests": logging.DEBUG,
        },
        "backtrace": True,
        "diagnose": True,
    }
)

logger.configure(handlers=handlers)
