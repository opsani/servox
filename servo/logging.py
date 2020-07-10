import sys
from pathlib import Path
import logging
from logging import Logger
from loguru import logger

root_path = Path(__file__).parents[1]

# logging configuration
handlers = [        
    { "sink": sys.stdout, "colorize": True, "filter": "app.asgi", "level": logging.DEBUG },
    { "sink": sys.stdout, "colorize": True, "filter": "uvicorn", "level": logging.DEBUG },
]

logs_path = root_path / 'logs' / f"servo.log"
handlers.append(
    { 
        "sink": logs_path, 
        "colorize": True, 
        "filter": {
            "servo": logging.DEBUG, 
            "tests": logging.DEBUG,
        },
        "backtrace": True, 
        "diagnose": True,
    }
)

logger.configure(
    handlers=handlers
)
