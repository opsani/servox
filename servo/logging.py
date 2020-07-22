import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from weakref import WeakKeyDictionary

import httpx
from loguru import logger
from servo.types import Duration

class Mixin:
    @property
    def logger(self) -> logging.Logger:
        """Returns the logger"""
        return logger.bind(connector=self.name)

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


logger.configure(handlers=handlers)


class ProgressHandler(logging.Handler):
    def __init__(self, connector: 'Connector', **kwargs) -> None:        
        super().__init__(**kwargs)
        self.connector = connector
        self.queue = asyncio.Queue()

        asyncio.get_event_loop().create_task(self.run())
    
    async def run(self) -> None:
        while True:
            record, event_context = await self.queue.get()
            await self._emit_async(record, event_context)

    def emit(self, record: logging.LogRecord) -> None:
        if progress := record.extra.get("progress", None):
            with logger.catch(message="an exception occurred while reporting progress logging"):
                event_context = self.connector.current_event
                asyncio.get_event_loop().call_soon_threadsafe(self.queue.put_nowait, (record, event_context))

    async def _emit_async(self, record: logging.LogRecord, event_context: 'EventContext') -> None:        
        with logger.catch(message="an exception occurred while reporting progress logging", reraise=True):            
            progress = record.extra.get("progress", None)
            if not progress:
                logger.warning("declining request to report progress on a record without a progress attribute")
                return

            connector = record.extra.get("connector", None)
            if not connector:
                logger.warning("declining request to report progress for record without a connector attribute")
                return

            operation = record.extra.get("operation", None)
            if not operation:
                if not event_context:
                    logger.warning("declining request to report progress for record without an operation parameter or inferrable value from event context")
                    return
                operation = event_context.operation()

            started_at = record.extra.get("started_at", event_context.created_at)
            if not started_at:
                logger.warning("declining request to report progress for record without a started_at parameter or inferrable value from event context")
                return
            event = record.extra.get("event", str())

            request = self.connector.progress_request(
                operation=operation,
                progress=progress,
                connector=connector,
                event_context=event_context,
                started_at=started_at,
                message=record.msg,
            )
            await self.connector._post_event(*request)