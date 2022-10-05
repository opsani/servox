import abc
import devtools
from typing import Any, AsyncIterator, Optional

import kubernetes_asyncio.watch

from servo.logging import logger


class BaseKubernetesHelper(abc.ABC):
    @classmethod
    @abc.abstractmethod
    async def watch_args(cls, api_object: object) -> AsyncIterator[dict[str, Any]]:
        ...

    @classmethod
    @abc.abstractmethod
    def is_ready(cls, api_object: object, event_type: Optional[str] = None) -> bool:
        ...

    @classmethod
    async def wait_until_deleted(cls, api_object: object) -> None:
        async with cls.watch_args(api_object) as watch_args:
            async with kubernetes_asyncio.watch.Watch().stream(**watch_args) as stream:
                async for event in stream:
                    cls.log_watch_event(event)

                    if event["type"] == "DELETED":
                        stream.stop()
                        return

    @classmethod
    async def wait_until_ready(cls, api_object: object) -> None:
        async with cls.watch_args(api_object) as watch_args:
            async with kubernetes_asyncio.watch.Watch().stream(**watch_args) as stream:
                async for event in stream:
                    cls.log_watch_event(event)

                    if cls.is_ready(event["object"], event["type"]):
                        stream.stop()
                        return

    @classmethod
    def log_watch_event(cls, event: dict[str, Any]) -> None:
        event_type: str = event["type"]
        obj: dict = event["object"].to_dict()
        kind: str = obj.get("kind", "UNKNOWN")
        metadata = obj.get("metadata", {})
        name: str = metadata.get("name", "UNKNOWN")
        namespace: str = metadata.get("namespace", "UNKNOWN")
        logger.debug(
            f"watch yielded event: {event_type} on kind {kind} {name}"
            f" in namespace {namespace}"
        )
        logger.trace(devtools.pformat(obj))
