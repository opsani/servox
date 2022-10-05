import contextlib
from typing import AsyncIterator

from kubernetes_asyncio.client import CoreV1Api, V1Namespace, ApiClient

from servo.logging import logger


class NamespaceHelper:
    @classmethod
    @contextlib.asynccontextmanager
    async def api_client(cls) -> AsyncIterator[CoreV1Api]:
        async with ApiClient() as api:
            yield CoreV1Api(api)

    @classmethod
    async def read(cls, name: str) -> V1Namespace:
        logger.debug(f'reading namespace "{name}"')
        async with cls.api_client() as api:
            return await api.read_namespace(name=name)
