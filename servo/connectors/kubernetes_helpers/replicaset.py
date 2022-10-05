import contextlib
from typing import AsyncIterator

from kubernetes_asyncio.client import (
    ApiClient,
    AppsV1Api,
    V1ReplicaSet,
    V1ReplicaSetList,
)

from servo.logging import logger
from .util import dict_to_selector


class ReplicasetHelper:
    @classmethod
    @contextlib.asynccontextmanager
    async def api_client(cls) -> AsyncIterator[AppsV1Api]:
        async with ApiClient() as api:
            yield AppsV1Api(api)

    @classmethod
    async def read(cls, name: str, namespace: str) -> V1ReplicaSet:
        logger.debug(f'reading replicaset "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api:
            return await api.read_namespaced_replica_set(name=name, namespace=namespace)

    @classmethod
    async def list_replicasets_with_labels(
        cls, namespace: str, match_labels: dict[str, str]
    ) -> list[V1ReplicaSet]:
        async with cls.api_client() as api:
            rs_list: V1ReplicaSetList = await api.list_namespaced_replica_set(
                namespace=namespace, label_selector=dict_to_selector(match_labels)
            )
            return rs_list.items or []


ReplicasetHelper()
