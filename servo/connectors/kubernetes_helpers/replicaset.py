# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
