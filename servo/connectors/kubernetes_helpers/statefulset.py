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

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from kubernetes_asyncio.client import (
    ApiClient,
    AppsV1Api,
    V1StatefulSet,
    V1ObjectMeta,
    V1Pod,
)

from servo.logging import logger
from .base_workload import BaseKubernetesWorkloadHelper
from .pod import PodHelper
from .util import dict_to_selector


class StatefulSetHelper(BaseKubernetesWorkloadHelper):
    @classmethod
    @asynccontextmanager
    async def api_client(cls) -> AsyncIterator[AppsV1Api]:
        async with ApiClient() as api:
            yield AppsV1Api(api)

    @classmethod
    async def read(cls, name: str, namespace: str) -> V1StatefulSet:
        logger.debug(f'reading statefulset "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api:
            return await api.read_namespaced_stateful_set(
                name=name, namespace=namespace
            )

    @classmethod
    async def patch(
        cls,
        workload: V1StatefulSet,
        api_client_default_headers: dict[str, str] = {
            "content-type": "application/strategic-merge-patch+json"
        },
    ) -> V1StatefulSet:
        name = workload.metadata.name
        namespace = workload.metadata.namespace
        logger.debug(f'patching statefulset "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api_client:
            # TODO: move up to baser class helper method
            for k, v in (api_client_default_headers or {}).items():
                api_client.api_client.set_default_header(k, v)

            return await api_client.patch_namespaced_stateful_set(
                name=name,
                namespace=namespace,
                body=workload,
            )

    @classmethod
    @asynccontextmanager
    async def watch_args(cls, workload: V1StatefulSet) -> AsyncIterator[dict[str, Any]]:
        async with cls.api_client() as api:
            metadata: V1ObjectMeta = workload.metadata
            watch_args = {"func": api.list_namespaced_stateful_set}
            watch_args["namespace"] = metadata.namespace
            watch_args["label_selector"] = dict_to_selector(metadata.labels)
            watch_args["field_selector"] = dict_to_selector(
                {"metadata.name": metadata.name}
            )
            yield watch_args

    @classmethod
    def check_conditions(cls, workload: V1StatefulSet) -> None:
        # https://github.com/kubernetes/kubernetes/issues/79606
        raise NotImplementedError("StatefulSets do not define conditions")

    @classmethod
    async def get_latest_pods(cls, workload: V1StatefulSet) -> list[V1Pod]:
        # make copy of selector dict for safe updates
        pod_labels = dict(workload.spec.selector.match_labels)
        pod_labels["controller-revision-hash"] = workload.status.update_revision
        return await PodHelper.list_pods_with_labels(
            workload.metadata.namespace, pod_labels
        )
        # TODO? validate pod owner references?


# Run a dummy instantiation to detect missing ABC implementations
StatefulSetHelper()
