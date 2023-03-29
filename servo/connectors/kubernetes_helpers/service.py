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
from typing import cast, AsyncIterator, Optional, Union

from kubernetes_asyncio.client import CoreV1Api, V1Service, V1ServicePort, ApiClient

from servo.logging import logger


class ServiceHelper:
    @classmethod
    @contextlib.asynccontextmanager
    async def api_client(cls) -> AsyncIterator[CoreV1Api]:
        async with ApiClient() as api:
            yield CoreV1Api(api)

    @classmethod
    async def read(cls, name: str, namespace: str) -> V1Service:
        logger.debug(f'reading service "{name}" in namespace {namespace}')
        async with cls.api_client() as api:
            return await api.read_namespaced_service(name=name, namespace=namespace)

    @classmethod
    async def patch(
        cls,
        workload: V1Service,
        api_client_default_headers: dict[str, str] = {
            "content-type": "application/strategic-merge-patch+json"
        },
    ) -> V1Service:
        name = workload.metadata.name
        namespace = workload.metadata.namespace
        logger.debug(f'patching service "{name}" in namespace "{namespace}"')
        async with cls.api_client() as api_client:
            # TODO: move up to baser class helper method
            for k, v in (api_client_default_headers or {}).items():
                api_client.api_client.set_default_header(k, v)

            return await api_client.patch_namespaced_service(
                name=name,
                namespace=namespace,
                body=workload,
            )

    @classmethod
    def find_port(
        cls, service: V1Service, selector: Union[str, int]
    ) -> Optional[V1ServicePort]:
        for port in cast(list[V1ServicePort], service.spec.ports):
            if isinstance(selector, str):
                if port.name == selector:
                    return port
            elif isinstance(selector, int):
                if port.port == selector:
                    return port
            else:
                raise TypeError(
                    f"Unknown port selector type '{selector.__class__.__name__}': {selector}"
                )

        return None
