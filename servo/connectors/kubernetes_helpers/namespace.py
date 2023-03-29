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
