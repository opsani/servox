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

from typing import Mapping, Optional, Union

from kubernetes_asyncio.client import (
    V1Container,
    V1Pod,
    V1StatefulSet,
    V1Deployment,
    V1PodTemplateSpec,
)


def dict_to_selector(mapping: Mapping[str, str]) -> str:
    # https://stackoverflow.com/a/17888002
    return ",".join(["=".join((k, v)) for k, v in mapping.items()])


def get_containers(
    workload: Union[V1Pod, V1PodTemplateSpec, V1StatefulSet, V1Deployment]
) -> list[V1Container]:
    if isinstance(workload, (V1Pod, V1PodTemplateSpec)):
        return workload.spec.containers
    else:
        return workload.spec.template.spec.containers


# NOTE this method may need to become async if other workload types need their container object deserialized
#   by the kubernetes_asyncio.client
def find_container(
    workload: Union[V1Pod, V1PodTemplateSpec, V1StatefulSet, V1Deployment], name: str
) -> Optional[V1Container]:
    return next(
        iter((c for c in get_containers(workload=workload) if c.name == name)), None
    )
