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

import enum


class Resource(str, enum.Enum):
    memory = "memory"
    cpu = "cpu"

    @classmethod
    def values(cls) -> list[str]:
        """
        Return a list of strings that identifies all resource values.
        """
        return list(map(lambda rsrc: rsrc.value, cls.__members__.values()))


class ResourceRequirement(enum.Enum):
    """
    The ResourceRequirement enumeration determines how optimization values are submitted to the
    Kubernetes scheduler to allocate core compute resources. Requests establish the lower bounds
    of the CPU and memory necessary for an application to execute while Limits define the upper
    bounds for resources that can be consumed by a given Pod. The Opsani engine can determine
    optimal values for these settings by identifying performant, low cost configurations that meet
    target SLOs and/or maximizing performance while identifying the point of diminishing returns
    on further resourcing.
    """

    request = "request"
    limit = "limit"

    @property
    def resources_key(self) -> str:
        """
        Return a string value for accessing resource requirements within a Kubernetes Container representation.
        """
        if self == ResourceRequirement.request:
            return "requests"
        elif self == ResourceRequirement.limit:
            return "limits"
        else:
            raise NotImplementedError(
                f'missing resources_key implementation for resource requirement "{self}"'
            )


class ContainerLogOptions(str, enum.Enum):
    previous = "previous"
    current = "current"
    both = "both"
