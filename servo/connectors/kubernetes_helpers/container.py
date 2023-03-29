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

import copy

from typing import Any, cast, Iterable, Optional

from kubernetes_asyncio.client import V1Container, V1EnvVar, V1ResourceRequirements

from servo.types.kubernetes import ResourceRequirement


class ContainerHelper:
    @classmethod
    def get_resource_requirements(
        cls, container: V1Container, resource_type: str
    ) -> dict[ResourceRequirement, Optional[str]]:
        """Return a dictionary mapping resource requirements to values for a given resource (e.g., cpu or memory).

        This method is safe to call for containers that do not define any resource requirements (e.g., the `resources` property is None).

        Requirements that are not defined for the named resource are returned as None. For example, a container
        that defines CPU requests but does not define limits would return a dict with a `None` value for
        the `ResourceRequirement.limit` key.

        Args:
            resource_type: The type of resource to get the requirements of (e.g., "cpu" or "memory").

        Returns:
            A dictionary mapping ResourceRequirement enum members to optional string values.
        """
        resources: V1ResourceRequirements = getattr(
            container, "resources", V1ResourceRequirements()
        )
        requirements = {}
        for requirement in ResourceRequirement:
            # Get the 'requests' or 'limits' nested structure
            requirement_subdict = getattr(resources, requirement.resources_key, {})
            if requirement_subdict:
                requirements[requirement] = requirement_subdict.get(resource_type)
            else:
                requirements[requirement] = None

        return requirements

    @classmethod
    def set_resource_requirements(
        cls,
        container: V1Container,
        resource_type: str,
        requirements: dict[ResourceRequirement, Optional[str]],
    ) -> None:
        """Sets resource requirements on the container for the values in the given dictionary.

        If no resources have been defined yet, a resources model is provisioned.
        If no requirements have been defined for the given resource name, a requirements dictionary is defined.
        Values of None are removed from the target requirements.
        ResourceRequirement keys that are not present in the dict are not modified.

        Args:
            resource_type: The name of the resource to set the requirements of (e.g., "cpu" or "memory").
            requirements: A dict mapping requirements to target values (e.g., `{ResourceRequirement.request: '500m', ResourceRequirement.limit: '2000m'})
        """
        resources: V1ResourceRequirements = copy.copy(
            getattr(container, "resources", V1ResourceRequirements())
        )

        for requirement, value in requirements.items():
            resource_to_values = getattr(resources, requirement.resources_key, {})
            if not resource_to_values:
                resource_to_values = {}

            if value is not None:
                # NOTE: Coerce to string as values are headed into Kubernetes resource model
                resource_to_values[resource_type] = str(value)
            else:
                resource_to_values.pop(resource_type, None)
            setattr(resources, requirement.resources_key, resource_to_values)

        container.resources = resources

    @classmethod
    def get_environment_variable(
        cls, container: V1Container, variable_name: str
    ) -> Optional[str]:
        if container.env:
            return next(
                iter(
                    v.value or f"valueFrom: {v.value_from}"
                    for v in cast(Iterable[V1EnvVar], container.env)
                    if v.name == variable_name
                ),
                None,
            )
        return None

    @classmethod
    def set_environment_variable(
        cls, container: V1Container, variable_name: str, value: Any
    ) -> None:
        # V1EnvVar value type is str so value will be converted eventually. Might as well do it up front
        val_str = str(value)
        if "valueFrom" in val_str:
            raise ValueError("Adjustment of valueFrom variables is not supported yet")

        new_vars: list[V1EnvVar] = container.env or []
        if new_vars:
            # Filter out vars with the same name as the ones we are setting
            new_vars = [v for v in new_vars if v.name != variable_name]

        new_vars.append(V1EnvVar(name=variable_name, value=val_str))
        container.env = new_vars
