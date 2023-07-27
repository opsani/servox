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

"""Optimize services and applications deployed on Kubernetes with Opsani.
"""
from __future__ import annotations, print_function

import abc
import asyncio
import contextlib
import decimal
import enum
import functools
import itertools
import os
import pathlib
import pydantic
import re
from typing import (
    AsyncIterator,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import kubernetes_asyncio
import kubernetes_asyncio.client
import kubernetes_asyncio.client.api_client
import kubernetes_asyncio.client.exceptions
import kubernetes_asyncio.client.models
from kubernetes_asyncio.client import (
    V1Container,
    V1Deployment,
    V1EnvVar,
    V1OwnerReference,
    V1Pod,
    V1PodTemplateSpec,
    V1StatefulSet,
)

import servo
from servo.telemetry import ONE_MiB
from servo.types.kubernetes import *

from .kubernetes_helpers import (
    ContainerHelper,
    DeploymentHelper,
    PodHelper,
    NamespaceHelper,
    ReplicasetHelper,
    StatefulSetHelper,
    find_container,
)


class Core(decimal.Decimal):
    """
    The Core class represents one vCPU or hyperthread in Kubernetes.

    Supports the following format specification (note values must be an exact match, any other specificiers are
    fulfilled by the base Decimal class):
    - n - nanocores (default for values < 1 microcore)
    - u - microcores (default for values < 1 millicore)
    - m - millicores (default for values < 1 core)
    - c - cores (default for values > 1 core)
    """

    @classmethod
    def __get_validators__(cls) -> pydantic.types.CallableGenerator:
        yield cls.parse

    @classmethod
    def parse(cls, v: pydantic.types.StrIntFloat) -> "Core":
        """
        Parses a string, integer, or float input value into Core units.

        Returns:
            The input value in Core units.

        Raises:
            ValueError: Raised if the input cannot be parsed.
        """
        if isinstance(v, Core):
            return v
        # TODO lots of trailing zeros from this parsing
        elif isinstance(v, str):
            if v[-1] == "m":
                return cls(decimal.Decimal(str(v[:-1])) / 1000)
            # Metrics server API returns usage in microcores and nanocores
            elif v[-1] == "u":
                return cls(decimal.Decimal(str(v[:-1])) / 1000000)
            elif v[-1] == "n":
                return cls(decimal.Decimal(str(v[:-1])) / 1000000000)
            else:
                return cls(decimal.Decimal(str(v)))
        elif isinstance(v, (int, float, decimal.Decimal)):
            return cls(decimal.Decimal(str(v)))
        else:
            raise ValueError(f"could not parse Core value {v}")

    def __str__(self) -> str:
        return self.__format__()

    def __format__(self, specifier: str = None) -> str:
        if not specifier:
            specifier = "c"
            if self.microcores < 1:
                specifier = "n"
            elif self.millicores < 1:
                specifier = "u"
            elif self < 1:
                specifier = "m"

        value = decimal.Decimal(self)
        if specifier == "n":
            value = self.nanocores
        elif specifier == "u":
            value = self.microcores
        elif specifier == "m":
            value = self.millicores
        elif specifier == "c":
            specifier = ""
        else:
            return super().__format__(specifier)

        # strip the trailing zero and dot when present for consistent representation
        str_val = re.sub(r"\.0*$", repl="", string=str(value))
        return f"{str_val}{specifier}"

    def __eq__(self, other) -> bool:
        if isinstance(other, (Core, decimal.Decimal)):
            return decimal.Decimal(self) == decimal.Decimal(other)
        else:
            return self == Core.parse(other)

    def human_readable(self) -> str:
        return str(self)

    def __opsani_repr__(self) -> float:
        return float(self)

    @property
    def millicores(self) -> decimal.Decimal:
        return self * 1000

    @property
    def microcores(self) -> decimal.Decimal:
        return self * 1000000

    @property
    def nanocores(self) -> decimal.Decimal:
        return self * 1000000000


class CPU(servo.CPU):
    """
    The CPU class models a Kubernetes CPU resource in Core units.
    """

    min: Core
    max: Core
    step: Core
    value: Optional[Core]

    # Kubernetes resource requirements
    request: Optional[Core]
    limit: Optional[Core]
    get: list[ResourceRequirement] = pydantic.Field(
        default=[
            ResourceRequirement.request,
            ResourceRequirement.limit,
        ],
        min_items=1,
    )
    set: list[ResourceRequirement] = pydantic.Field(
        default=[
            ResourceRequirement.request,
            ResourceRequirement.limit,
        ],
        min_items=1,
    )

    def __opsani_repr__(self) -> dict:
        o_dict = super().__opsani_repr__()

        # Always return Core values in units of cores no matter how small
        for field in ("min", "max", "step", "value"):
            value: Optional[Core] = getattr(self, field)
            # TODO switch back to string for sending to API
            # o_dict["cpu"][field] = "{0:f}".format(value) if value is not None else None
            o_dict["cpu"][field] = (
                value.__opsani_repr__() if value is not None else None
            )

        return o_dict


# Gibibyte is the base unit of Kubernetes memory
MiB = 2**20
GiB = 2**30


class ShortByteSize(pydantic.ByteSize):
    """Kubernetes omits the 'B' suffix for some reason"""

    @classmethod
    def validate(cls, v: pydantic.StrIntFloat) -> "ShortByteSize":
        if isinstance(v, str):
            # Unitless decimals are not use by k8s API but are used in servo protocol implicitly as GiB
            if re.match(r"^\d*\.\d+$", v):
                v = f"{v}GiB"

            try:
                return super().validate(v)
            except:
                # Append the byte suffix and retry parsing
                return super().validate(v + "b")
        elif isinstance(v, float):
            # Unitless decimals are not use by k8s API but are used in servo protocol implicitly as GiB
            v = v * GiB
        return super().validate(v)

    def human_readable(self) -> str:
        """NOTE: only represents precision up to 1 decimal place (see pydantic's human_readable)"""
        sup = super().human_readable()
        # Remove the 'B' suffix to align with Kubernetes units (`GiB` -> `Gi`)
        if sup[-1] == "B" and sup[-2].isalpha():
            sup = sup[0:-1]
        return sup

    def __opsani_repr__(self) -> float:
        return float(decimal.Decimal(self) / GiB)

    def __str__(self) -> str:
        num = decimal.Decimal(self)
        units = ["B", "Ki", "Mi", "Gi", "Ti", "Pi"]
        for unit in units:
            if abs(num) < 1024:
                return f"{num:f}{unit}"
            num /= 1024

        return f"{num:f}Ei"


class Memory(servo.Memory):
    """
    The Memory class models a Kubernetes Memory resource.
    """

    min: ShortByteSize
    max: ShortByteSize
    step: ShortByteSize
    value: Optional[ShortByteSize]

    # Kubernetes resource requirements
    request: Optional[ShortByteSize]
    limit: Optional[ShortByteSize]
    get: list[ResourceRequirement] = pydantic.Field(
        default=[
            ResourceRequirement.request,
            ResourceRequirement.limit,
        ],
        min_items=1,
    )
    set: list[ResourceRequirement] = pydantic.Field(
        default=[
            ResourceRequirement.request,
            ResourceRequirement.limit,
        ],
        min_items=1,
    )

    def __opsani_repr__(self) -> dict:
        o_dict = super().__opsani_repr__()

        # normalize values into floating point Gibibyte units
        for field in ("min", "max", "step", "value"):
            value: Optional[ShortByteSize] = getattr(self, field)
            o_dict["mem"][field] = (
                value.__opsani_repr__() if value is not None else None
            )
        return o_dict


def _normalize_adjustment(
    adjustment: servo.Adjustment,
) -> Tuple[str, Union[str, servo.Numeric]]:
    """Normalize an adjustment object into a Kubernetes native setting key/value pair."""
    setting = "memory" if adjustment.setting_name == "mem" else adjustment.setting_name
    value = adjustment.value

    if setting == "memory":
        # Add GiB suffix to Numerics and Numeric strings
        if isinstance(value, (int, float)) or (
            isinstance(value, str) and value.replace(".", "", 1).isdigit()
        ):
            value = f"{value}Gi"
    elif setting == "cpu":
        core_value = Core.parse(value)
        if core_value % decimal.Decimal("0.001") != 0:
            raise ValueError(
                f"Kubernetes does not support CPU precision lower than 1m (one millicore). Found {value}"
            )
        value = str(core_value)
    elif setting == "replicas":
        value = int(float(value))
    # TODO support for env var format descriptor
    return setting, value


class BaseOptimization(abc.ABC, pydantic.BaseModel, servo.logging.Mixin):
    """
    BaseOptimization is the base class for concrete implementations of optimization strategies.

    Attributes:
        name (str): The name of the Optimization. Used to set the name for the corresponding component.
        timeout (Duration): Time interval to wait before considering Kubernetes operations to have failed.
        adjustments (List[Adjustment]): List of adjustments applied to this optimization (NOTE optimizations are re-created for each
            event dispatched to the connector. Thus, this value will only be populated during adjust event handling with only the adjustments
            pertaining to that adjust event dispatch)
    """

    name: str
    timeout: servo.Duration
    adjustments: List[servo.Adjustment] = []

    @abc.abstractclassmethod
    async def create(
        cls, config: "BaseKubernetesConfiguration", *args, **kwargs
    ) -> "BaseOptimization":
        """"""
        ...

    @abc.abstractmethod
    def adjust(
        self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()
    ) -> servo.Description:
        """
        Adjust a setting on the underlying Deployment/Pod or Container.
        """
        ...

    @abc.abstractmethod
    async def apply(self) -> None:
        """
        Apply the adjusted settings to the Kubernetes cluster.
        """
        ...

    @abc.abstractmethod
    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        ...

    @property
    @abc.abstractmethod
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior.
        """
        ...

    async def handle_error(self, error: Exception) -> bool:
        """
        Handle an operational failure in accordance with the failure mode configured by the operator.

        Well executed error handling requires context and strategic thinking. The servo base library
        provides a rich set of primitives and patterns for approaching error handling but ultimately
        the experience is reliant on the connector developer who has knowledge of the essential context
        and understands the user needs and expectations.

        The error handling implementation provided in this method handles the general cases out of the
        box and relies on abstract methods (see below) to implement more advanced behaviors such as
        rollback and tear-down.

        Returns:
            A boolean value that indicates if the error was handled.

        Raises:
            NotImplementedError: Raised if there is no handler for a given failure mode. Subclasses
                must filter failure modes before calling the superclass implementation.
        """
        # Ensure that we chain any underlying exceptions that may occur
        try:
            self.logger.error(
                f"handling error with with failure mode {self.on_failure}: {error.__class__.__name__} - {str(error)}"
            )
            self.logger.opt(exception=error).debug(f"kubernetes error details")

            if self.on_failure == FailureMode.exception:
                raise error

            elif self.on_failure == FailureMode.ignore:
                self.logger.opt(exception=error).warning(f"ignoring exception")
                return True

            elif self.on_failure == FailureMode.shutdown:
                await self.shutdown(error)

            else:
                # Trap any new modes that need to be handled
                raise NotImplementedError(
                    f"missing error handler for failure mode '{self.on_failure}'"
                )

            raise error  # Always communicate errors to backend unless ignored

        except Exception as handler_error:
            raise handler_error from error  # reraising an error from itself is safe

    @abc.abstractmethod
    async def shutdown(self, error: Optional[Exception] = None) -> None:
        """
        Asynchronously shut down the Optimization.

        Args:
            error: An optional exception that contextualizes the cause of the destruction.
        """
        ...

    @abc.abstractmethod
    def to_components(self) -> List[servo.Component]:
        """
        Return a list of Component representations of the Optimization.

        Components are the canonical representation of optimizations in the Opsani API.
        """
        ...

    @abc.abstractmethod
    async def is_ready(self) -> bool:
        """
        Verify Optimization target Resource/Controller is ready.
        """
        ...

    def __hash__(self):
        return hash(
            (
                self.name,
                id(self),
            )
        )

    class Config:
        arbitrary_types_allowed = True


class SaturationOptimization(BaseOptimization):
    """
    The SaturationOptimization class implements an optimization strategy based on directly reconfiguring a Kubernetes
    workload and its associated containers.
    """

    workload_helper: Optional[Union[Type[DeploymentHelper], Type[StatefulSetHelper]]]
    workload_config: Optional[
        Union["DeploymentConfiguration", "StatefulSetConfiguration"]
    ]
    workload: Optional[Union[V1Deployment, V1StatefulSet]]

    container_config: "ContainerConfiguration"
    container: V1Container

    @classmethod
    async def create(
        cls,
        config: Union["DeploymentConfiguration", "StatefulSetConfiguration"],
        **kwargs,
    ) -> "SaturationOptimization":
        # TODO switch for type of config
        if isinstance(config, StatefulSetConfiguration):
            workload_helper = StatefulSetHelper
        elif isinstance(config, DeploymentConfiguration):
            workload_helper = DeploymentHelper
        else:
            raise ValueError(
                f"Unrecognized workload for configuration type of {config.__class__.__name__}"
            )

        workload = await workload_helper.read(config.name, config.namespace)
        replicas = config.replicas.copy()
        # NOTE: Assign to the config to trigger validations
        replicas.value = workload.spec.replicas

        # FIXME: Currently only supporting one container
        for container_config in config.containers:
            container = find_container(workload=workload, name=container_config.name)
            if not container:
                names = servo.utilities.strings.join_to_series(
                    list(
                        map(
                            lambda c: c.metadata.name,
                            workload.spec.template.spec.containers,
                        )
                    )
                )
                raise ValueError(
                    f'no container named "{container_config.name}" exists in the Pod (found {names})'
                )

            if container_config.static_environment_variables:
                raise NotImplementedError(
                    "Configurable environment variables are not currently supported under Deployment optimization (saturation mode)"
                )

            name = container_config.alias or (
                f"{workload.metadata.name}/{container.name}"
                if container
                else workload.metadata.name
            )
            return cls(
                name=name,
                workload_config=config,
                workload=workload,
                workload_helper=workload_helper,
                container_config=container_config,
                container=container,
                **kwargs,
            )

    @property
    def cpu(self) -> CPU:
        """
        Return the current CPU setting for the optimization.
        """
        cpu = self.container_config.cpu.copy()

        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.container, "cpu"
        )
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.cpu.get,
                ),
                None,
            )
        )
        value = Core.parse(value)
        # NOTE: use safe_set to apply values that may be outside of the range
        return cpu.safe_set_value_copy(value)

    @property
    def memory(self) -> Memory:
        """
        Return the current Memory setting for the optimization.
        """
        memory = self.container_config.memory.copy()

        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.container, "memory"
        )
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.memory.get,
                ),
                None,
            )
        )
        value = ShortByteSize.validate(value)
        # NOTE: use safe_set to apply values that may be outside of the range
        return memory.safe_set_value_copy(value)

    @property
    def env(self) -> Optional[list[servo.EnvironmentSetting]]:
        env: list[servo.EnvironmentSetting] = []
        env_setting: Union[servo.EnvironmentRangeSetting, servo.EnvironmentEnumSetting]
        for env_setting in self.container_config.env or []:
            if env_val := ContainerHelper.get_environment_variable(
                self.container, env_setting.name
            ):
                env_setting = env_setting.safe_set_value_copy(env_val)
            env.append(env_setting)

        return env or None

    @property
    def replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the optimization.
        """
        replicas = self.workload_config.replicas.copy()
        replicas.value = self.workload.spec.replicas
        return replicas

    @property
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior. If not set explicitly, this will be cascaded
        from the base kubernetes configuration (or its default)
        """
        return self.workload_config.on_failure

    async def shutdown(self, error: Optional[Exception] = None) -> None:
        """
        Initiates the asynchronous deletion of all pods in the Deployment under optimization.

        Args:
            error: An optional error that triggered the destruction.
        """
        self.logger.info(f"adjustment failed: shutting down deployment's pods...")

        retries = 3
        while retries > 0:
            # patch the deployment
            try:
                self.workload = await self.workload_helper.read(
                    self.workload_config.name, self.workload_config.namespace
                )
                self.workload.spec.replicas = 0
                self.workload = await asyncio.wait_for(
                    self.workload_helper.patch(self.workload),
                    timeout=self.timeout.total_seconds(),
                )
            except kubernetes_asyncio.client.ApiException as ae:
                retries -= 1
                if retries == 0:
                    self.logger.error(
                        "Failed to shutdown SaturationOptimization after 3 retries"
                    )
                    raise

                if ae.status == 409 and ae.reason == "Conflict":
                    # If we have a conflict, just load the existing object and try again
                    pass
                else:
                    raise
            else:
                # No need to retry if no exception raised
                break

    def to_components(self) -> List[servo.Component]:
        settings = [self.cpu, self.memory, self.replicas]
        if env := self.env:
            settings.extend(env)
        return [servo.Component(name=self.name, settings=settings)]

    def adjust(
        self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()
    ) -> None:
        """
        Adjust the settings on the Deployment or a component Container.

        Adjustments do not take effect on the cluster until the `apply` method is invoked
        to enable aggregation of related adjustments and asynchronous application.
        """
        self.adjustments.append(adjustment)
        setting_name, value = _normalize_adjustment(adjustment)
        self.logger.info(f"adjusting {setting_name} to {value}")
        env_setting: Optional[
            servo.EnvironmentSetting
        ] = None  # Declare type since type not compatible with :=

        if setting_name in ("cpu", "memory"):
            # NOTE: use copy + update to apply values that may be outside of the range
            servo.logger.debug(f"Adjusting {setting_name}={value}")
            setting = getattr(self.container_config, setting_name).copy(
                update={"value": value}
            )

            # Set only the requirements defined in the config
            requirements: Dict[ResourceRequirement, Optional[str]] = {}
            for requirement in setting.set:
                requirements[requirement] = value

            ContainerHelper.set_resource_requirements(
                self.container, setting_name, requirements
            )

        elif setting_name == "replicas":
            # NOTE: Assign to the config to trigger validations
            self.workload_config.replicas.value = value
            self.workload.spec.replicas = value

        elif env_setting := servo.find_setting(self.container_config.env, setting_name):
            env_setting = env_setting.safe_set_value_copy(value)
            ContainerHelper.set_environment_variable(
                self.container, env_setting.variable_name, env_setting.value
            )

        else:
            raise RuntimeError(
                f"failed adjustment of unsupported Kubernetes setting '{adjustment.setting_name}'"
            )

    async def apply(self) -> None:
        """
        Apply changes asynchronously and wait for them to roll out to the cluster.

        Kubernetes deployments orchestrate a number of underlying resources. Awaiting the
        outcome of a deployment change requires observation of the `resource_version` which
        indicates if a given patch actually changed the resource, the `observed_generation`
        which is a value managed by the deployments controller and indicates the effective
        version of the deployment exclusive of insignificant changes that do not affect runtime
        (such as label updates), and the `conditions` of the deployment status which reflect
        state at a particular point in time. How these elements change during a rollout is
        dependent on the deployment strategy in effect and its requirements (max unavailable,
        surge, etc).

        The logic implemented by this method is as follows:
            - Capture the `resource_version` and `observed_generation`.
            - Patch the underlying Deployment object via the Kubernetes API.
            - Check that `resource_version` has been incremented or return early if nothing has changed.
            - Create a Kubernetes Watch on the Deployment targeted by label selector and resource version.
            - Observe events streamed via the watch.
            - Look for the Deployment to report a Status Condition of `"Progressing"`.
            - Wait for the `observed_generation` to increment indicating that the Deployment is applying our changes.
            - Track the value of the `available_replicas`, `ready_replicas`, `unavailable_replicas`,
                and `updated_replicas` attributes of the Deployment Status until `available_replicas`,
                `ready_replicas`, and `updated_replicas` are all equal to the value of the `replicas` attribute of
                the Deployment and `unavailable_replicas` is `None`. Return success.
            - Raise an error upon expiration of an adjustment timeout or encountering a Deployment Status Condition
                where `type=Progressing` and `status=False`.

        This method abstracts the details of adjusting a Deployment and returns once the desired
        changes have been fully rolled out to the cluster or an error has been encountered.

        See https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

        # The resource_version attribute lets us efficiently watch for changes
        # reference: https://kubernetes.io/docs/reference/using-api/api-concepts/#efficient-detection-of-changes
        """
        # Patch the Deployment via the Kubernetes API
        self.workload = await self.workload_helper.patch(self.workload)
        try:
            await asyncio.wait_for(
                self.workload_helper.wait_until_ready(self.workload),
                timeout=self.timeout.total_seconds(),
            )
        except asyncio.exceptions.TimeoutError:
            servo.logger.error(
                f"Timed out waiting for {self.workload.__class__.__name__} to become ready..."
            )
            await self.raise_for_status()
        servo.logger.success(
            f"adjustments to {self.workload.kind} '{self.workload.metadata.name}' rolled out successfully"
        )

    async def is_ready(self) -> bool:
        self.workload = await self.workload_helper.read(
            self.workload.metadata.name, self.workload.metadata.namespace
        )
        return (
            self.workload_helper.is_ready(self.workload)
            and await self.workload_helper.get_restart_count(self.workload) == 0
        )

    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        self.workload = await self.workload_helper.read(
            self.workload.metadata.name, self.workload.metadata.namespace
        )
        await self.workload_helper.raise_for_status(
            workload=self.workload,
            adjustments=self.adjustments,
            include_container_logs=self.workload_config.container_logs_in_error_status,
        )


class CanaryOptimization(BaseOptimization):
    """CanaryOptimization objects manage the optimization of Containers within a Deployment using
    a tuning Pod that is adjusted independently and compared against the performance and cost profile
    of its siblings.
    """

    # The helper static classes define the abstractions/interfaces for interacting with the various workload types
    # NOTE CanaryOptimization currently only supports Deployment
    workload_helper: Type[DeploymentHelper]

    # The deployment and container stanzas from the configuration
    workload_config: "DeploymentConfiguration"
    container_config: "ContainerConfiguration"

    # State for mainline resources. Read from the cluster
    workload: V1Deployment
    main_container: V1Container

    # State for tuning resources
    tuning_pod: Optional[V1Pod]
    tuning_container: Optional[V1Container]

    _tuning_pod_template_spec: Optional[V1PodTemplateSpec] = pydantic.PrivateAttr()

    @classmethod
    async def create(
        cls,
        workload_config: "DeploymentConfiguration",
        **kwargs,
    ) -> "CanaryOptimization":
        # NOTE may eventually support other workload types
        workload_helper: Type[DeploymentHelper] = None
        if isinstance(workload_config, DeploymentConfiguration):
            workload_helper = DeploymentHelper
        else:
            raise NotImplementedError(
                f"Unknown/incompatible configuration type '{workload_config.__class__.__name__}'"
            )

        workload = await workload_helper.read(
            name=workload_config.name, namespace=workload_config.namespace
        )

        # NOTE: Currently only supporting one container
        assert (
            len(workload_config.containers) == 1
        ), "CanaryOptimization currently only supports a single container"
        container_config = workload_config.containers[0]
        main_container: V1Container = find_container(
            workload=workload, name=container_config.name
        )

        alias = getattr(workload_config.strategy, "alias", None)
        name = (
            alias if alias else f"{workload_config.name}/{main_container.name}-tuning"
        )

        optimization = cls(
            name=name,
            workload_helper=workload_helper,
            workload_config=workload_config,
            workload=workload,
            container_config=container_config,
            main_container=main_container,
            **kwargs,
        )
        await optimization._load_tuning_state()
        await optimization._configure_tuning_pod_template_spec()

        return optimization

    async def _load_tuning_state(self) -> None:
        # Find an existing tuning Pod/Container if available
        try:
            tuning_pod = await PodHelper.read(self.tuning_pod_name, self.namespace)
            tuning_container = find_container(tuning_pod, self.container_config.name)

        except kubernetes_asyncio.client.exceptions.ApiException as e:
            if e.status != 404 or e.reason != "Not Found":
                servo.logger.trace(f"Failed reading tuning pod: {e}")
                raise
            else:
                tuning_pod = None
                tuning_container = None

        self.tuning_pod = tuning_pod
        self.tuning_container = tuning_container

    @property
    def pod_template_spec_container(self) -> V1Container:
        if not self._tuning_pod_template_spec:
            raise servo.EventError(
                "Cannot retrieve tuning container: tuning pod template spec not loaded"
            )
        return find_container(
            workload=self._tuning_pod_template_spec, name=self.container_config.name
        )

    def adjust(
        self, adjustment: servo.Adjustment, control: servo.Control = servo.Control()
    ) -> None:
        assert self.tuning_pod, "Tuning Pod not loaded"
        assert self.tuning_container, "Tuning Container not loaded"

        self.adjustments.append(adjustment)
        setting_name, value = _normalize_adjustment(adjustment)
        self.logger.info(f"adjusting {setting_name} to {value}")
        env_setting: Optional[
            servo.EnvironmentSetting
        ] = None  # Declare type since type not compatible with :=

        if setting_name in ("cpu", "memory"):
            # NOTE: use copy + update to apply values that may be outside of the range
            servo.logger.debug(f"Adjusting {setting_name}={value}")
            # NOTE copy is called from pydantic.BaseModel due to CPU/Memory setting chain of inheritance
            # https://github.com/pydantic/pydantic/blob/abd687700afe28745a3af5bca6f0f0ba48c86d1e/pydantic/main.py#L627
            setting: Union[CPU, Memory] = getattr(
                self.container_config, setting_name, pydantic.BaseModel
            ).copy(update={"value": value})

            # Set only the requirements defined in the config
            requirements: Dict[ResourceRequirement, Optional[str]] = {}
            for requirement in setting.set:
                requirements[requirement] = value
                servo.logger.debug(f"Assigning {setting_name}.{requirement}={value}")

            servo.logger.debug(
                f"Setting resource requirements for {setting_name} to {requirements} on PodTemplateSpec"
            )
            ContainerHelper.set_resource_requirements(
                self.pod_template_spec_container, setting_name, requirements
            )

        elif setting_name == "replicas":
            if value != 1:
                servo.logger.warning(f'ignored attempt to set replicas to "{value}"')

        elif env_setting := servo.find_setting(self.container_config.env, setting_name):
            env_setting = env_setting.safe_set_value_copy(value)
            ContainerHelper.set_environment_variable(
                self.pod_template_spec_container,
                env_setting.variable_name,
                env_setting.value,
            )

        else:
            raise servo.AdjustmentFailedError(
                f"failed adjustment of unsupported Kubernetes setting '{setting_name}'"
            )

    async def apply(self) -> None:
        """Apply the adjustments to the target."""
        assert self.tuning_pod, "Tuning Pod not loaded"
        assert self.tuning_container, "Tuning Container not loaded"

        servo.logger.info("Deleting existing tuning pod (if any)")
        await self.delete_tuning_pod(raise_if_not_found=False)

        servo.logger.info("Applying adjustments to Tuning Pod")
        await self.create_tuning_pod()

        servo.logger.success(
            f"Built new tuning pod with container resources: {self.tuning_container.resources}, env: {self.tuning_container.env}"
        )

    @property
    def namespace(self) -> str:
        return self.workload_config.namespace

    @property
    def tuning_pod_name(self) -> str:
        """
        Return the name of tuning Pod for this optimization.
        """
        return f"{self.workload_config.name}-tuning"

    async def delete_tuning_pod(
        self, *, raise_if_not_found: bool = True
    ) -> Optional[V1Pod]:
        """
        Delete the tuning Pod.
        """
        try:
            # TODO: Provide context manager or standard read option that handle not found? Lots of duplication on not found/conflict handling...
            tuning_pod = await PodHelper.read(self.tuning_pod_name, self.namespace)
            self.logger.info(
                f"Deleting tuning Pod '{tuning_pod.metadata.name}' from namespace '{tuning_pod.metadata.namespace}'..."
            )
            await PodHelper.delete(tuning_pod)
            await PodHelper.wait_until_deleted(tuning_pod)
            self.logger.info(
                f"Deleted tuning Pod '{tuning_pod.metadata.name}' from namespace '{tuning_pod.metadata.namespace}'."
            )

            self.tuning_pod = None
            self.tuning_container = None
            return tuning_pod

        except kubernetes_asyncio.client.exceptions.ApiException as e:
            if e.status != 404 or e.reason != "Not Found" or raise_if_not_found:
                raise

            self.logger.info(
                f"Ignoring delete tuning Pod '{self.tuning_pod_name}' from namespace '{self.namespace}' (pod not found)."
            )
            self.tuning_pod = None
            self.tuning_container = None

        return None

    async def _configure_tuning_pod_template_spec(self) -> None:
        # Configure a PodSpecTemplate for the tuning Pod state
        pod_template_spec = self.workload_helper.get_pod_template_spec_copy(
            self.workload
        )
        pod_template_spec.metadata.name = self.tuning_pod_name
        pod_template_spec.metadata.namespace = self.namespace

        if pod_template_spec.metadata.annotations is None:
            pod_template_spec.metadata.annotations = {}
        pod_template_spec.metadata.annotations[
            "opsani.com/opsani_tuning_for"
        ] = self.name
        if pod_template_spec.metadata.labels is None:
            pod_template_spec.metadata.labels = {}
        pod_template_spec.metadata.labels["opsani_role"] = "tuning"

        # Build a container from the raw podspec
        container = find_container(pod_template_spec, self.container_config.name)
        servo.logger.debug(
            f"Initialized new tuning container from Pod spec template: {container.name}"
        )

        if self.container_config.static_environment_variables:
            if container.env is None:
                container.env = []

            # Filter out vars with the same name as the ones we are setting
            container.env = [
                e
                for e in cast(list[V1EnvVar], container.env)
                if e.name not in self.container_config.static_environment_variables
            ]
            env_list = [
                V1EnvVar(name=k, value=v)
                for k, v in self.container_config.static_environment_variables.items()
            ]
            container.env.extend(env_list)

        if self.tuning_container:
            servo.logger.debug(
                "Copying resource requirements from existing tuning pod container"
                f" '{self.tuning_pod.metadata.name}/{self.tuning_container.name}'"
            )
            resource_requirements = self.tuning_container.resources
            container.resources = resource_requirements
        else:
            servo.logger.debug(
                f"No existing tuning pod container found, initializing resource requirement defaults"
            )
            set_container_resource_defaults_from_config(
                container, self.container_config
            )

        # If the servo is running inside Kubernetes, register self as the controller for the Pod and ReplicaSet
        servo_pod_name = os.environ.get("POD_NAME")
        servo_pod_namespace = os.environ.get("POD_NAMESPACE")
        if servo_pod_name is not None and servo_pod_namespace is not None:
            self.logger.debug(
                "running within Kubernetes, registering as Pod controller..."
                f" (pod={servo_pod_name}, namespace={servo_pod_namespace})"
            )

            # ephemeral, get its controller
            servo_pod = await PodHelper.read(servo_pod_name, servo_pod_namespace)

            pod_controller = next(
                iter(
                    ow
                    for ow in cast(
                        list[V1OwnerReference], servo_pod.metadata.owner_references
                    )
                    if ow.controller
                )
            )
            # still ephemeral
            servo_rs = await ReplicasetHelper.read(
                name=pod_controller.name, namespace=servo_pod_namespace
            )

            rs_controller = next(
                iter(
                    ow
                    for ow in cast(
                        list[V1OwnerReference], servo_rs.metadata.owner_references
                    )
                    if ow.controller
                )
            )
            # not ephemeral
            servo_dep = await DeploymentHelper.read(
                name=rs_controller.name, namespace=servo_pod_namespace
            )

            pod_template_spec.metadata.owner_references = [
                V1OwnerReference(
                    api_version=servo_dep.api_version,
                    block_owner_deletion=True,
                    controller=True,  # Ensures the pod will not be adopted by another controller
                    kind="Deployment",
                    name=servo_dep.metadata.name,
                    uid=servo_dep.metadata.uid,
                )
            ]

        self._tuning_pod_template_spec = pod_template_spec

    async def create_tuning_pod(self) -> V1Pod:
        """
        Creates a new Tuning Pod from the current optimization state.
        """
        assert self._tuning_pod_template_spec, "Must have tuning pod template spec"
        assert self.tuning_pod is None, "Tuning Pod already exists"
        assert self.tuning_container is None, "Tuning Pod Container already exists"
        self.logger.debug(
            f"creating tuning pod '{self.tuning_pod_name}' based on {self.workload.kind}"
            f" '{self.workload.metadata.name}' in namespace '{self.namespace}'"
        )

        # Setup the tuning Pod -- our settings are updated on the underlying PodSpec template
        self.logger.trace(f"building new tuning pod")
        pod_obj = V1Pod(
            metadata=self._tuning_pod_template_spec.metadata,
            spec=self._tuning_pod_template_spec.spec,
        )

        # TODO when supporting Argo rollout, must add rollout.status.current_pod_hash to pod labels
        #   under key "rollouts-pod-template-hash"

        # Create the Pod and wait for it to get ready
        self.logger.info(
            f"Creating tuning Pod '{self.tuning_pod_name}' in namespace '{self.namespace}'"
        )
        tuning_pod = await PodHelper.create(pod_obj)
        servo.logger.success(
            f"Created Tuning Pod '{self.tuning_pod_name}' in namespace '{self.namespace}'"
        )

        servo.logger.info(
            f"waiting up to {self.timeout} for Tuning Pod to become ready..."
        )
        progress = servo.EventProgress(self.timeout)
        progress_logger = lambda p: self.logger.info(
            p.annotate(
                f"waiting for '{self.tuning_pod_name}' to become ready...", prefix=False
            )
        )
        progress.start()

        task = asyncio.create_task(PodHelper.wait_until_ready(tuning_pod))
        task.add_done_callback(lambda _: progress.complete())
        gather_task = asyncio.gather(
            task,
            progress.watch(progress_logger),
        )

        try:
            await asyncio.wait_for(gather_task, timeout=self.timeout.total_seconds())

        except asyncio.TimeoutError:
            servo.logger.error(f"Timed out waiting for Tuning Pod to become ready...")
            servo.logger.debug(f"Cancelling Task: {task}, progress: {progress}")
            for t in {task, gather_task}:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
                    servo.logger.debug(f"Cancelled Task: {t}, progress: {progress}")

            # get latest status of tuning pod for raise_for_status
            await self.raise_for_status()

        # Hydrate local state
        await self._load_tuning_state()

        servo.logger.info(f"Tuning Pod successfully created")
        return tuning_pod

    @contextlib.asynccontextmanager
    async def temporary_tuning_pod(self) -> AsyncIterator[V1Pod]:
        """Mostly used for testing where automatic teardown is not available"""
        try:
            tuning_pod = await self.create_tuning_pod()
            yield tuning_pod
        finally:
            await self.delete_tuning_pod(raise_if_not_found=False)

    @property
    def tuning_cpu(self) -> Optional[CPU]:
        """
        Return the current CPU setting for the target container of the tuning Pod (if any).
        """
        if not self.tuning_pod:
            return None

        cpu = self.container_config.cpu.copy()

        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.tuning_container, Resource.cpu.value
        )
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.cpu.get,
                ),
                None,
            )
        )
        value = Core.parse(value)
        # NOTE: use safe_set to apply values that may be outside of the range
        return cpu.safe_set_value_copy(value)

    @property
    def tuning_memory(self) -> Optional[Memory]:
        """
        Return the current Memory setting for the target container of the tuning Pod (if any).
        """
        if not self.tuning_pod:
            return None

        memory = self.container_config.memory.copy()

        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.tuning_container, Resource.memory.value
        )
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.memory.get,
                ),
                None,
            )
        )
        value = ShortByteSize.validate(value)
        # NOTE: use safe_set to apply values that may be outside of the range
        memory = memory.safe_set_value_copy(value)
        return memory

    @property
    def tuning_env(self) -> Optional[list[servo.EnvironmentSetting]]:
        if not self.tuning_pod:
            return None

        env: list[servo.EnvironmentSetting] = []
        env_setting: Union[servo.EnvironmentRangeSetting, servo.EnvironmentEnumSetting]
        for env_setting in self.container_config.env or []:
            if env_val := ContainerHelper.get_environment_variable(
                self.tuning_container, env_setting.name
            ):
                env_setting = env_setting.safe_set_value_copy(env_val)
            env.append(env_setting)

        return env or None

    @property
    def tuning_replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the optimization.
        """
        value = 1 if self.tuning_pod else 0
        return servo.Replicas(
            min=0,
            max=1,
            value=value,
            pinned=True,
        )

    @property
    def on_failure(self) -> FailureMode:
        """
        Return the configured failure behavior. If not set explicitly, this will be cascaded
        from the base kubernetes configuration (or its default)
        """
        return self.workload_config.on_failure

    @property
    def main_cpu(self) -> CPU:
        """
        Return the current CPU setting for the main containers.
        """
        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.main_container, Resource.cpu.value
        )
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.cpu.get,
                ),
                None,
            )
        )
        cores = Core.parse(value)

        # NOTE: use safe_set to accept values from mainline outside of our range
        cpu: CPU = self.container_config.cpu.safe_set_value_copy(cores)
        cpu.pinned = True
        cpu.request = resource_requirements.get(ResourceRequirement.request)
        cpu.limit = resource_requirements.get(ResourceRequirement.limit)
        return cpu

    @property
    def main_memory(self) -> Memory:
        """
        Return the current Memory setting for the main containers.
        """
        # Determine the value in priority order from the config
        resource_requirements = ContainerHelper.get_resource_requirements(
            self.main_container, Resource.memory.value
        )
        value = resource_requirements.get(
            next(
                filter(
                    lambda r: resource_requirements[r] is not None,
                    self.container_config.memory.get,
                ),
                None,
            )
        )
        short_byte_size = ShortByteSize.validate(value)

        # NOTE: use safe_set to accept values from mainline outside of our range
        memory: Memory = self.container_config.memory.safe_set_value_copy(value)
        memory.pinned = True
        memory.request = resource_requirements.get(ResourceRequirement.request)
        memory.limit = resource_requirements.get(ResourceRequirement.limit)
        return memory

    @property
    def main_env(self) -> list[servo.EnvironmentSetting]:
        env: list[servo.EnvironmentSetting] = []
        env_setting: Union[servo.EnvironmentRangeSetting, servo.EnvironmentEnumSetting]
        for env_setting in self.container_config.env or []:
            if env_val := ContainerHelper.get_environment_variable(
                self.main_container, env_setting.name
            ):
                env_setting = env_setting.safe_set_value_copy(env_val)
            env_setting.pinned = True
            env.append(env_setting)

        return env or None

    @property
    def main_replicas(self) -> servo.Replicas:
        """
        Return the current Replicas setting for the main Pods Deployment.

        NOTE: This is a synthetic setting because the replica count of the main Deployment is not
        under out control. The min, max, and value are aligned on each synthetic read.
        """
        return servo.Replicas(
            min=0,
            max=99999,
            value=self.workload.spec.replicas,
            pinned=True,
        )

    @property
    def main_name(self) -> str:
        """Return the name for identifying the main instance settings & metrics.

        The name respects the alias defined in the config or else synthesizes a name from the Deployment
        and Container names.
        """
        return (
            self.container_config.alias
            or f"{self.workload_config.name}/{self.container_config.name}"
        )

    def to_components(self) -> List[servo.Component]:
        """
        Return a Component representation of the canary and its reference target.

        Note that all settings on the target are implicitly pinned because only the canary
        is to be modified during optimization.
        """
        main_settings = [
            self.main_cpu,
            self.main_memory,
            self.main_replicas,
        ]
        if main_env := self.main_env:
            main_settings.extend(main_env)
        tuning_settings = [
            self.tuning_cpu,
            self.tuning_memory,
            self.tuning_replicas,
        ]
        if tuning_env := self.tuning_env:
            tuning_settings.extend(tuning_env)
        return [
            servo.Component(name=self.main_name, settings=main_settings),
            servo.Component(name=self.name, settings=tuning_settings),
        ]

    async def destroy(self, error: Optional[Exception] = None) -> None:
        if await self.delete_tuning_pod(raise_if_not_found=False) is None:
            self.logger.debug(f"no tuning pod exists, ignoring destroy")
            return

        self.logger.success(f'destroyed tuning Pod "{self.tuning_pod_name}"')

    async def shutdown(self, error: Optional[Exception] = None) -> None:
        await self.destroy(error)

    async def handle_error(self, error: Exception) -> bool:
        if self.on_failure == FailureMode.shutdown:
            # Ensure that we chain any underlying exceptions that may occur
            try:
                try:
                    await asyncio.wait_for(
                        self.shutdown(), timeout=self.timeout.total_seconds()
                    )
                except asyncio.exceptions.TimeoutError:
                    self.logger.exception(level="TRACE")
                    raise RuntimeError(
                        f"Time out after {self.timeout} waiting for tuning pod shutdown"
                    )

                # create a new canary against baseline
                self.logger.info(
                    "creating new tuning pod against baseline following failed adjust"
                )
                await self._configure_tuning_pod_template_spec()  # reset to baseline from the target controller
                self.tuning_pod = await self.create_tuning_pod()

                raise error  # Always communicate errors to backend unless ignored

            except Exception as handler_error:
                raise handler_error from error

        else:
            return await super().handle_error(error)

    async def is_ready(self) -> bool:
        # Refresh pod state
        self.tuning_pod = await PodHelper.read(
            self.tuning_pod.metadata.name, self.tuning_pod.metadata.namespace
        )
        return (
            PodHelper.is_ready(self.tuning_pod)
            and PodHelper.get_restart_count(self.tuning_pod) == 0
        )

    async def raise_for_status(self) -> None:
        """Raise an exception if in an unhealthy state."""
        self.tuning_pod = await PodHelper.read(self.tuning_pod_name, self.namespace)
        await PodHelper.raise_for_status(
            self.tuning_pod,
            adjustments=self.adjustments,
            include_container_logs=self.workload_config.container_logs_in_error_status,
        )

    class Config:
        arbitrary_types_allowed = True
        extra = pydantic.Extra.forbid


class KubernetesOptimizations(pydantic.BaseModel, servo.logging.Mixin):
    """
    Models the state of resources under optimization in a Kubernetes cluster.
    """

    config: "KubernetesConfiguration"
    optimizations: List[BaseOptimization]
    runtime_id: str
    spec_id: str
    version_id: str

    @classmethod
    async def create(
        cls, config: "KubernetesConfiguration"
    ) -> "KubernetesOptimizations":
        """
        Read the state of all components under optimization from the cluster and return an object representation.
        """
        optimizations: List[BaseOptimization] = []
        images = {}
        runtime_ids = {}
        pod_tmpl_specs = {}

        # TODO rename varname to workload_configs
        for workload_config in config.workloads:
            if workload_config.strategy == OptimizationStrategy.default:
                optimization = await SaturationOptimization.create(
                    workload_config,
                    timeout=workload_config.timeout,
                )
                workload = optimization.workload
                container = optimization.container
            elif workload_config.strategy == OptimizationStrategy.canary:
                optimization = await CanaryOptimization.create(
                    workload_config,
                    timeout=workload_config.timeout,
                )
                workload = optimization.workload
                container = optimization.main_container

                # Ensure the canary is available
                # TODO: We don't want to do this implicitly but this is a first step
                if not optimization.tuning_pod:
                    servo.logger.info("Creating new tuning pod...")
                    await optimization.create_tuning_pod()
            else:
                raise ValueError(
                    f"unknown optimization strategy: {workload_config.strategy}"
                )

            optimizations.append(optimization)

            # compile artifacts for checksum calculation
            pods = await PodHelper.list_pods_with_labels(
                workload.metadata.namespace, workload.spec.selector.match_labels
            )
            runtime_ids[optimization.name] = [pod.metadata.uid for pod in pods]
            pod_tmpl_specs[workload.metadata.name] = workload.spec.template.spec
            images[container.name] = container.image

        # Compute checksums for change detection
        spec_id = servo.utilities.hashing.get_hash(
            [pod_tmpl_specs[k] for k in sorted(pod_tmpl_specs.keys())]
        )
        runtime_id = servo.utilities.hashing.get_hash(runtime_ids)
        version_id = servo.utilities.hashing.get_hash(
            [images[k] for k in sorted(images.keys())]
        )

        return KubernetesOptimizations(
            config=config,
            optimizations=optimizations,
            spec_id=spec_id,
            runtime_id=runtime_id,
            version_id=version_id,
        )

    def to_components(self) -> List[servo.Component]:
        """
        Return a list of Component objects modeling the state of local optimization activities.

        Components are the canonical representation of systems under optimization. They
        are used for data exchange with the Opsani API
        """
        components = list(map(lambda opt: opt.to_components(), self.optimizations))
        return list(itertools.chain(*components))

    def to_description(self) -> servo.Description:
        """
        Return a representation of the current state as a Description object.

        Description objects are used to report state to the Opsani API in order
        to synchronize with the Optimizer service.

        Returns:
            A Description of the current state.
        """
        return servo.Description(components=self.to_components())

    def find_optimization(self, name: str) -> Optional[BaseOptimization]:
        """
        Find and return an optimization by name.
        """
        return next(filter(lambda a: a.name == name, self.optimizations), None)

    async def apply(self, adjustments: List[servo.Adjustment]) -> None:
        """
        Apply a sequence of adjustments and wait for them to take effect on the cluster.
        """
        # Exit early if there is nothing to do
        if not adjustments:
            self.logger.debug("early exiting from adjust: no adjustments")
            return

        summary = f"[{', '.join(list(map(str, adjustments)))}]"
        self.logger.info(
            f"Applying {len(adjustments)} Kubernetes adjustments: {summary}"
        )

        # Adjust settings on the local data model
        for adjustment in adjustments:
            if adjustable := self.find_optimization(adjustment.component_name):
                self.logger.info(f"adjusting {adjustment.component_name}: {adjustment}")
                adjustable.adjust(adjustment)

            else:
                self.logger.debug(f'ignoring unrecognized adjustment "{adjustment}"')

        # Apply the changes to Kubernetes and wait for the results
        timeout = self.config.timeout
        if self.optimizations:
            self.logger.debug(
                f"waiting for adjustments to take effect on {len(self.optimizations)} optimizations"
            )
            try:
                gather_apply = asyncio.gather(
                    *list(map(lambda a: a.apply(), self.optimizations)),
                    return_exceptions=True,
                )
                results = await asyncio.wait_for(
                    gather_apply, timeout=timeout.total_seconds() + 60
                )  # allow sub-optimization timeouts to expire first

            except asyncio.exceptions.TimeoutError as error:
                self.logger.error(
                    f"timed out after {timeout} + 60s waiting for adjustments to apply"
                )
                # Prevent "_GatheringFuture exception was never retrieved" warning if the above wait_for raises a timeout error
                # https://bugs.python.org/issue29432
                try:
                    await gather_apply
                except asyncio.CancelledError:
                    pass
                for optimization in self.optimizations:
                    if await optimization.handle_error(error):
                        # Stop error propagation once it has been handled
                        break
                raise  # No results to process in this case, reraise timeout if handlers didn't

            for result in results:
                if isinstance(result, Exception):
                    for optimization in self.optimizations:
                        if await optimization.handle_error(result):
                            # Stop error propagation once it has been handled
                            break
        else:
            self.logger.warning(f"failed to apply adjustments: no adjustables")

        # TODO: Run sanity checks to look for out of band changes

    async def raise_for_status(self) -> None:
        handle_error_tasks = []

        def _raise_for_task(task: asyncio.Task, optimization: BaseOptimization) -> None:
            if task.done() and not task.cancelled():
                if exception := task.exception():
                    handle_error_tasks.append(
                        asyncio.create_task(optimization.handle_error(exception))
                    )

        tasks = []
        for optimization in self.optimizations:
            task = asyncio.create_task(optimization.raise_for_status())
            task.add_done_callback(
                functools.partial(_raise_for_task, optimization=optimization)
            )
            tasks.append(task)

        for future in asyncio.as_completed(
            tasks, timeout=self.config.timeout.total_seconds()
        ):
            try:
                await future
            except Exception as error:
                servo.logger.exception(f"Optimization failed with error: {error}")

        # TODO: first handler to raise will likely interrupt other tasks.
        #   Gather with return_exceptions=True and aggregate resulting exceptions before raising
        await asyncio.gather(*handle_error_tasks)

    async def is_ready(self):
        if self.optimizations:
            self.logger.debug(
                f"Checking for readiness of {len(self.optimizations)} optimizations"
            )
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *list(map(lambda a: a.is_ready(), self.optimizations)),
                    ),
                    timeout=self.config.timeout.total_seconds(),
                )

                return all(results)

            except asyncio.TimeoutError:
                return False

        else:
            return True

    class Config:
        arbitrary_types_allowed = True


DNSSubdomainName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=253,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z\\.-])*[0-9A-Za-z]$",
)
DNSSubdomainName.__doc__ = """DNSSubdomainName models a Kubernetes DNS Subdomain Name used as the name for most resource types.

    Valid DNS Subdomain Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-' or '.'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
    """


DNSLabelName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=63,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$",
)
DNSLabelName.__doc__ = """DNSLabelName models a Kubernetes DNS Label Name identified used to name some resource types.

    Valid DNS Label Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain at most 63 characters
        * contain only lowercase alphanumeric characters or '-'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-label-names
    """


ContainerTagName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=128,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z_\\.\\-/:@])*$",
)  # NOTE: This regex is not a full validation
ContainerTagName.__doc__ = """ContainerTagName models the name of a container referenced in a Kubernetes manifest.

    Valid container tags must:
        * be valid ASCII and may contain lowercase and uppercase letters, digits, underscores, periods and dashes.
        * not start with a period or a dash
        * may contain a maximum of 128 characters
    """


class ContainerConfiguration(servo.BaseConfiguration):
    """
    The ContainerConfiguration class models the configuration of an optimizeable container within a Kubernetes Deployment.
    """

    name: ContainerTagName
    alias: Optional[ContainerTagName]
    command: Optional[str]  # TODO: create model...
    cpu: CPU
    memory: Memory
    env: Optional[servo.EnvironmentSettingList]
    static_environment_variables: Optional[Dict[str, str]]


class OptimizationStrategy(str, enum.Enum):
    """
    OptimizationStrategy is an enumeration of the possible ways to perform optimization on a Kubernetes Deployment.
    """

    default = "default"
    """The default strategy directly applies adjustments to the target Deployment and its containers.
    """

    canary = "canary"
    """The canary strategy creates a servo managed standalone tuning Pod based on the target Deployment and makes
    adjustments to it instead of the Deployment itself.
    """


class BaseOptimizationStrategyConfiguration(pydantic.BaseModel):
    type: OptimizationStrategy = pydantic.Field(..., const=True)

    def __eq__(self, other) -> bool:
        if isinstance(other, OptimizationStrategy):
            return self.type == other
        return super().__eq__(other)

    class Config:
        extra = pydantic.Extra.forbid


class DefaultOptimizationStrategyConfiguration(BaseOptimizationStrategyConfiguration):
    type = pydantic.Field(OptimizationStrategy.default, const=True)


class CanaryOptimizationStrategyConfiguration(BaseOptimizationStrategyConfiguration):
    type = pydantic.Field(OptimizationStrategy.canary, const=True)
    alias: Optional[ContainerTagName]


class FailureMode(str, enum.Enum):
    """
    The FailureMode enumeration defines how to handle a failed adjustment of a Kubernetes resource.
    """

    shutdown = "shutdown"
    ignore = "ignore"
    exception = "exception"

    destroy = "destroy"  # deprecated, but accepted as "shutdown"

    @classmethod
    def options(cls) -> List[str]:
        """
        Return a list of strings that identifies all failure mode configuration options.
        """
        return list(map(lambda mode: mode.value, cls.__members__.values()))


class PermissionSet(pydantic.BaseModel):
    """Permissions objects model Kubernetes permissions granted through RBAC."""

    group: str
    resources: List[str]
    verbs: List[str]


STANDARD_PERMISSIONS = [
    PermissionSet(
        group="apps",
        resources=["deployments", "replicasets"],
        verbs=["get", "list", "watch", "update", "patch"],
    ),
    PermissionSet(
        group="",
        resources=["namespaces"],
        verbs=["get"],
    ),
    PermissionSet(
        group="",
        resources=["pods", "pods/logs", "pods/status"],
        verbs=["create", "delete", "get", "list", "watch"],
    ),
]

STATEFULSET_PERMISSIONS = [
    PermissionSet(
        group="apps",
        resources=["statefulsets"],
        verbs=["get", "list", "watch", "update", "patch"],
    ),
]


class BaseKubernetesConfiguration(servo.BaseConfiguration):
    """
    BaseKubernetesConfiguration provides a set of configuration primitives for optimizable Kubernetes resources.

    Child classes of `BaseKubernetesConfiguration` such as the `DeploymentConfiguration` can benefit from
    the cascading configuration behavior implemented on the `KubernetesConfiguration` class.

    Common settings will be cascaded from the containing class for attributes if they have not been explicitly set
    and are equal to the default value. Settings that are mandatory in the superclass (such as timeout and namespace)
    but are available for override should be declared as optional on `BaseKubernetesConfiguration` and overridden and
    declared as mandatory in `BaseKubernetesConfiguration`'.
    """

    kubeconfig: Optional[pydantic.FilePath] = pydantic.Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = pydantic.Field(
        description="Name of the kubeconfig context to use."
    )
    namespace: Optional[DNSSubdomainName] = pydantic.Field(
        description="Kubernetes namespace where the target deployments are running.",
    )
    settlement: Optional[servo.Duration] = pydantic.Field(
        description="Duration to observe the application after an adjust to ensure the deployment is stable. May be overridden by optimizer supplied `control.adjust.settlement` value."
    )
    on_failure: FailureMode = pydantic.Field(
        FailureMode.exception,
        description=f"How to handle a failed adjustment. Options are: {servo.utilities.strings.join_to_series(list(FailureMode.__members__.values()))}",
    )
    timeout: Optional[servo.Duration] = pydantic.Field(
        description="Time interval to wait before considering Kubernetes operations to have failed."
    )
    container_logs_in_error_status: bool = pydantic.Field(
        False, description="Enable to include container logs in error message"
    )
    create_tuning_pod: bool = pydantic.Field(
        True,
        description="Disable to prevent a canary strategy with tuning pod adjustments",
    )

    @pydantic.validator("on_failure")
    def validate_failure_mode(cls, v):
        if v == FailureMode.destroy:
            servo.logger.warning(
                f"Deprecated value 'destroy' used for 'on_failure', replacing with 'shutdown'"
            )
            return FailureMode.shutdown
        return v


StrategyTypes = Union[
    OptimizationStrategy,
    DefaultOptimizationStrategyConfiguration,
    CanaryOptimizationStrategyConfiguration,
]


class DeploymentConfiguration(BaseKubernetesConfiguration):
    """
    The DeploymentConfiguration class models the configuration of an optimizable Kubernetes Deployment.
    """

    name: DNSSubdomainName
    containers: List[ContainerConfiguration]
    strategy: StrategyTypes = OptimizationStrategy.default
    replicas: servo.Replicas


class StatefulSetConfiguration(DeploymentConfiguration):
    @pydantic.validator("strategy")
    def validate_strategy(cls, v):
        if v == OptimizationStrategy.canary:
            raise NotImplementedError(
                "Canary mode is not currently supported on StatefulSets"
            )
        return v


class KubernetesConfiguration(BaseKubernetesConfiguration):
    namespace: DNSSubdomainName = DNSSubdomainName("default")
    timeout: servo.Duration = "5m"
    permissions: List[PermissionSet] = pydantic.Field(
        STANDARD_PERMISSIONS,
        description="Permissions required by the connector to operate in Kubernetes.",
    )

    # TODO streamlining with a 'workloads' property name as the these three are used for the same purpose. Their
    # differences are a k8s implementation detail, not relevant to servox beyond the variance in API calls
    stateful_sets: Optional[List[StatefulSetConfiguration]] = pydantic.Field(
        description="StatefulSets to be optimized.",
    )

    deployments: Optional[List[DeploymentConfiguration]] = pydantic.Field(
        description="Deployments to be optimized.",
    )

    @property
    def workloads(
        self,
    ) -> list[Union[StatefulSetConfiguration, DeploymentConfiguration]]:
        return (self.deployments or []) + (self.stateful_sets or [])

    @pydantic.root_validator
    def check_workload(cls, values):
        if (not values.get("deployments")) and (
            not values.get("rollouts") and (not values.get("stateful_sets"))
        ):
            raise ValueError("No optimization target(s) were specified")
        return values

    @classmethod
    def generate(cls, **kwargs) -> "KubernetesConfiguration":
        return cls(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                DeploymentConfiguration(
                    name="app",
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        ContainerConfiguration(
                            name="opsani/fiber-http:latest",
                            cpu=CPU(min="250m", max=4, step="125m"),
                            memory=Memory(min="256MiB", max="4GiB", step="128MiB"),
                        )
                    ],
                )
            ],
            **kwargs,
        )

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        super().__init__(*args, **kwargs)
        self.cascade_common_settings()

    def cascade_common_settings(self, *, overwrite: bool = False) -> None:
        """
        Apply common settings to child models that inherit from BaseKubernetesConfiguration.

        This method provides enables hierarchical overrides of common configuration values
        based on shared inheritance. Each attribute is introspected and if it inherits from
        `BaseKubernetesConfiguration`, any common attribute values are copied onto the child
        model, cascading them downward. Only attributes whose value is equal to the default
        and have not been explicitly set are updated.

        # FIXME: Cascaded settings should only be optional if they can be optional at the top level. Right now we are implying that namespace can be None as well.
        """
        for name, field in self.__fields__.items():
            if issubclass(field.type_, BaseKubernetesConfiguration):
                attribute = getattr(self, name)
                for obj in (
                    attribute if isinstance(attribute, Collection) else [attribute]
                ):
                    # don't cascade if optional and not set
                    if obj is None:
                        continue
                    for (
                        field_name,
                        field,
                    ) in BaseKubernetesConfiguration.__fields__.items():
                        if field_name in servo.BaseConfiguration.__fields__:
                            # don't cascade from the base class
                            continue

                        if field_name in obj.__fields_set__ and not overwrite:
                            self.logger.trace(
                                f"skipping config cascade for field '{field_name}' set with value '{getattr(obj, field_name)}'"
                            )
                            continue

                        current_value = getattr(obj, field_name)
                        if overwrite or current_value == field.default:
                            parent_value = getattr(self, field_name)
                            setattr(obj, field_name, parent_value)
                            self.logger.trace(
                                f"cascaded setting '{field_name}' from KubernetesConfiguration to child '{attribute}': value={parent_value}"
                            )

                        else:
                            self.logger.trace(
                                f"declining to cascade value to field '{field_name}': the default value is set and overwrite is false"
                            )

    async def load_kubeconfig(self) -> None:
        """
        Asynchronously load the Kubernetes configuration
        """
        config_file = pathlib.Path(
            self.kubeconfig
            or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION
        ).expanduser()
        if config_file.exists():
            await kubernetes_asyncio.config.load_kube_config(
                config_file=str(config_file),
                context=self.context,
            )
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            kubernetes_asyncio.config.load_incluster_config()
        else:
            raise RuntimeError(
                f"unable to configure Kubernetes client: no kubeconfig file nor in-cluster environment variables found"
            )


KubernetesOptimizations.update_forward_refs()
SaturationOptimization.update_forward_refs()
CanaryOptimization.update_forward_refs()


class KubernetesChecks(servo.BaseChecks):
    """Checks for ensuring that the Kubernetes connector is ready to run."""

    config: KubernetesConfiguration

    @servo.require("Connectivity to Kubernetes")
    async def check_kubernetes_connectivity(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            await v1.get_code()

    @servo.warn("Kubernetes version")
    async def check_kubernetes_version(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            version = await v1.get_code()
            assert int(version.major) >= 1
            # EKS sets minor to "17+"
            assert int(int("".join(c for c in version.minor if c.isdigit()))) >= 16

    @servo.require("Required permissions")
    async def check_kubernetes_permissions(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            required_permissions = self.config.permissions
            if self.config.stateful_sets:
                required_permissions.extend(STATEFULSET_PERMISSIONS)
            # TODO stateful_set permissions
            for permission in required_permissions:
                for resource in permission.resources:
                    for verb in permission.verbs:
                        attributes = (
                            kubernetes_asyncio.client.models.V1ResourceAttributes(
                                namespace=self.config.namespace,
                                group=permission.group,
                                resource=resource,
                                verb=verb,
                            )
                        )

                        spec = kubernetes_asyncio.client.models.V1SelfSubjectAccessReviewSpec(
                            resource_attributes=attributes
                        )
                        review = (
                            kubernetes_asyncio.client.models.V1SelfSubjectAccessReview(
                                spec=spec
                            )
                        )
                        access_review = await v1.create_self_subject_access_review(
                            body=review
                        )
                        assert (
                            access_review.status.allowed
                        ), f'Not allowed to "{verb}" resource "{resource}"'

    @servo.require('Namespace "{self.config.namespace}" is readable')
    async def check_kubernetes_namespace(self) -> None:
        await NamespaceHelper.read(self.config.namespace)

    @servo.multicheck('Deployment "{item.name}" is readable')
    async def check_kubernetes_deployments(self) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_dep(dep_config: DeploymentConfiguration) -> None:
            await DeploymentHelper.read(dep_config.name, dep_config.namespace)

        return (self.config.deployments or []), check_dep

    @servo.multicheck('StatefulSet "{item.name}" is readable')
    async def check_kubernetes_statefulsets(
        self,
    ) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_ss(ss_config: StatefulSetConfiguration) -> None:
            await StatefulSetHelper.read(ss_config.name, ss_config.namespace)

        return (self.config.stateful_sets or []), check_ss

    async def _check_container_resource_requirements(
        self,
        target_controller: Union[V1Deployment, V1StatefulSet],
        target_config: Union[DeploymentConfiguration, StatefulSetConfiguration],
    ) -> None:
        for cont_config in target_config.containers:
            container = find_container(target_controller, cont_config.name)
            assert (
                container
            ), f"{type(target_controller).__name__} {target_config.name} has no container {cont_config.name}"

            for resource in Resource.values():
                current_state = None
                container_requirements = ContainerHelper.get_resource_requirements(
                    container, resource
                )
                get_requirements = cast(
                    Union[CPU, Memory], getattr(cont_config, resource)
                ).get
                for requirement in get_requirements:
                    current_state = container_requirements.get(requirement)
                    if current_state:
                        break

                assert current_state, (
                    f"{target_controller.kind} {target_config.name} target container {cont_config.name} spec does not define the resource {resource}. "
                    f"At least one of the following must be specified: {', '.join(map(lambda req: req.resources_key, get_requirements))}"
                )

    @servo.multicheck(
        'Containers in the "{item.name}" Deployment have resource requirements'
    )
    async def check_kubernetes_resource_requirements(
        self,
    ) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_dep_resource_requirements(
            dep_config: DeploymentConfiguration,
        ) -> None:
            deployment = await DeploymentHelper.read(
                dep_config.name, dep_config.namespace
            )
            await self._check_container_resource_requirements(deployment, dep_config)

        return (self.config.deployments or []), check_dep_resource_requirements

    @servo.multicheck(
        'Containers in the "{item.name}" StatefulSet have resource requirements'
    )
    async def check_kubernetes_stateful_set_resource_requirements(
        self,
    ) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_ss_resource_requirements(
            ss_config: StatefulSetConfiguration,
        ) -> None:
            stateful_set = await StatefulSetHelper.read(
                ss_config.name, ss_config.namespace
            )
            await self._check_container_resource_requirements(stateful_set, ss_config)

        return (self.config.stateful_sets or []), check_ss_resource_requirements

    @servo.multicheck('Deployment "{item.name}" is ready')
    async def check_kubernetes_deployments_are_ready(
        self,
    ) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_deployment(dep_config: DeploymentConfiguration) -> None:
            deployment = await DeploymentHelper.read(
                dep_config.name, dep_config.namespace
            )
            if not DeploymentHelper.is_ready(deployment):
                raise RuntimeError(
                    f'Deployment "{deployment.metadata.name}" is not ready'
                )

        return (self.config.deployments or []), check_deployment

    @servo.multicheck('StatefulSet "{item.name}" is ready')
    async def check_kubernetes_stateful_sets_are_ready(
        self,
    ) -> Tuple[Iterable, servo.CheckHandler]:
        async def check_stateful_set(ss_config: StatefulSetConfiguration) -> None:
            stateful_set = await StatefulSetHelper.read(
                ss_config.name, ss_config.namespace
            )
            if not StatefulSetHelper.is_ready(stateful_set):
                raise RuntimeError(
                    f'Rollout "{stateful_set.metadata.name}" is not ready'
                )

        return (self.config.stateful_sets or []), check_stateful_set


@servo.metadata(
    description="Kubernetes adjust connector",
    version="1.5.0",
    homepage="https://github.com/opsani/kubernetes-connector",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class KubernetesConnector(servo.BaseConnector):
    config: KubernetesConfiguration

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        # Ensure we are ready to talk to Kubernetes API
        await self.config.load_kubeconfig()

        self.telemetry[f"{self.name}.namespace"] = self.config.namespace

        with self.logger.catch(
            level="DEBUG",
            message=f"Unable to set version telemetry for connector {self.name}",
        ):
            async with kubernetes_asyncio.client.api_client.ApiClient() as api:
                v1 = kubernetes_asyncio.client.VersionApi(api)
                version_obj = await v1.get_code()
                self.telemetry[
                    f"{self.name}.version"
                ] = f"{version_obj.major}.{version_obj.minor}"
                self.telemetry[f"{self.name}.platform"] = version_obj.platform

    @servo.on_event()
    async def detach(self, servo_: servo.Servo) -> None:
        self.telemetry.remove(f"{self.name}.namespace")
        self.telemetry.remove(f"{self.name}.version")
        self.telemetry.remove(f"{self.name}.platform")

    @servo.on_event()
    async def describe(
        self, control: servo.Control = servo.Control()
    ) -> servo.Description:
        state = await self._create_optimizations()
        return state.to_description()

    @servo.on_event()
    async def components(self) -> List[servo.Component]:
        state = await self._create_optimizations()
        return state.to_components()

    @servo.before_event(servo.Events.measure)
    async def before_measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> None:
        # Build state before a measurement to ensure all necessary setup is done
        # (e.g., Tuning Pod is up and running)
        await self._create_optimizations()

    @servo.on_event()
    async def adjust(
        self,
        adjustments: List[servo.Adjustment],
        control: servo.Control = servo.Control(),
    ) -> servo.Description:
        state = await self._create_optimizations()

        # Apply the adjustments and emit progress status
        progress_logger = lambda p: self.logger.info(
            p.annotate(
                f"waiting up to {p.timeout} for adjustments to be applied...",
                prefix=False,
            ),
            progress=p.progress,
        )
        progress = servo.EventProgress(timeout=self.config.timeout)
        future = asyncio.create_task(state.apply(adjustments))
        future.add_done_callback(lambda _: progress.trigger())

        # Catch-all for spaghettified non-EventError usage
        try:
            await asyncio.gather(
                future,
                progress.watch(progress_logger),
            )

            # Handle settlement
            settlement = control.settlement or self.config.settlement
            if settlement:
                self.logger.info(
                    f"Settlement duration of {settlement} requested, waiting for pods to settle..."
                )
                progress = servo.DurationProgress(settlement)
                progress_logger = lambda p: self.logger.info(
                    p.annotate(f"waiting {settlement} for pods to settle...", False),
                    progress=p.progress,
                )

                async def readiness_monitor() -> None:
                    while not progress.finished:
                        if not await state.is_ready():
                            # Raise a specific exception if the optimization defines one
                            try:
                                await state.raise_for_status()
                            except servo.AdjustmentRejectedError as e:
                                # Update rejections with start-failed to indicate the initial rollout was successful
                                if e.reason == "start-failed":
                                    e.reason = "unstable"
                                raise

                        await asyncio.sleep(servo.Duration("50ms").total_seconds())

                await asyncio.gather(
                    progress.watch(progress_logger), readiness_monitor()
                )
                if not await state.is_ready():
                    self.logger.warning(
                        "Rejection triggered without running error handler"
                    )
                    raise servo.AdjustmentRejectedError(
                        "Optimization target became unready after adjustment settlement period (WARNING: error handler was not run)",
                        reason="unstable",
                    )
                self.logger.info(
                    f"Settlement duration of {settlement} has elapsed, resuming optimization."
                )

            description = state.to_description()
        except servo.EventError:  # this is recognized by the runner
            raise
        except Exception as e:
            # Convert generic errors AdjustmentFailed errors to be sent to the backend instead of shutting down servo
            # NOTE the generic error raising is left as is due to being appropriate for other points in the driver lifecycle (eg. startup and checks)
            raise servo.AdjustmentFailedError(str(e)) from e

        return description

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        await self.config.load_kubeconfig()

        return await KubernetesChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    async def _create_optimizations(self) -> KubernetesOptimizations:
        # Build a KubernetesOptimizations object with progress reporting
        # This ensures that the Servo isn't reported as offline
        progress_logger = lambda p: self.logger.info(
            p.annotate(
                f"waiting up to {p.timeout} for Kubernetes optimization setup to complete",
                prefix=False,
            ),
            progress=p.progress,
        )
        progress = servo.EventProgress(timeout=self.config.timeout)
        future = asyncio.create_task(KubernetesOptimizations.create(self.config))
        future.add_done_callback(lambda _: progress.trigger())

        await asyncio.gather(
            future,
            progress.watch(progress_logger),
        )

        return future.result()


def dns_subdomainify(name: str) -> str:
    """
    Valid DNS Subdomain Names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-' or '.'
        * start with an alphanumeric character
        * end with an alphanumeric character

    See https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
    """

    # lowercase alphanumerics
    name = name.lower()

    # replace slashes with dots
    name = re.sub(r"\/", ".", name)

    # replace whitespace with hyphens
    name = re.sub(r"\s", "-", name)

    # strip any remaining disallowed characters
    name = re.sub(r"/[^a-z0-9\.\-]+/g", "", name)

    # truncate to our maximum length
    name = name[:253]

    # ensure starts with an alphanumeric by prefixing with `0-`
    boundaryRegex = re.compile("^[a-z0-9]")
    if not boundaryRegex.match(name):
        name = ("0-" + name)[:253]

    # ensure ends with an alphanumeric by suffixing with `-1`
    if not boundaryRegex.match(name[-1]):
        name = name[:251] + "-1"

    return name


def dns_labelize(name: str) -> str:
    """
    Transform a string into a valid Kubernetes label value.

    Valid Kubernetes label values:
        * must be 63 characters or less (cannot be empty)
        * must begin and end with an alphanumeric character ([a-z0-9A-Z])
        * may contain dashes (-), underscores (_), dots (.), and alphanumerics between
    """

    # replace slashes with underscores
    name = re.sub(r"\/", "_", name)

    # replace whitespace with hyphens
    name = re.sub(r"\s", "-", name)

    # strip any remaining disallowed characters
    name = re.sub(r"[^a-z0-9A-Z\.\-_]+", "", name)

    # truncate to our maximum length
    name = name[:63]

    # ensure starts with an alphanumeric by prefixing with `0-`
    boundaryRegex = re.compile("[a-z0-9A-Z]")
    if not boundaryRegex.match(name[0]):
        name = ("0-" + name)[:63]

    # ensure ends with an alphanumeric by suffixing with `-1`
    if not boundaryRegex.match(name[-1]):
        name = name[:61] + "-1"

    return name


def set_container_resource_defaults_from_config(
    container: V1Container, config: ContainerConfiguration
) -> None:
    for resource in Resource.values():
        # NOTE: cpu/memory stanza in container config
        resource_config = getattr(config, resource)
        requirements = ContainerHelper.get_resource_requirements(container, resource)
        servo.logger.debug(
            f"Loaded resource requirements for '{resource}': {requirements}"
        )
        for requirement in ResourceRequirement:
            # Use the request/limit from the container.[cpu|memory].[request|limit] as default/override
            if resource_value := getattr(resource_config, requirement.name):
                if (existing_resource_value := requirements.get(requirement)) is None:
                    servo.logger.debug(
                        f"Setting default value for {resource}.{requirement} to: {resource_value}"
                    )
                else:
                    servo.logger.debug(
                        f"Overriding existing value for {resource}.{requirement} ({existing_resource_value}) to: {resource_value}"
                    )

                requirements[requirement] = resource_value

        servo.logger.debug(
            f"Setting resource requirements for '{resource}' to: {requirements}"
        )
        requirements = ContainerHelper.set_resource_requirements(
            container, resource, requirements
        )
