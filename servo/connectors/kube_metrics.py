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

# TODO (may not require action) metrics server timestamps don't line up with time of query

import asyncio
from collections import defaultdict
from datetime import datetime
from dateutil.parser import isoparse
from enum import Enum
import functools
import os
import pathlib
import pydantic
import time
from typing import Any, Dict, List, Optional, FrozenSet, Union

import servo
from servo.checks import CheckError
from servo.connectors.kubernetes_helpers import (
    dict_to_selector,
    find_container,
    get_containers,
    ContainerHelper,
    DeploymentHelper,
    PodHelper,
    StatefulSetHelper,
)
from servo.connectors.kubernetes import (
    DNSSubdomainName,
    Core,
    PermissionSet,
    ShortByteSize,
)
import servo.types
from servo.types import DataPoint, Metric, TimeSeries, Resource, ResourceRequirement

import kubernetes_asyncio.client
from kubernetes_asyncio.client import V1Container, V1Deployment, V1Pod, V1StatefulSet
import kubernetes_asyncio.client.api_client
import kubernetes_asyncio.client.exceptions
import kubernetes_asyncio.config
import kubernetes_asyncio.config.kube_config

KUBERNETES_PERMISSIONS = [
    PermissionSet(
        group="metrics.k8s.io",
        resources=["pods"],
        verbs=["list"],
    ),
]


class SupportedKubeMetrics(str, Enum):
    TUNING_CPU_USAGE = "tuning_cpu_usage"
    TUNING_CPU_REQUEST = "tuning_cpu_request"
    TUNING_CPU_LIMIT = "tuning_cpu_limit"
    TUNING_CPU_SATURATION = "tuning_cpu_saturation"
    TUNING_MEM_USAGE = "tuning_mem_usage"
    TUNING_MEM_REQUEST = "tuning_mem_request"
    TUNING_MEM_LIMIT = "tuning_mem_limit"
    TUNING_MEM_SATURATION = "tuning_mem_saturation"
    TUNING_POD_RESTART_COUNT = "tuning_pod_restart_count"
    MAIN_CPU_USAGE = "main_cpu_usage"
    MAIN_CPU_REQUEST = "main_cpu_request"
    MAIN_CPU_LIMIT = "main_cpu_limit"
    MAIN_CPU_SATURATION = "main_cpu_saturation"
    MAIN_MEM_USAGE = "main_mem_usage"
    MAIN_MEM_REQUEST = "main_mem_request"
    MAIN_MEM_LIMIT = "main_mem_limit"
    MAIN_MEM_SATURATION = "main_mem_saturation"
    MAIN_POD_RESTART_COUNT = "main_pod_restart_count"


TUNING_METRICS_REQUIRE_CUST_OBJ: FrozenSet[SupportedKubeMetrics] = {
    SupportedKubeMetrics.TUNING_CPU_USAGE,
    SupportedKubeMetrics.TUNING_CPU_SATURATION,
    SupportedKubeMetrics.TUNING_MEM_USAGE,
    SupportedKubeMetrics.TUNING_MEM_SATURATION,
}

MAIN_METRICS_REQUIRE_CUST_OBJ: FrozenSet[SupportedKubeMetrics] = {
    SupportedKubeMetrics.MAIN_CPU_USAGE,
    SupportedKubeMetrics.MAIN_CPU_SATURATION,
    SupportedKubeMetrics.MAIN_MEM_USAGE,
    SupportedKubeMetrics.MAIN_MEM_SATURATION,
}


class KubeMetricsConfiguration(servo.BaseConfiguration):
    namespace: DNSSubdomainName = pydantic.Field(
        description="Namespace of the target resource"
    )
    name: str = pydantic.Field(description="Name of the target resource")
    kind: str = pydantic.Field(
        default="Deployment",
        description="Kind of the target resource",
        regex=r"^([Dd]eployment|[Ss]tateful[Ss]et)$",
    )
    container: Optional[str] = pydantic.Field(
        default=None, description="Name of the target resource container"
    )
    # Optional config
    metrics_to_collect: List[SupportedKubeMetrics] = pydantic.Field(
        default=[m.value for m in SupportedKubeMetrics],
        description="Use this configuration to select which metrics are reported from this connector. Defaults to all supported metrics",
    )
    metric_collection_frequency: servo.Duration = pydantic.Field(
        default="1m",
        description="How often to get metrics from the metrics-server. Default is once per minute",
    )
    kubeconfig: Optional[pydantic.FilePath] = pydantic.Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = pydantic.Field(
        description="Name of the kubeconfig context to use."
    )

    @pydantic.validator("metrics_to_collect")
    def config_metrics_must_be_supported(cls, value: List[str]) -> List[str]:
        supported_metrics_set = {m.value for m in SupportedKubeMetrics}
        unsupported_metrics = [m for m in value if m not in supported_metrics_set]
        assert (
            not unsupported_metrics
        ), f"Found unsupported metrics in metrics_to_collect configuration: {', '.join(unsupported_metrics)}"
        return value

    @classmethod
    def generate(cls, **kwargs) -> "KubeMetricsConfiguration":
        return cls(
            namespace="default",
            name="app",
            container="app_container_name",
            kind="Deployment",
            description="Update the namespace, resource name, etc. to match the resource to monitor",
            **kwargs,
        )


class KubeMetricsChecks(servo.BaseChecks):
    config: KubeMetricsConfiguration

    @servo.require('{self.config.kind} "{self.config.name}" is readable')
    async def check_target_resource(self) -> None:
        await _get_target_resource(self.config)

    @servo.require("Metrics API Permissions")
    async def check_metrics_api_permissions(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            for permission in KUBERNETES_PERMISSIONS:
                for resource in permission.resources:
                    for verb in permission.verbs:
                        attributes = kubernetes_asyncio.client.V1ResourceAttributes(
                            namespace=self.config.namespace,
                            group=permission.group,
                            resource=resource,
                            verb=verb,
                        )

                        spec = kubernetes_asyncio.client.V1SelfSubjectAccessReviewSpec(
                            resource_attributes=attributes
                        )
                        review = kubernetes_asyncio.client.V1SelfSubjectAccessReview(
                            spec=spec
                        )
                        access_review = await v1.create_self_subject_access_review(
                            body=review
                        )
                        assert (
                            access_review.status.allowed
                        ), f'Not allowed to "{verb}" resource "{resource}" in group "{permission.group}"'

    @servo.require("Metrics API connectivity")
    async def check_metrics_api(self) -> None:
        target_resource = await _get_target_resource(self.config)
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api_client=api)
            await cust_obj_api.list_namespaced_custom_object(
                label_selector=dict_to_selector(
                    target_resource.spec.selector.match_labels
                ),
                namespace=self.config.namespace,
                **METRICS_CUSTOM_OJBECT_CONST_ARGS,
            )

    @servo.require("Container configured or target is single container application")
    async def check_target_containers(self) -> None:
        target_resource = await _get_target_resource(self.config)
        if self.config.container:
            assert (
                find_container(workload=target_resource, name=self.config.container)
                is not None
            ), (
                f"Configured container {self.config.container} was not found in target app containers"
                f" ({', '.join((c.name for c in get_containers(workload=target_resource)))})"
            )
        elif len(get_containers(workload=target_resource)) > 1:
            raise CheckError(
                "Container name must be configured for target application with multiple containers"
            )


METRICS_CUSTOM_OJBECT_CONST_ARGS = dict(
    group="metrics.k8s.io",
    version="v1beta1",
    plural="pods",
)


@servo.metadata(
    description="Kubernetes metrics-server connector",
    version="0.0.1",
    homepage="https://github.com/opsani/servox",
    license=servo.License.apache2,
    maturity=servo.Maturity.experimental,
)
class KubeMetricsConnector(servo.BaseConnector):
    config: KubeMetricsConfiguration

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        config_file = pathlib.Path(
            self.config.kubeconfig
            or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION
        ).expanduser()
        if config_file.exists():
            await kubernetes_asyncio.config.load_kube_config(
                config_file=str(config_file),
                context=self.config.context,
            )
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            kubernetes_asyncio.config.load_incluster_config()
        else:
            raise RuntimeError(
                f"unable to configure Kubernetes client: no kubeconfig file nor in-cluser environment variables found"
            )

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        return await KubeMetricsChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    @servo.on_event()
    def metrics(self) -> List[Metric]:
        return [_name_to_metric(m.value) for m in self.config.metrics_to_collect]

    @servo.on_event()
    async def describe(
        self, control: servo.types.Control = servo.types.Control()
    ) -> servo.Description:
        return servo.Description(metrics=self.metrics())

    @servo.on_event()
    async def measure(
        self,
        metrics: List[str] = [m.value for m in SupportedKubeMetrics],
        control: servo.types.Control = servo.types.Control(),
    ) -> servo.Measurement:
        target_metrics = [
            m for m in self.config.metrics_to_collect if m.value in metrics
        ]

        progress_duration = servo.Duration(control.warmup + control.duration)
        progress = servo.EventProgress(timeout=progress_duration)
        progress_reporter_task = asyncio.create_task(
            progress.watch(
                notify=lambda progress: servo.logger.info(
                    progress.annotate(
                        f"measuring kubernetes metrics for {progress_duration}", False
                    ),
                    progress=progress.progress,
                )
            )
        )

        await asyncio.sleep(control.warmup.total_seconds())

        datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(
            lambda: defaultdict(list)
        )
        while not progress.finished:
            iteration_start_time = time.time()

            try:
                await self.periodic_measure(
                    target_metrics=target_metrics,
                    datapoints_dicts=datapoints_dicts,
                )
            except kubernetes_asyncio.client.exceptions.ApiException as ae:
                if ae.status == 404:
                    raise servo.MeasurementFailedError(
                        f"Resource not found, failing measurement: {ae.body}"
                    ) from ae
                else:
                    raise

            sleep_time = max(
                0,
                self.config.metric_collection_frequency.total_seconds()
                - (time.time() - iteration_start_time),
            )
            await asyncio.sleep(sleep_time)

        # Convert data points dicts to TimeSeries list
        readings = []
        for metric_name, pod_datapoints in datapoints_dicts.items():
            for pod_name, datapoints in pod_datapoints.items():
                readings.append(
                    TimeSeries(
                        metric=_name_to_metric(metric_name),
                        data_points=datapoints,
                        id=pod_name,
                    )
                )

        # TODO (fix here and other connectors)
        # await asyncio.sleep(control.delay.total_seconds())

        measurement = servo.Measurement(readings=readings)
        return measurement

    def _get_target_container_metrics(
        self, pod_metrics_list_item: Dict[str, Any]
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        pod_name = pod_metrics_list_item["metadata"]["name"]
        if self.config.container:
            target_container = next(
                (
                    c
                    for c in pod_metrics_list_item["containers"]
                    if c["name"] == self.config.container
                ),
                None,
            )
            if target_container is None:
                raise RuntimeError(
                    f"Unable to find target container {self.config.container} in pod {pod_name} "
                    f"(found {', '.join(c['name'] for c in pod_metrics_list_item['containers'])})"
                )
            return target_container
        # TODO (improvement) exclude opsani-envoy container by default?
        elif len(pod_metrics_list_item["containers"]) > 1:
            raise RuntimeError(
                f"Found multiple containers ({', '.join((c['name'] for c in pod_metrics_list_item['containers']))}) in "
                f"pod {pod_name}, unable to determine target wihtout container configuration"
            )
        else:
            return pod_metrics_list_item["containers"][0]

    async def periodic_measure(
        self,
        target_metrics: list[SupportedKubeMetrics],
        datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]],
    ) -> None:
        # Retrieve latest main state
        target_resource = await _get_target_resource(self.config)
        target_resource_container = _get_target_resource_container(
            self.config, target_resource
        )

        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api_client=api)
            label_selector_str = dict_to_selector(
                target_resource.spec.selector.match_labels
            )
            timestamp = datetime.now()

            if any((m in MAIN_METRICS_REQUIRE_CUST_OBJ for m in target_metrics)):
                main_metrics = await cust_obj_api.list_namespaced_custom_object(
                    label_selector=f"{label_selector_str},opsani_role!=tuning",
                    namespace=self.config.namespace,
                    **METRICS_CUSTOM_OJBECT_CONST_ARGS,
                )
                # NOTE items can be empty list
                for pod_entry in main_metrics["items"]:
                    pod_name = pod_entry["metadata"]["name"]
                    timestamp = isoparse(pod_entry["timestamp"])
                    _append_data_point_for_pod = functools.partial(
                        _append_data_point,
                        datapoints_dicts=datapoints_dicts,
                        pod_name=pod_name,
                        time=timestamp,
                    )

                    target_container = self._get_target_container_metrics(
                        pod_metrics_list_item=pod_entry
                    )
                    if SupportedKubeMetrics.MAIN_CPU_USAGE in target_metrics:
                        cpu_usage = Core.parse(target_container["usage"]["cpu"])
                        _append_data_point_for_pod(
                            metric_name=SupportedKubeMetrics.MAIN_CPU_USAGE.value,
                            value=cpu_usage,
                        )

                    if SupportedKubeMetrics.MAIN_MEM_USAGE in target_metrics:
                        mem_usage = ShortByteSize.validate(
                            target_container["usage"]["memory"]
                        )
                        _append_data_point_for_pod(
                            metric_name=SupportedKubeMetrics.MAIN_MEM_USAGE.value,
                            value=mem_usage,
                        )

                    cpu_resources = ContainerHelper.get_resource_requirements(
                        target_resource_container, Resource.cpu.value
                    )
                    # Set requests = limits if not specified
                    if (
                        cpu_request := cpu_resources[ResourceRequirement.request]
                    ) is None:
                        cpu_request = cpu_resources[ResourceRequirement.limit]

                    if SupportedKubeMetrics.MAIN_CPU_REQUEST in target_metrics:
                        if cpu_request is not None:
                            cpu_request = Core.parse(cpu_request)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_CPU_REQUEST.value,
                                value=cpu_request,
                            )

                    if SupportedKubeMetrics.MAIN_CPU_LIMIT in target_metrics:
                        if (
                            cpu_limit := cpu_resources[ResourceRequirement.limit]
                        ) is not None:
                            cpu_limit = Core.parse(cpu_limit)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_CPU_LIMIT.value,
                                value=cpu_limit,
                            )

                    if SupportedKubeMetrics.MAIN_CPU_SATURATION in target_metrics:
                        if cpu_request is not None:
                            cpu_request = Core.parse(cpu_request)
                            cpu_usage = Core.parse(target_container["usage"]["cpu"])
                            cpu_saturation = 100 * cpu_usage / cpu_request
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_CPU_SATURATION.value,
                                value=cpu_saturation,
                            )

                    mem_resources = ContainerHelper.get_resource_requirements(
                        target_resource_container, Resource.memory.value
                    )
                    # Set requests = limits if not specified
                    if (
                        mem_request := mem_resources[ResourceRequirement.request]
                    ) is None:
                        mem_request = mem_resources[ResourceRequirement.limit]

                    if SupportedKubeMetrics.MAIN_MEM_REQUEST in target_metrics:
                        if mem_request is not None:
                            mem_request = ShortByteSize.validate(mem_request)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_MEM_REQUEST.value,
                                value=mem_request,
                            )

                    if SupportedKubeMetrics.MAIN_MEM_LIMIT in target_metrics:
                        if (
                            mem_limit := mem_resources[ResourceRequirement.limit]
                        ) is not None:
                            mem_limit = ShortByteSize.validate(mem_limit)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_MEM_LIMIT.value,
                                value=mem_limit,
                            )

                    if SupportedKubeMetrics.MAIN_MEM_SATURATION in target_metrics:
                        if mem_request is not None:
                            mem_request = ShortByteSize.validate(mem_request)
                            mem_usage = ShortByteSize.validate(
                                target_container["usage"]["memory"]
                            )
                            mem_saturation = 100 * mem_usage / mem_request
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.MAIN_MEM_SATURATION.value,
                                value=mem_saturation,
                            )

            if SupportedKubeMetrics.MAIN_POD_RESTART_COUNT in target_metrics:
                _append_data_point_for_time = functools.partial(
                    _append_data_point,
                    datapoints_dicts=datapoints_dicts,
                    time=timestamp,
                )
                target_pods = [
                    pod
                    for pod in await PodHelper.list_pods_with_labels(
                        target_resource.metadata.namespace,
                        target_resource.spec.selector.match_labels,
                    )
                    if "tuning" not in pod.metadata.name
                ]
                for pod in target_pods:
                    _append_data_point_for_time(
                        pod_name=pod.metadata.name,
                        metric_name=SupportedKubeMetrics.MAIN_POD_RESTART_COUNT.value,
                        value=PodHelper.get_restart_count(pod),
                    )

            # Retrieve latest tuning state
            target_resource_tuning_pod_name = f"{target_resource.metadata.name}-tuning"
            try:
                target_resource_tuning_pod = await PodHelper.read(
                    target_resource_tuning_pod_name, target_resource.metadata.namespace
                )
            except kubernetes_asyncio.client.exceptions.ApiException as e:
                if e.status != 404 or e.reason != "Not Found":
                    raise
                target_resource_tuning_pod = None

            if target_resource_tuning_pod:
                target_resource_tuning_pod_container = _get_target_resource_container(
                    self.config, target_resource_tuning_pod
                )
                cpu_resources = ContainerHelper.get_resource_requirements(
                    target_resource_tuning_pod_container, Resource.cpu.value
                )
                # Set requests = limits if not specified
                if (cpu_request := cpu_resources[ResourceRequirement.request]) is None:
                    cpu_request = cpu_resources[ResourceRequirement.limit]

                mem_resources = ContainerHelper.get_resource_requirements(
                    target_resource_tuning_pod_container, Resource.memory.value
                )
                if (mem_request := mem_resources[ResourceRequirement.request]) is None:
                    mem_request = mem_resources[ResourceRequirement.limit]
            else:
                target_resource_tuning_pod_container = None
                cpu_resources = {
                    ResourceRequirement.request: None,
                    ResourceRequirement.limit: None,
                }
                mem_resources = {
                    ResourceRequirement.request: None,
                    ResourceRequirement.limit: None,
                }

            restart_count = None
            if SupportedKubeMetrics.TUNING_POD_RESTART_COUNT in target_metrics:
                if target_resource_tuning_pod is not None:
                    restart_count = PodHelper.get_restart_count(
                        target_resource_tuning_pod
                    )
                else:
                    restart_count = 0

            if any((m in TUNING_METRICS_REQUIRE_CUST_OBJ for m in target_metrics)):
                tuning_metrics = await cust_obj_api.list_namespaced_custom_object(
                    label_selector=f"{label_selector_str},opsani_role=tuning",
                    namespace=self.config.namespace,
                    **METRICS_CUSTOM_OJBECT_CONST_ARGS,
                )
                # TODO: (potential improvement) raise error if more than 1 tuning pod?
                for pod_entry in tuning_metrics["items"]:
                    pod_name = pod_entry["metadata"]["name"]
                    if pod_name != target_resource_tuning_pod_name:
                        raise RuntimeError(f"Got unexpected tuning pod name {pod_name}")
                    timestamp = isoparse(pod_entry["timestamp"])
                    _append_data_point_for_pod = functools.partial(
                        _append_data_point,
                        datapoints_dicts=datapoints_dicts,
                        pod_name=pod_name,
                        time=timestamp,
                    )

                    if restart_count is not None:
                        _append_data_point_for_pod(
                            metric_name=SupportedKubeMetrics.TUNING_POD_RESTART_COUNT.value,
                            value=restart_count,
                        )

                    target_container = self._get_target_container_metrics(
                        pod_metrics_list_item=pod_entry
                    )
                    if SupportedKubeMetrics.TUNING_CPU_USAGE in target_metrics:
                        cpu_usage = Core.parse(target_container["usage"]["cpu"])
                        _append_data_point_for_pod(
                            metric_name=SupportedKubeMetrics.TUNING_CPU_USAGE.value,
                            value=cpu_usage,
                        )

                    if SupportedKubeMetrics.TUNING_MEM_USAGE in target_metrics:
                        mem_usage = ShortByteSize.validate(
                            target_container["usage"]["memory"]
                        )
                        _append_data_point_for_pod(
                            metric_name=SupportedKubeMetrics.TUNING_MEM_USAGE.value,
                            value=mem_usage,
                        )

                    if SupportedKubeMetrics.TUNING_CPU_REQUEST in target_metrics:
                        if cpu_request is not None:
                            cpu_request = Core.parse(cpu_request)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_CPU_REQUEST.value,
                                value=cpu_request,
                            )

                    if SupportedKubeMetrics.TUNING_CPU_LIMIT in target_metrics:
                        if (
                            cpu_limit := cpu_resources[ResourceRequirement.limit]
                        ) is not None:
                            cpu_limit = Core.parse(cpu_limit)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_CPU_LIMIT.value,
                                value=cpu_limit,
                            )

                    if SupportedKubeMetrics.TUNING_CPU_SATURATION in target_metrics:
                        if cpu_request is not None:
                            cpu_request = Core.parse(cpu_request)
                            cpu_usage = Core.parse(target_container["usage"]["cpu"])
                            cpu_saturation = 100 * cpu_usage / cpu_request
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_CPU_SATURATION.value,
                                value=cpu_saturation,
                            )

                    if SupportedKubeMetrics.TUNING_MEM_REQUEST in target_metrics:
                        if mem_request is not None:
                            mem_request = ShortByteSize.validate(mem_request)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_MEM_REQUEST.value,
                                value=mem_request,
                            )

                    if SupportedKubeMetrics.TUNING_MEM_LIMIT in target_metrics:
                        if (
                            mem_limit := mem_resources[ResourceRequirement.limit]
                        ) is not None:
                            mem_limit = ShortByteSize.validate(mem_limit)
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_MEM_LIMIT.value,
                                value=mem_limit,
                            )

                    if SupportedKubeMetrics.TUNING_MEM_SATURATION in target_metrics:
                        if mem_request is not None:
                            mem_request = ShortByteSize.validate(mem_request)
                            mem_usage = ShortByteSize.validate(
                                target_container["usage"]["memory"]
                            )
                            mem_saturation = 100 * mem_usage / mem_request
                            _append_data_point_for_pod(
                                metric_name=SupportedKubeMetrics.TUNING_MEM_SATURATION.value,
                                value=mem_saturation,
                            )

            elif restart_count is not None:
                _append_data_point(
                    datapoints_dicts=datapoints_dicts,
                    pod_name=target_resource_tuning_pod_name,
                    time=datetime.now(),
                    metric_name=SupportedKubeMetrics.TUNING_POD_RESTART_COUNT.value,
                    value=restart_count,
                )


def _append_data_point(
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]],
    pod_name: str,
    metric_name: str,
    time: datetime,
    value: Union[Core, ShortByteSize, Any],
):
    if isinstance(value, (Core, ShortByteSize)):
        value = value.__opsani_repr__()
    datapoints_dicts[metric_name][pod_name].append(
        DataPoint(metric=_name_to_metric(metric_name), time=time, value=value)
    )


async def _get_target_resource(
    config: KubeMetricsConfiguration,
) -> Union[V1Deployment, V1StatefulSet]:
    read_args = dict(name=config.name, namespace=config.namespace)
    if config.kind.lower() == "deployment":
        return await DeploymentHelper.read(**read_args)
    elif config.kind.lower() == "statefulset":
        return await StatefulSetHelper.read(**read_args)
    else:
        raise NotImplementedError(
            f"Resource type {config.kind} is not supported by the kube-metrics connector"
        )


def _get_target_resource_container(
    config: KubeMetricsConfiguration,
    target_resource: Union[V1Deployment, V1StatefulSet, V1Pod],
) -> V1Container:
    if config.container:
        target_resource_container = find_container(
            workload=target_resource, name=config.container
        )
        if target_resource_container is None:
            raise RuntimeError(
                f"Unable to locate container {config.container} in {target_resource.kind} {target_resource.metadata.name}"
            )
    else:
        containers = get_containers(workload=target_resource)
        # TODO (improvement) can support this with ID append
        if len(containers) > 1:
            raise RuntimeError(
                f"Unable to derive metrics for multi-container resources"
            )

        target_resource_container = containers[0]

    return target_resource_container


def _name_to_metric(metric_name: str) -> Metric:
    if "saturation" in metric_name:
        unit = servo.Unit.percentage
    elif "count" in metric_name:
        unit = servo.Unit.count
    else:
        unit = servo.Unit.float

    return Metric(name=metric_name, unit=unit)
