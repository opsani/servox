# TODO metrics server timestamps don't line up with time of query

import asyncio
from collections import defaultdict
from datetime import datetime
from dateutil.parser import isoparse
from enum import Enum
import functools
import os
import pathlib
import pydantic
from typing import Any, Dict, List, Optional, FrozenSet, Union

import servo
from servo.checks import CheckError
from servo.connectors.kubernetes import (
    Container,
    Deployment,
    DNSSubdomainName,
    Millicore,
    PermissionSet,
    Pod,
    ResourceRequirement,
    Rollout,
    selector_string,
    ShortByteSize
)
from servo.types import DataPoint, Metric, TimeSeries

import kubernetes_asyncio.client
import kubernetes_asyncio.client.api_client
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
    namespace: DNSSubdomainName = pydantic.Field(description="Namespace of the target resource")
    name: str = pydantic.Field(description="Name of the target resource")
    kind: pydantic.constr(regex=r"^([Dd]eployment|[Rr]ollout)$") = pydantic.Field(default="Deployment", description="Kind of the target resource")
    container: Optional[str] = pydantic.Field(default=None, description="Name of the target resource container")
    # Optional config
    metrics_to_collect: List[SupportedKubeMetrics] = pydantic.Field(
        default=[m.value for m in SupportedKubeMetrics], # TODO test servo schema
        description="Use this configuration to select which metrics are reported from this connector. Defaults to all supported metrics"
    )
    metric_collection_frequency: servo.Duration = pydantic.Field(
        default="1m",
        description="How often to get metrics from the metrics-server. Default is once per minute"
    )
    kubeconfig: Optional[pydantic.FilePath] = pydantic.Field(
        description="Path to the kubeconfig file. If `None`, use the default from the environment.",
    )
    context: Optional[str] = pydantic.Field(description="Name of the kubeconfig context to use.")

    @pydantic.validator("metrics_to_collect")
    def config_metrics_must_be_supported(cls, value: List[str]) -> List[str]:
        supported_metrics_set = {m.value for m in SupportedKubeMetrics}
        unsupported_metrics = [m for m in value if m not in supported_metrics_set]
        assert not unsupported_metrics, f"Found unsupported metrics in metrics_to_collect configuration: {', '.join(unsupported_metrics)}"
        return value

class KubeMetricsChecks(servo.BaseChecks):
    config: KubeMetricsConfiguration

    @servo.require('{self.config.kind} "{self.config.name}" is readable')
    async def check_target_resource(self) -> None:
        await _get_target_resource(self.config)

    @servo.require('Metrics API Permissions')
    async def check_metrics_api_permissions(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            for permission in KUBERNETES_PERMISSIONS:
                for resource in permission.resources:
                    for verb in permission.verbs:
                        attributes = kubernetes_asyncio.client.models.V1ResourceAttributes(
                            namespace=self.config.namespace,
                            group=permission.group,
                            resource=resource,
                            verb=verb,
                        )

                        spec = kubernetes_asyncio.client.models.V1SelfSubjectAccessReviewSpec(
                            resource_attributes=attributes
                        )
                        review = kubernetes_asyncio.client.models.V1SelfSubjectAccessReview(spec=spec)
                        access_review = await v1.create_self_subject_access_review(
                            body=review
                        )
                        assert (
                            access_review.status.allowed
                        ), f'Not allowed to "{verb}" resource "{resource}"'

    @servo.require('Metrics API connectivity')
    async def check_metrics_api(self) -> None:
        target_resource = await _get_target_resource(self.config)
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api_client=api)
            await cust_obj_api.list_namespaced_custom_object(
                label_selector=selector_string(target_resource.match_labels),
                **METRICS_CUSTOM_OJBECT_CONST_ARGS
            )

    @servo.require('Container configured or target is single container application')
    async def check_target_containers(self) -> None:
        target_resource = await _get_target_resource(self.config)
        if self.config.container:
            assert next((c for c in target_resource.containers if c.name == self.config.container), None) is not None, \
                f"Configured container {self.config.container} was not found in target app containers ({', '.join((c.name for c in target_resource.containers))})"
        elif len(target_resource.containers) > 1:
            raise CheckError("Container name must be configured for target application with multiple containers")

METRICS_CUSTOM_OJBECT_CONST_ARGS = dict(
    group="metrics.k8s.io",
    version="v1beta1",
    namespace="bank-of-anthos-opsani",
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
    async def attach(self) -> None:
        config_file = pathlib.Path(self.config.kubeconfig or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION).expanduser()
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
        await KubeMetricsChecks.run(self.config, matching=matching, halt_on=halt_on)

    @servo.on_event()
    def metrics(self) -> List[Metric]:
        return [_name_to_metric(m.value) for m in self.config.metrics_to_collect]

    @servo.on_event()
    async def describe(self, control: servo.types.Control = servo.types.Control()) -> servo.Description:
        return servo.Description(metrics=self.metrics())

    @servo.on_event()
    async def measure(self,
        metrics: List[str] = None,
        control: servo.types.Control = servo.types.Control()
    ) -> servo.Measurement:
        target_metrics = [m for m in self.config.metrics_to_collect if m.value in metrics]
        target_resource = await _get_target_resource(self.config)

        progress = servo.EventProgress(timeout=servo.Duration(control.warmup + control.duration))
        progress_reporter_task = asyncio.create_task(progress.watch(notify=lambda progress: servo.logger.info(
            progress.annotate(f"measuring kubernetes metrics for {control.duration}", False),
            progress=progress.progress,
        )))

        await asyncio.sleep(control.warmup.total_seconds())

        datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(lambda: defaultdict(list))
        while not progress.completed:
            iteration_start_time = datetime.now()

            # Retrieve latest main state
            await target_resource.refresh()
            target_resource_container = _get_target_resource_container(self.config, target_resource)

            async with kubernetes_asyncio.client.api_client.ApiClient() as api:
                cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api_client=api)
                label_selector_str = selector_string(target_resource.match_labels)

                if any((m in MAIN_METRICS_REQUIRE_CUST_OBJ for m in target_metrics)):
                    main_metrics = await cust_obj_api.list_namespaced_custom_object(
                        label_selector=f"{label_selector_str},opsani_role!=tuning",
                        **METRICS_CUSTOM_OJBECT_CONST_ARGS
                    )
                    # NOTE items can be empty list
                    for pod_entry in main_metrics["items"]:
                        pod_name = pod_entry["metadata"]["name"]
                        timestamp = isoparse(pod_entry["timestamp"])
                        _append_data_point_for_pod = functools.partial(
                            _append_data_point, datapoints_dicts=datapoints_dicts, pod_name=pod_name, time=timestamp
                        )

                        if SupportedKubeMetrics.MAIN_POD_RESTART_COUNT in target_metrics:
                            pod: Pod = next((p for p in target_resource.get_pods() if p.name == pod_name), None)
                            if pod is None:
                                # TODO may need to skip these with continue depending on whether this comes up in testing
                                raise RuntimeError(f"Unable to find pod {pod_name} in pods for {target_resource.obj.kind} {target_resource.name}")
                            restart_count = await pod.get_restart_count()
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_POD_RESTART_COUNT.value, value=restart_count)

                        target_container = self._get_target_container_metrics(pod_metrics_list_item=pod_entry)
                        if SupportedKubeMetrics.MAIN_CPU_USAGE in target_metrics:
                            cpu_usage = Millicore.parse(target_container["usage"]["cpu"])
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_CPU_USAGE.value, value=cpu_usage)

                        if SupportedKubeMetrics.MAIN_MEM_USAGE in target_metrics:
                            mem_usage=ShortByteSize.validate(target_container["usage"]["memory"])
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_MEM_USAGE.value, value=mem_usage)

                        cpu_resources = target_resource_container.get_resource_requirements("cpu")
                        if SupportedKubeMetrics.MAIN_CPU_REQUEST in target_metrics:
                            if cpu_request := cpu_resources[ResourceRequirement.request] is not None:
                                cpu_request = Millicore.parse(cpu_request)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_CPU_REQUEST.value, value=cpu_request)

                        if SupportedKubeMetrics.MAIN_CPU_LIMIT in target_metrics:
                            if cpu_limit := cpu_resources[ResourceRequirement.limit] is not None:
                                cpu_limit = Millicore.parse(cpu_limit)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_CPU_LIMIT.value, value=cpu_limit)

                        if SupportedKubeMetrics.MAIN_CPU_SATURATION in target_metrics:
                            if cpu_request := cpu_resources[ResourceRequirement.request] is not None:
                                cpu_request = Millicore.parse(cpu_request)
                                cpu_usage = Millicore.parse(target_container["usage"]["cpu"])
                                cpu_saturation = 100 * cpu_usage / cpu_request
                            else:
                                cpu_saturation = None # TODO return "NaN" string instead?
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_CPU_SATURATION.value, value=cpu_saturation)

                        mem_resources = target_resource_container.get_resource_requirements("memory")
                        if SupportedKubeMetrics.MAIN_MEM_REQUEST in target_metrics:
                            if mem_request := mem_resources[ResourceRequirement.request] is not None:
                                mem_request = ShortByteSize.validate(mem_request)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_MEM_REQUEST.value, value=mem_request)

                        if SupportedKubeMetrics.MAIN_MEM_LIMIT in target_metrics:
                            if mem_limit := mem_resources[ResourceRequirement.limit] is not None:
                                mem_limit = ShortByteSize.validate(mem_limit)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_MEM_LIMIT.value, value=mem_limit)

                        if SupportedKubeMetrics.MAIN_MEM_SATURATION in target_metrics:
                            if mem_request := mem_resources[ResourceRequirement.request] is not None:
                                mem_request = Millicore.parse(mem_request)
                                mem_usage = Millicore.parse(target_container["usage"]["memory"])
                                mem_saturation = 100 * mem_usage / mem_request
                            else:
                                mem_saturation = None
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.MAIN_MEM_SATURATION.value, value=mem_saturation)

                # Retrieve latest tuning state
                target_resource_tuning_pod: Pod = next((p for p in target_resource.get_pods() if p.name == f"{target_resource.name}-tuning"), None)
                if target_resource_tuning_pod:
                    target_resource_tuning_pod_container = _get_target_resource_container(self.config, target_resource_tuning_pod)
                    cpu_resources = target_resource_tuning_pod_container.get_resource_requirements("cpu")
                    mem_resources = target_resource_container.get_resource_requirements("memory")
                else:
                    target_resource_tuning_pod_container = None
                    cpu_resources = { ResourceRequirement.request: None, ResourceRequirement.limit: None }
                    mem_resources = { ResourceRequirement.request: None, ResourceRequirement.limit: None }

                if any((m in TUNING_METRICS_REQUIRE_CUST_OBJ for m in target_metrics)):
                    tuning_metrics = await cust_obj_api.list_namespaced_custom_object(
                        label_selector=f"{label_selector_str},opsani_role=tuning"
                        **METRICS_CUSTOM_OJBECT_CONST_ARGS
                    )
                    # TODO: raise error if more than 1 tuning pod?
                    for pod_entry in tuning_metrics["items"]:
                        pod_name = pod_entry["metadata"]["name"]
                        if pod_name != f"{target_resource.name}-tuning":
                            raise RuntimeError(f"Got unexpected tuning pod name {pod_name}")
                        timestamp = isoparse(pod_entry["timestamp"])
                        _append_data_point_for_pod = functools.partial(
                            _append_data_point, datapoints_dicts=datapoints_dicts, pod_name=pod_name, time=timestamp
                        )

                        # TODO additional logic to get this metric without custom object response
                        if SupportedKubeMetrics.TUNING_POD_RESTART_COUNT in target_metrics:
                            if target_resource_tuning_pod is not None:
                                restart_count = await target_resource_tuning_pod.get_restart_count()
                            else:
                                restart_count = None
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_POD_RESTART_COUNT.value, value=restart_count)

                        target_container = self._get_target_container_metrics(pod_metrics_list_item=pod_entry)
                        if SupportedKubeMetrics.TUNING_CPU_USAGE in target_metrics:
                            cpu_usage = Millicore.parse(target_container["usage"]["cpu"])
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_CPU_USAGE.value, value=cpu_usage)

                        if SupportedKubeMetrics.TUNING_MEM_USAGE in target_metrics:
                            mem_usage = ShortByteSize.validate(target_container["usage"]["memory"])
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_MEM_USAGE.value, value=mem_usage)

                        if SupportedKubeMetrics.TUNING_CPU_REQUEST in target_metrics:
                            if cpu_request := cpu_resources[ResourceRequirement.request] is not None:
                                cpu_request = Millicore.parse(cpu_request)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_CPU_REQUEST.value, value=cpu_request)

                        if SupportedKubeMetrics.TUNING_CPU_LIMIT in target_metrics:
                            if cpu_limit := cpu_resources[ResourceRequirement.limit] is not None:
                                cpu_limit = Millicore.parse(cpu_limit)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_CPU_LIMIT.value, value=cpu_limit)

                        if SupportedKubeMetrics.TUNING_CPU_SATURATION in target_metrics:
                            if cpu_request := cpu_resources[ResourceRequirement.request] is not None:
                                cpu_request = Millicore.parse(cpu_request)
                                cpu_usage = Millicore.parse(target_container["usage"]["cpu"])
                                cpu_saturation = 100 * cpu_usage / cpu_request
                            else:
                                cpu_saturation = None
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_CPU_SATURATION.value, value=cpu_saturation)

                        if SupportedKubeMetrics.TUNING_MEM_REQUEST in target_metrics:
                            if mem_request := mem_resources[ResourceRequirement.request] is not None:
                                mem_request = ShortByteSize.validate(mem_request)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_MEM_REQUEST.value, value=mem_request)

                        if SupportedKubeMetrics.TUNING_MEM_LIMIT in target_metrics:
                            if mem_limit := mem_resources[ResourceRequirement.limit] is not None:
                                mem_limit = ShortByteSize.validate(mem_limit)
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_MEM_LIMIT.value, value=mem_limit)

                        if SupportedKubeMetrics.TUNING_MEM_SATURATION in target_metrics:
                            if mem_request := mem_resources[ResourceRequirement.request] is not None:
                                mem_request = Millicore.parse(mem_request)
                                mem_usage = Millicore.parse(target_container["usage"]["memory"])
                                mem_saturation = 100 * mem_usage / mem_request
                            else:
                                mem_saturation = None
                            _append_data_point_for_pod(metric_name=SupportedKubeMetrics.TUNING_MEM_SATURATION.value, value=mem_saturation)

            sleep_time = max(0, self.config.metric_collection_frequency.total_seconds() - (datetime.now() - iteration_start_time))
            await asyncio.sleep(sleep_time)

        # Convert data points dicts to TimeSeries list
        readings = []
        for metric_name, pod_datapoints in datapoints_dicts:
            for pod_name, datapoints in pod_datapoints:
                readings.append(TimeSeries(
                    metric=_name_to_metric(metric_name),
                    data_points=datapoints,
                    id=pod_name
                ))

        await asyncio.sleep(control.delay.total_seconds())

        return readings

    def _get_target_container_metrics(self, pod_metrics_list_item: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, str]]]:
        pod_name = pod_metrics_list_item["metadata"]["name"]
        if self.config.container:
            target_container = next((c for c in pod_metrics_list_item["containers"] if c["name"] == self.config.container), None)
            if target_container is None:
                raise RuntimeError(
                    f"Unable to find target container {self.config.container} in pod {pod_name} "
                    f"(found {', '.join(c['name'] for c in pod_metrics_list_item['containers'])})"
                )
            return target_container
        # TODO exclude opsani-envoy container by default?
        elif len(pod_metrics_list_item["containers"]) > 1:
            raise RuntimeError(
                f"Found multiple containers ({', '.join((c['name'] for c in pod_metrics_list_item['containers']))}) in "
                f"pod {pod_name}, unable to determine target wihtout container configuration"
            )
        else:
            return pod_metrics_list_item["containers"][0]

def _append_data_point(
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]], pod_name: str, metric_name: str, time: datetime, value: Any
):
    datapoints_dicts[metric_name][pod_name].append(
        DataPoint(
            metric=_name_to_metric(metric_name),
            time=time,
            value=value
        )
    )

async def _get_target_resource(config: KubeMetricsConfiguration) -> Union[Deployment, Rollout]:
    read_args = dict(name=config.name, namespace=config.namespace)
    if config.kind.lower() == "deployment":
        return await Deployment.read(**read_args)
    elif config.kind.lower() == "rollout":
        return await Rollout.read(**read_args)
    else:
        raise NotImplementedError(f"Resource type {config.kind} is not supported by the kube-metrics connector")

def _get_target_resource_container(
    config: KubeMetricsConfiguration, target_resource: Union[Deployment, Rollout, Pod]
) -> Container:
    if config.container:
        if isinstance(target_resource, Pod):
            target_resource_container: Container = target_resource.get_container(config.container)
        else:
            target_resource_container: Container = target_resource.find_container(config.container)

        if target_resource_container is None:
            raise RuntimeError(f"Unable to locate container {config.container} in {target_resource.obj.kind} {target_resource.name}")
    elif len(target_resource.containers) > 1:
        # TODO can support this with ID append
        raise RuntimeError(f"Unable to derive metrics for multi-container resources")
    else:
        target_resource_container: Container = target_resource.containers[0]

    return target_resource_container

def _name_to_metric(metric_name: str) -> Metric:
    if "saturation" in metric_name:
        unit = servo.Unit.percentage
    elif "count" in metric_name:
        unit = servo.Unit.count
    else:
        unit = servo.Unit.float

    return Metric(name=metric_name, unit=unit)
