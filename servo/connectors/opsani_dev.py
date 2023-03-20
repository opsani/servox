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

import abc
import json
import operator
import os
from typing import cast, Dict, List, Optional, Union, Type

import kubernetes_asyncio
import kubernetes_asyncio.client
import pydantic

import servo
import servo.types
from servo.types.kubernetes import Resource
import servo.connectors.kubernetes
from servo.connectors.kubernetes_helpers import (
    find_container,
    ContainerHelper,
    DeploymentHelper,
    NamespaceHelper,
    PodHelper,
    ServiceHelper,
)
import servo.connectors.kube_metrics
import servo.connectors.prometheus

KUBERNETES_PERMISSIONS = [
    servo.connectors.kubernetes.PermissionSet(
        group="apps",
        resources=["deployments", "deployments/status", "replicasets"],
        verbs=["get", "list", "watch", "update", "patch"],
    ),
    servo.connectors.kubernetes.PermissionSet(
        group="",
        resources=["namespaces"],
        verbs=["get"],
    ),
    servo.connectors.kubernetes.PermissionSet(
        group="",
        resources=[
            "pods",
            "pods/logs",
            "pods/status",
            "pods/exec",
            "pods/portforward",
            "services",
        ],
        verbs=["create", "delete", "get", "list", "watch", "update", "patch"],
    ),
]
PROMETHEUS_SIDECAR_BASE_URL = "http://localhost:9090"
PROMETHEUS_ANNOTATION_DEFAULTS = {
    "prometheus.opsani.com/scrape": "true",
    "prometheus.opsani.com/scheme": "http",
    "prometheus.opsani.com/path": "/stats/prometheus",
    "prometheus.opsani.com/port": "9901",
}
ENVOY_SIDECAR_IMAGE_TAG = "opsani/envoy-proxy:v1.24.0"
ENVOY_SIDECAR_LABELS = {"sidecar.opsani.com/type": "envoy"}
ENVOY_SIDECAR_DEFAULT_PORT = 9980


class CPU(servo.connectors.kubernetes.CPU):
    step: servo.connectors.kubernetes.Core = "125m"


class Memory(servo.connectors.kubernetes.Memory):
    step: servo.connectors.kubernetes.ShortByteSize = "128 MiB"


class OpsaniDevConfiguration(servo.BaseConfiguration):
    namespace: str
    workload_name: str = pydantic.Field(
        alias="deployment",
        env=["deployment", "workload"],
        title="Workload Name",
        description=(
            "Name of the targeted workload (NOTE: the workload_name key should be used for this config going"
            " forward. The deployment key is supported for backwards compatibility)"
        ),
    )  # alias to maintain backward compatibility
    workload_kind: str = pydantic.Field(
        default="Deployment",
        regex=r"^([Dd]eployment)$",
    )
    container: str
    service: str
    port: Optional[Union[pydantic.StrictInt, str]] = None
    cpu: CPU
    memory: Memory
    env: Optional[servo.EnvironmentSettingList]
    static_environment_variables: Optional[Dict[str, str]]
    prometheus_base_url: str = PROMETHEUS_SIDECAR_BASE_URL
    envoy_sidecar_image: str = ENVOY_SIDECAR_IMAGE_TAG
    timeout: servo.Duration = "5m"
    settlement: Optional[servo.Duration] = pydantic.Field(
        description="Duration to observe the application after an adjust to ensure the deployment is stable. May be overridden by optimizer supplied `control.adjust.settlement` value."
    )
    container_logs_in_error_status: bool = pydantic.Field(
        False, description="Enable to include container logs in error message"
    )
    create_tuning_pod: bool = pydantic.Field(
        True,
        description="Disable to prevent a canary strategy",
    )

    class Config(servo.AbstractBaseConfiguration.Config):
        allow_population_by_field_name = True

    @classmethod
    def generate(cls, **kwargs) -> "OpsaniDevConfiguration":
        return cls(
            namespace="default",
            workload_name="app-deployment",
            container="main",
            service="app",
            cpu=CPU(min="250m", max="4000m"),
            memory=Memory(min="256 MiB", max="4.0 GiB"),
        )

    def generate_kubernetes_config(
        self, **kwargs
    ) -> servo.connectors.kubernetes.KubernetesConfiguration:
        """Generate a configuration for running an Opsani Dev optimization under servo.connectors.kubernetes.

        Returns:
            A Kubernetes connector configuration object.
        """
        strategy: Union[
            servo.connectors.kubernetes.CanaryOptimizationStrategyConfiguration,
            servo.connectors.kubernetes.DefaultOptimizationStrategyConfiguration,
        ] = servo.connectors.kubernetes.DefaultOptimizationStrategyConfiguration()

        if self.create_tuning_pod:
            strategy = (
                servo.connectors.kubernetes.CanaryOptimizationStrategyConfiguration(
                    type=servo.connectors.kubernetes.OptimizationStrategy.canary,
                    alias="tuning",
                )
            )

            replicas = servo.Replicas(min=0, max=1, pinned=True)

        else:
            # NOTE: currently assuming we NEVER want to adjust the main deployment with the opsani_dev connector
            # TODO: Do we ever need to support opsani dev bootstrapping of non-canary adjusted optimization of deployments?
            self.cpu.pinned = True
            self.memory.pinned = True

            replicas = servo.Replicas(min=0, max=99999, pinned=True)

        workload_kwargs = dict(
            name=self.workload_name,
            strategy=strategy,
            replicas=replicas,
            containers=[
                servo.connectors.kubernetes.ContainerConfiguration(
                    name=self.container,
                    alias="main",
                    cpu=self.cpu,
                    memory=self.memory,
                    static_environment_variables=self.static_environment_variables,
                    env=self.env,
                )
            ],
        )
        main_config_kwargs = dict(
            namespace=self.namespace,
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            timeout=self.timeout,
            settlement=self.settlement,
            container_logs_in_error_status=self.container_logs_in_error_status,
            create_tuning_pod=self.create_tuning_pod,
        )
        if self.workload_kind.lower() == "deployment":
            workload_config = servo.connectors.kubernetes.DeploymentConfiguration(
                **workload_kwargs
            )
            main_config_kwargs["deployments"] = [workload_config]

        else:
            raise servo.EventError(
                f"Incompatible workload_kind configured: {self.workload_kind}"
            )

        return servo.connectors.kubernetes.KubernetesConfiguration(
            **main_config_kwargs,
            **kwargs,
        )

    def generate_prometheus_config(
        self, **kwargs
    ) -> servo.connectors.prometheus.PrometheusConfiguration:
        """Generate a configuration for running an Opsani Dev optimization that utilizes
        Prometheus and Envoy sidecars to produce and aggregate the necessary metrics.

        Returns:
            A Prometheus connector configuration object.
        """
        metrics = [
            servo.connectors.prometheus.PrometheusMetric(
                "main_instance_count",
                servo.types.Unit.count,
                query='sum(envoy_cluster_membership_healthy{opsani_role!="tuning"})',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_instance_count",
                servo.types.Unit.count,
                query='envoy_cluster_membership_healthy{opsani_role="tuning"}',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "total_request_rate",
                servo.types.Unit.requests_per_second,
                query="sum(rate(envoy_cluster_upstream_rq_total[1m]))",
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "main_request_rate",
                servo.types.Unit.requests_per_second,
                query='avg(rate(envoy_cluster_upstream_rq_total{opsani_role!="tuning"}[1m]))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_request_rate",
                servo.types.Unit.requests_per_second,
                query='avg(rate(envoy_cluster_upstream_rq_total{opsani_role="tuning"}[1m]))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "main_success_rate",
                servo.types.Unit.requests_per_second,
                query='avg(sum by(kubernetes_pod_name)(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"2|3"}[1m])))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_success_rate",
                servo.types.Unit.requests_per_second,
                query='avg(sum by(kubernetes_pod_name)(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"2|3"}[1m])))',
                absent=servo.connectors.prometheus.AbsentMetricPolicy.zero,
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "main_error_rate",
                servo.types.Unit.requests_per_second,
                query='avg(sum by(kubernetes_pod_name)(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"4|5"}[1m])))',
                absent=servo.connectors.prometheus.AbsentMetricPolicy.zero,
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_error_rate",
                servo.types.Unit.requests_per_second,
                query='avg(sum by(kubernetes_pod_name)(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"4|5"}[1m])))',
                absent=servo.connectors.prometheus.AbsentMetricPolicy.zero,
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "main_p90_latency",
                servo.types.Unit.milliseconds,
                query='avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role!="tuning"}[1m])))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_p90_latency",
                servo.types.Unit.milliseconds,
                query='avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role="tuning"}[1m])))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "main_p50_latency",
                servo.types.Unit.milliseconds,
                query='avg(histogram_quantile(0.5,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role!="tuning"}[1m])))',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_p50_latency",
                servo.types.Unit.milliseconds,
                query='avg(histogram_quantile(0.5,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role="tuning"}[1m])))',
            ),
        ]
        if not self.create_tuning_pod:
            metrics = list(
                filter(lambda m: 'opsani_role="tuning"' not in m.query, metrics)
            )

        return servo.connectors.prometheus.PrometheusConfiguration(
            description="A sidecar configuration for aggregating metrics from Envoy sidecar proxies.",
            base_url=PROMETHEUS_SIDECAR_BASE_URL,
            streaming_interval="10s",
            metrics=metrics,
            **kwargs,
        )

    def generate_kube_metrics_config(
        self, **kwargs
    ) -> servo.connectors.kube_metrics.KubeMetricsConfiguration:
        """Generate a configuration for running an Opsani Dev optimization under servo.connectors.kubernetes.

        Returns:
            A Kubernetes connector configuration object.
        """
        metrics = [
            m.value for m in list(servo.connectors.kube_metrics.SupportedKubeMetrics)
        ]
        if not self.create_tuning_pod:
            metrics = list(filter(lambda m: "tuning" not in m, metrics))

        return servo.connectors.kube_metrics.KubeMetricsConfiguration(
            namespace=self.namespace,
            name=self.workload_name,
            kind=self.workload_kind,
            container=self.container,
            metrics_to_collect=metrics,
            **kwargs,
        )


class OpsaniDevChecks(servo.BaseChecks):
    config: OpsaniDevConfiguration
    optimizer: servo.configuration.OptimizerTypes

    # FIXME make this a property of worklod helper?
    @property
    def required_permissions(self) -> List[servo.connectors.kubernetes.PermissionSet]:
        if self.config.workload_kind.lower() == "deployment":
            return KUBERNETES_PERMISSIONS
        else:
            raise servo.EventError(
                f"Incompatible workload_kind configured: {self.workload_kind}"
            )

    def _get_generated_controller_config(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> servo.connectors.kubernetes.DeploymentConfiguration:
        if self.config.workload_kind.lower() == "deployment":
            return config.deployments[0]
        else:
            raise servo.EventError(
                f"Incompatible workload_kind configured: {self.workload_kind}"
            )

    # NOTE for rollout support, will need to get current replicaset of rollout as target
    async def _get_port_forward_target(self) -> str:
        return f"{self.config.workload_kind}/{self.config.workload_name}"

    @property
    def workload_helper(self) -> type[DeploymentHelper]:
        if self.config.workload_kind.lower() == "deployment":
            return DeploymentHelper
        else:
            raise servo.EventError(
                f"Incompatible workload_kind configured: {self.workload_kind}"
            )

    ##
    # Kubernetes essentials
    @servo.checks.require("Optimizer Configuration")
    def check_optimizer(self) -> None:
        assert isinstance(
            self.optimizer,
            servo.configuration.OpsaniOptimizer,
        ) or isinstance(
            self.optimizer,
            servo.configuration.AppdynamicsOptimizer,
        ), f"Opsani Dev connector is incompatible with non OpsaniOptimizer type {self.optimizer.__class__.__name__}"

    @servo.checks.require("Connectivity to Kubernetes")
    async def check_connectivity(self) -> None:
        async with kubernetes_asyncio.client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            await v1.get_code()

    @servo.checks.warn("Kubernetes version")
    async def check_version(self) -> None:
        async with kubernetes_asyncio.client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            version = await v1.get_code()
            assert int(version.major) >= 1
            # EKS sets minor to "17+"
            assert int(int("".join(c for c in version.minor if c.isdigit()))) >= 16

    @servo.checks.require("Kubernetes permissions")
    async def check_permissions(self) -> None:
        async with kubernetes_asyncio.client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            for permission in self.required_permissions:
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
                        ), f'Not allowed to "{verb}" resource "{resource}"'

    @servo.checks.require('Namespace "{self.config.namespace}" is readable')
    async def check_opsani_dev_kubernetes_namespace(self) -> None:
        await NamespaceHelper.read(self.config.namespace)

    @servo.checks.require(
        '{self.config.workload_kind} "{self.config.workload_name}" is readable'
    )
    async def check_opsani_dev_kubernetes_controller(self) -> None:
        await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )

    @servo.checks.require('Container "{self.config.container}" is readable')
    async def check_opsani_dev_kubernetes_container(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        container = find_container(controller, self.config.container)
        assert (
            container
        ), f"failed reading Container '{self.config.container}' in {self.config.workload_kind} '{self.config.workload_name}'"

    @servo.require('Container "{self.config.container}" has resource requirements')
    async def check_resource_requirements(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        container = find_container(controller, self.config.container)
        assert container
        assert container.resources, "missing container resources"

        # Apply any defaults/overrides for requests/limits from config
        servo.connectors.kubernetes.set_container_resource_defaults_from_config(
            container, self.config
        )

        for resource in servo.connectors.kubernetes.Resource.values():
            current_state = None
            container_requirements = ContainerHelper.get_resource_requirements(
                container, resource
            )
            get_requirements = cast(
                Union[CPU, Memory], getattr(self.config, resource)
            ).get
            for requirement in get_requirements:
                current_state = container_requirements.get(requirement)
                if current_state:
                    break

            assert current_state, (
                f"{self.config.workload_kind} {self.config.workload_name} target container {self.config.container} spec does not define the resource {resource}. "
                f"At least one of the following must be specified: {', '.join(map(lambda req: req.resources_key, get_requirements))}"
            )

    @servo.checks.require("Target container resources fall within optimization range")
    async def check_target_container_resources_within_limits(self) -> None:
        # Load the Controller
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        assert (
            controller
        ), f"failed to read {self.config.workload_kind} '{self.config.workload_name}' in namespace '{self.config.namespace}'"

        # Find the target Container
        target_container = find_container(controller, self.config.container)
        assert (
            target_container
        ), f"failed to find container '{self.config.container}' when verifying resource limits"

        # Apply any defaults/overrides from the config
        servo.connectors.kubernetes.set_container_resource_defaults_from_config(
            target_container, self.config
        )

        # Get resource requirements from container
        # TODO: This needs to reuse the logic from CanaryOptimization class (tuning_cpu, tuning_memory, etc properties)
        cpu_resource_requirements = ContainerHelper.get_resource_requirements(
            target_container, Resource.cpu.value
        )
        cpu_resource_value = cpu_resource_requirements.get(
            next(
                filter(
                    lambda r: cpu_resource_requirements[r] is not None,
                    self.config.cpu.get,
                ),
                None,
            )
        )
        container_cpu_value = servo.connectors.kubernetes.Core.parse(cpu_resource_value)

        memory_resource_requirements = ContainerHelper.get_resource_requirements(
            target_container, Resource.memory.value
        )
        memory_resource_value = memory_resource_requirements.get(
            next(
                filter(
                    lambda r: memory_resource_requirements[r] is not None,
                    self.config.memory.get,
                ),
                None,
            )
        )
        container_memory_value = servo.connectors.kubernetes.ShortByteSize.validate(
            memory_resource_value
        )

        # Get config values
        config_cpu_min = self.config.cpu.min
        config_cpu_max = self.config.cpu.max
        config_memory_min = self.config.memory.min
        config_memory_max = self.config.memory.max

        # Check values against config.
        assert (
            container_cpu_value >= config_cpu_min
        ), f"target container CPU value {container_cpu_value.human_readable()} must be greater than optimizable minimum {config_cpu_min.human_readable()}"
        assert (
            container_cpu_value <= config_cpu_max
        ), f"target container CPU value {container_cpu_value.human_readable()} must be less than optimizable maximum {config_cpu_max.human_readable()}"
        assert (
            container_memory_value >= config_memory_min
        ), f"target container Memory value {container_memory_value.human_readable()} must be greater than optimizable minimum {config_memory_min.human_readable()}"
        assert (
            container_memory_value <= config_memory_max
        ), f"target container Memory value {container_memory_value.human_readable()} must be less than optimizable maximum {config_memory_max.human_readable()}"

    @servo.require(
        '{self.config.workload_kind} "{self.config.workload_name}"  is ready'
    )
    async def check_controller_readiness(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        if not self.workload_helper.is_ready(controller):
            raise RuntimeError(
                f'{self.config.workload_name} "{controller.metadata.name}" is not ready'
            )

    @servo.checks.require("Service {self.config.service} is readable")
    async def check_opsani_dev_kubernetes_service(self) -> None:
        await ServiceHelper.read(self.config.service, self.config.namespace)

    @servo.checks.warn("Service {self.config.service} has compatible type")
    async def check_opsani_dev_kubernetes_service_type(self) -> None:
        service = await ServiceHelper.read(self.config.service, self.config.namespace)
        service_type = service.spec.type
        if not service_type in ("ClusterIP", "LoadBalancer", "NodePort"):
            raise ValueError(
                f"expected service type of ClusterIP, LoadBalancer, or NodePort but found {service_type}"
            )

    @servo.checks.check("Service {self.config.service} has unambiguous target port")
    async def check_opsani_dev_kubernetes_service_port(self) -> None:
        service = await ServiceHelper.read(self.config.service, self.config.namespace)
        if len(service.spec.ports) > 1:
            if not self.config.port:
                raise ValueError(
                    f"service defines more than one port: a `port` (name or number) must be specified in the configuration"
                )

            port = ServiceHelper.find_port(service, self.config.port)
            if not port:
                if isinstance(self.config.port, str):
                    raise LookupError(
                        f"could not find a port named: {self.config.port}"
                    )
                elif isinstance(self.config.port, int):
                    raise LookupError(
                        f"could not find a port numbered: {self.config.port}"
                    )
                else:
                    raise RuntimeError(f"unknown port value: {self.config.port}")
        else:
            port: kubernetes_asyncio.client.V1ServicePort = service.spec.ports[0]

        return (
            f"Service Port: {port.name} {port.port}:{port.target_port}/{port.protocol}"
        )

    @servo.checks.check("Service routes traffic to {self.config.workload_name} Pods")
    async def check_service_routes_traffic_to_controller(self) -> None:
        service = await ServiceHelper.read(self.config.service, self.config.namespace)
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )

        # NOTE: The Service labels should be a subset of the controller labels
        controller_labels: dict[str, str] = controller.spec.selector.match_labels
        service_labels: dict[str, str] = service.spec.selector
        delta = dict(set(service_labels.items()) - set(controller_labels.items()))
        if delta:
            desc = " ".join(map("=".join, delta.items()))
            raise RuntimeError(
                f"Service selector does not match {self.config.workload_kind} labels. Missing labels: {desc}"
            )

    ##
    # Prometheus sidecar

    @servo.checks.require("Prometheus ConfigMap exists")
    async def check_prometheus_config_map(self) -> None:
        namespace = os.getenv("POD_NAMESPACE", self.config.namespace)
        optimizer_subdomain = servo.connectors.kubernetes.dns_subdomainify(
            self.optimizer.name
        )

        # Read optimizer namespaced resources
        names = [f"servo.prometheus-{optimizer_subdomain}", "prometheus-config"]
        config = None
        for name in names:
            try:
                async with kubernetes_asyncio.client.ApiClient() as api:
                    corev1 = kubernetes_asyncio.client.CoreV1Api(api)
                    config = await corev1.read_namespaced_config_map(name, namespace)
                    if config:
                        break
            except kubernetes_asyncio.client.ApiException as e:
                if e.status != 404 or e.reason != "Not Found":
                    raise

        self.logger.trace(f"read Prometheus ConfigMap: {repr(config)}")
        assert (
            config
        ), f"failed: no ConfigMap named '{names}' found in namespace '{namespace}'"

    @servo.checks.check("Prometheus sidecar is running")
    async def check_prometheus_sidecar_exists(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(
                f"no servo pod is running in namespace '{self.config.namespace}'"
            )

        if not find_container(workload=pod, name="prometheus"):
            raise servo.checks.CheckError(
                f"no 'prometheus' container found in pod '{pod.metadata.name}' in namespace '{self.config.namespace}'"
            )

    @servo.checks.check("Prometheus sidecar is ready")
    async def check_prometheus_sidecar_is_ready(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        if not PodHelper.is_ready(pod):
            raise servo.checks.CheckError(f"pod '{pod.metadata.name}' is not ready")

    @servo.checks.warn("Prometheus sidecar is stable")
    async def check_prometheus_restart_count(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        container = find_container(workload=pod, name="prometheus")
        assert container, "could not find a Prometheus sidecar container"
        # TODO PodHelper.get_restart_count
        restart_count = PodHelper.get_restart_count(pod, container_name="prometheus")
        assert (
            restart_count == 0
        ), f"container 'prometheus' in pod '{pod.metadata.name}' has restarted {restart_count} times"

    @servo.checks.require("Prometheus has container port on 9090")
    async def check_prometheus_container_port(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"failed: no servo pod was found")

        container = find_container(workload=pod, name="prometheus")
        assert container, "could not find Prometheus sidecar container"

        assert (
            len(container.ports) == 1
        ), f"expected 1 container port but found {len(container.ports)}"
        port: int = container.ports[0].container_port
        assert (
            port == 9090
        ), f"expected Prometheus container port on 9090 but found {port}"

    @servo.checks.require("Prometheus is accessible")
    async def check_prometheus_is_accessible(self) -> str:
        client = servo.connectors.prometheus.Client(
            base_url=self.config.prometheus_base_url
        )
        await client.list_targets()
        return f"Prometheus is accessible at {self.config.prometheus_base_url}"

    async def _read_servo_pod(self) -> Optional[kubernetes_asyncio.client.V1Pod]:
        return await self._read_servo_pod_from_env() or next(
            reversed(await self._list_servo_pods()), None
        )

    async def _read_servo_pod_from_env(
        self,
    ) -> Optional[kubernetes_asyncio.client.V1Pod]:
        """Reads the servo Pod from Kubernetes by referencing the `POD_NAME` and
        `POD_NAMESPACE` environment variables.

        Returns:
            The Pod object that was read or None if the Pod could not be read.
        """
        pod_name = os.getenv("POD_NAME")
        pod_namespace = os.getenv("POD_NAMESPACE")
        if None in (pod_name, pod_namespace):
            return None

        return await PodHelper.read(pod_name, pod_namespace)

    async def _list_servo_pods(self) -> list[kubernetes_asyncio.client.V1Pod]:
        """Lists all servo pods in the configured namespace.

        Returns:
            A list of servo pods in the configured namespace.
        """
        return await PodHelper.list_pods_with_labels(
            namespace=self.config.namespace,
            match_labels={"app.kubernetes.io/name": "servo"},
        )

    ##
    # Kubernetes Controller edits

    @servo.checks.check("{self.config.workload_name} PodSpec has expected annotations")
    async def check_controller_annotations(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        assert (
            controller
        ), f"failed to read {self.config.workload_kind} '{self.config.workload_name}' in namespace '{self.config.namespace}'"

        # Add optimizer annotation to the static Prometheus values
        required_annotations = PROMETHEUS_ANNOTATION_DEFAULTS.copy()
        required_annotations["servo.opsani.com/optimizer"] = self.optimizer.id

        # NOTE: Only check for annotation keys
        annotations: dict[str, str] = (
            controller.spec.template.metadata.annotations or dict()
        )
        actual_annotations = set(annotations.keys())
        delta = set(required_annotations.keys()).difference(actual_annotations)
        if delta:
            annotations = dict(map(lambda k: (k, required_annotations[k]), delta))
            patch = {"spec": {"template": {"metadata": {"annotations": annotations}}}}
            patch_json = json.dumps(patch, indent=None)
            # NOTE: custom resources don't support strategic merge type. json merge is acceptable for both cases because the patch json doesn't contain lists
            command = (
                f"kubectl --namespace {self.config.namespace}"
                f" patch {self.config.workload_kind} {self.config.workload_name}"
                f" --type='merge' -p '{patch_json}'"
            )
            desc = ", ".join(sorted(delta))
            raise servo.checks.CheckError(
                f"{self.config.workload_kind} '{controller.metadata.name}' is missing annotations: {desc}",
                hint=f"Patch annotations via: `{command}`",
                remedy=lambda: _stream_remedy_command(command),
            )

    @servo.checks.check("{self.config.workload_kind} PodSpec has expected labels")
    async def check_controller_labels(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        assert (
            controller
        ), f"failed to read {self.config.workload_kind} '{self.config.workload_name}' in namespace '{self.config.namespace}'"

        labels = controller.spec.template.metadata.labels
        assert (
            labels
        ), f"{self.config.workload_kind} '{controller.metadata.name}' does not have any labels"
        # Add optimizer label to the static values
        required_labels = ENVOY_SIDECAR_LABELS.copy()
        required_labels[
            "servo.opsani.com/optimizer"
        ] = servo.connectors.kubernetes.dns_labelize(self.optimizer.id)

        # NOTE: Check for exact labels as this isn't configurable
        delta = dict(set(required_labels.items()) - set(labels.items()))
        if delta:
            desc = ", ".join(sorted(map("=".join, delta.items())))
            patch = {"spec": {"template": {"metadata": {"labels": delta}}}}
            patch_json = json.dumps(patch, indent=None)
            # NOTE: custom resources don't support strategic merge type. json merge is acceptable for both cases because the patch json doesn't contain lists
            command = (
                f"kubectl --namespace {self.config.namespace}"
                f" patch {self.config.workload_kind} {self.config.workload_name}"
                f" --type='merge' -p '{patch_json}'"
            )
            raise servo.checks.CheckError(
                f"{self.config.workload_kind} '{controller.metadata.name}' is missing labels: {desc}",
                hint=f"Patch labels via: `{command}`",
                remedy=lambda: _stream_remedy_command(command),
            )

    @servo.checks.require("{self.config.workload_kind} has Envoy sidecar container")
    async def check_controller_envoy_sidecars(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        assert (
            controller
        ), f"failed to read {self.config.workload_kind} '{self.config.workload_name}' in namespace '{self.config.namespace}'"

        # Search the containers list for the sidecar
        if find_container(controller, "opsani-envoy"):
            return

        port_switch = (
            f" --port {self.config.port}" if self.config.port is not None else ""
        )
        command = (
            f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- "
            f"servo inject-sidecar --image {self.config.envoy_sidecar_image} "
            f"--namespace {self.config.namespace} --service {self.config.service}{port_switch} "
            f"{self.config.workload_kind.lower()}/{self.config.workload_name}"
        )
        raise servo.checks.CheckError(
            f"{self.config.workload_kind} '{controller.metadata.name}' pod template spec does not include envoy sidecar container ('opsani-envoy')",
            hint=f"Inject Envoy sidecar container via: `{command}`",
            remedy=lambda: _stream_remedy_command(command),
        )

    @servo.checks.check("Pods have Envoy sidecar containers")
    async def check_pod_envoy_sidecars(self) -> None:
        controller = await self.workload_helper.read(
            self.config.workload_name, self.config.namespace
        )
        assert (
            controller
        ), f"failed to read {self.config.workload_kind} '{self.config.workload_name}' in namespace '{self.config.namespace}'"

        pods_without_sidecars = []
        for pod in await self.workload_helper.get_latest_pods(controller):
            # Search the containers list for the sidecar
            if not find_container(pod, "opsani-envoy"):
                pods_without_sidecars.append(pod)

        if pods_without_sidecars:
            desc = ", ".join(
                map(operator.attrgetter("metadata.name"), pods_without_sidecars)
            )
            raise servo.checks.CheckError(
                f"pods '{desc}' do not have envoy sidecar container ('opsani-envoy')"
            )

    ##
    # Connecting the dots

    @servo.check("Prometheus is discovering targets")
    async def check_prometheus_targets(self) -> None:
        client = servo.connectors.prometheus.Client(
            base_url=self.config.prometheus_base_url
        )
        targets = await client.list_targets()
        assert len(targets.active) > 0, "no active targets were found"

        return f"found {targets.active} active targets"

    @servo.check("Envoy proxies are being scraped")
    async def check_envoy_sidecar_metrics(self) -> str:
        # NOTE: We don't care about the response status code, we just want to see that traffic is being metered by Envoy
        metric = servo.connectors.prometheus.PrometheusMetric(
            "main_request_total",
            servo.types.Unit.requests_per_second,
            query=f'sum(envoy_cluster_upstream_rq_total{{opsani_role!="tuning", kubernetes_namespace="{self.config.namespace}"}})',
        )
        client = servo.connectors.prometheus.Client(
            base_url=self.config.prometheus_base_url
        )
        response = await client.query(metric)
        assert response.data, f"query returned no response data: '{metric.query}'"
        assert (
            response.data.result_type == servo.connectors.prometheus.ResultType.vector
        ), f"expected a vector result but found {response.data.result_type}"
        assert (
            len(response.data) == 1
        ), f"expected Prometheus API to return a single result for metric '{metric.name}' but found {len(response.data)}"
        result = response.data[0]
        _, value = result.value
        if value in {None, 0.0}:
            port_forward_target = await self._get_port_forward_target()
            command = (
                f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- "
                f'sh -c "kubectl port-forward --namespace={self.config.namespace} {port_forward_target} 9980 & '
                f"echo 'GET http://localhost:9980/' | vegeta attack -duration 10s | vegeta report -every 3s\""
            )
            raise servo.checks.CheckError(
                f"Envoy is not reporting any traffic to Prometheus for metric '{metric.name}' ({metric.query})",
                hint=f"Send traffic to your application on port 9980. Try `{command}`",
                remedy=lambda: _stream_remedy_command(command),
            )
        return f"{metric.name}={value}{metric.unit}"

    @servo.checks.require("Traffic is proxied through Envoy")
    async def check_service_proxy(self) -> str:
        proxy_service_port = ENVOY_SIDECAR_DEFAULT_PORT  # TODO: move to configuration
        service = await ServiceHelper.read(self.config.service, self.config.namespace)
        if self.config.port:
            port = ServiceHelper.find_port(service, self.config.port)
        else:
            port = service.spec.ports[0]

        # return if we are already proxying to Envoy
        if port.target_port == proxy_service_port:
            return

        # patch the target port to pass traffic through Envoy
        patch = {
            "spec": {
                "type": service.spec.type,
                "ports": [
                    {
                        "protocol": "TCP",
                        "name": port.name,
                        "port": port.port,
                        "targetPort": proxy_service_port,
                    }
                ],
            }
        }
        patch_json = json.dumps(patch, indent=None)
        command = f"kubectl --namespace {self.config.namespace} patch service {self.config.service} -p '{patch_json}'"
        raise servo.checks.CheckError(
            f"service '{service.metadata.name}' is not routing traffic through Envoy sidecar on port {proxy_service_port}",
            hint=f"Update target port via: `{command}`",
            remedy=lambda: _stream_remedy_command(command),
        )

    @servo.check("Tuning pod is running")
    async def check_tuning_is_running(self) -> None:
        if self.config.create_tuning_pod:
            # Generate a KubernetesConfiguration to initialize the optimization class
            kubernetes_config = self.config.generate_kubernetes_config()
            controller_config = self._get_generated_controller_config(kubernetes_config)
            optimization = await servo.connectors.kubernetes.CanaryOptimization.create(
                controller_config, timeout=kubernetes_config.timeout
            )

            # Ensure the tuning pod is available
            try:
                if optimization.tuning_pod is None:
                    servo.logger.info(
                        f"Creating tuning pod '{optimization.tuning_pod_name}'"
                    )
                    await optimization.create_tuning_pod()
                else:
                    servo.logger.info(
                        f"Found existing tuning pod '{optimization.tuning_pod_name}'"
                    )

            except Exception as error:
                servo.logger.exception("Failed creating tuning Pod: {error}")
                raise servo.checks.CheckError(
                    f"could not find tuning pod '{optimization.tuning_pod_name}''"
                ) from error

        else:
            servo.logger.info(
                f"Skipping tuning pod check as create_tuning_pod is disabled"
            )

    @servo.check("Pods are processing traffic")
    async def check_traffic_metrics(self) -> str:
        metrics = [
            servo.connectors.prometheus.PrometheusMetric(
                "main_request_total",
                servo.types.Unit.requests_per_second,
                query=f'sum(envoy_cluster_upstream_rq_total{{opsani_role!="tuning", kubernetes_namespace="{self.config.namespace}"}})',
            ),
            servo.connectors.prometheus.PrometheusMetric(
                "tuning_request_total",
                servo.types.Unit.requests_per_second,
                query=f'sum(envoy_cluster_upstream_rq_total{{opsani_role="tuning", kubernetes_namespace="{self.config.namespace}"}})',
            ),
        ]
        client = servo.connectors.prometheus.Client(
            base_url=self.config.prometheus_base_url
        )
        summaries = []
        for metric in metrics:
            response = await client.query(metric)
            assert (
                response.data.result_type
                == servo.connectors.prometheus.ResultType.vector
            ), f"expected a vector result but found {response.data.result_type}"

            assert (
                len(response.data) == 1
            ), f"expected Prometheus API to return a single result for metric '{metric.name}' but found {len(response.data)}"

            result = response.data[0]
            value = result.value[1]
            assert (
                value is not None and value > 0.0
            ), f"Envoy is reporting a value of {value} which is not greater than zero for metric '{metric.name}' ({metric.query})"

            summaries.append(f"{metric.name}={value}{metric.unit}")
            return ", ".join(summaries)

    @property
    def _servo_resource_target(self) -> str:
        if pod_name := os.environ.get("POD_NAME"):
            return f"pods/{pod_name}"
        else:
            return "deployment/servo"


@servo.metadata(
    description="Optimize a single service via a tuning instance and an Envoy sidecar",
    version="2.0.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class OpsaniDevConnector(servo.BaseConnector):
    """Opsani Dev is a turnkey solution for optimizing a single service."""

    config: OpsaniDevConfiguration

    @servo.on_event()
    async def attach(self, servo_: servo.Servo) -> None:
        # FIXME figure out why servo.events.MetaClass is screwing with Servo type hinting
        await servo_.add_connector(
            "opsani-dev:kubernetes",
            servo.connectors.kubernetes.KubernetesConnector(
                optimizer=self.optimizer,
                config=self.config.generate_kubernetes_config(),
            ),
        )
        await servo_.add_connector(
            "opsani-dev:prometheus",
            servo.connectors.prometheus.PrometheusConnector(
                optimizer=self.optimizer,
                config=self.config.generate_prometheus_config(),
            ),
        )
        km_config = self.config.generate_kube_metrics_config()
        # NOTE: connector should technically be attached prior to running checks but k8s connector attached above takes care of necessary setup for check
        if (
            check := await servo.connectors.kube_metrics.KubeMetricsChecks(
                config=km_config
            ).run_one(id="check_metrics_api")
        ).success:
            await servo_.add_connector(
                "opsani-dev:kube-metrics",
                servo.connectors.kube_metrics.KubeMetricsConnector(
                    optimizer=self.optimizer, config=km_config
                ),
            )
        else:
            self.logger.warning(
                f"Omitting kube_metrics connector from opsani_dev assembly due to failed check {check.name}: {check.message}"
            )
            self.logger.opt(exception=check.exception).debug(
                "Failed kube_metrics check exception"
            )

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        return await OpsaniDevChecks(
            config=self.config, optimizer=self.optimizer
        ).run_all(matching=matching, halt_on=halt_on)


async def _stream_remedy_command(command: str) -> None:
    await servo.utilities.subprocess.stream_subprocess_shell(
        command,
        stdout_callback=lambda msg: servo.logger.debug(f"[stdout] {msg}"),
        stderr_callback=lambda msg: servo.logger.warning(f"[stderr] {msg}"),
    )
