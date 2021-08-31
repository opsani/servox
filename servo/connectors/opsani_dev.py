import abc
import json
import operator
import os
from typing import Dict, List, Optional, Union, Type

import kubernetes_asyncio
import pydantic

import servo
import servo.connectors.kubernetes
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
        resources=["pods", "pods/logs", "pods/status", "pods/exec", "pods/portforward", "services"],
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
ENVOY_SIDECAR_LABELS = {
    "sidecar.opsani.com/type": "envoy"
}
ENVOY_SIDECAR_DEFAULT_PORT = 9980

class CPU(servo.connectors.kubernetes.CPU):
    step: servo.connectors.kubernetes.Millicore = "125m"

class Memory(servo.connectors.kubernetes.Memory):
    step: servo.connectors.kubernetes.ShortByteSize = "128 MiB"

class OpsaniDevConfiguration(servo.BaseConfiguration):
    namespace: str
    deployment: Optional[str]
    rollout: Optional[str]
    container: str
    service: str
    port: Optional[Union[pydantic.StrictInt, str]] = None
    cpu: CPU
    memory: Memory
    prometheus_base_url: str = PROMETHEUS_SIDECAR_BASE_URL
    timeout: servo.Duration = "5m"
    settlement: Optional[servo.Duration] = pydantic.Field(
        description="Duration to observe the application after an adjust to ensure the deployment is stable. May be overridden by optimizer supplied `control.adjust.settlement` value."
    )

    @pydantic.root_validator
    def check_deployment_and_rollout(cls, values):
        if values.get('deployment') is not None and values.get('rollout') is not None:
            raise ValueError("Configuration cannot specify both rollout and deployment")

        if values.get('deployment') is None and values.get('rollout') is None:
            raise ValueError("Configuration must specify either rollout or deployment")

        return values

    @classmethod
    def generate(cls, **kwargs) -> "OpsaniDevConfiguration":
        return cls(
            namespace="default",
            deployment="app-deployment",
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
        main_config = servo.connectors.kubernetes.DeploymentConfiguration(
            name=(self.deployment or self.rollout),
            strategy=servo.connectors.kubernetes.CanaryOptimizationStrategyConfiguration(
                type=servo.connectors.kubernetes.OptimizationStrategy.canary,
                alias="tuning"
            ),
            replicas=servo.Replicas(
                min=0,
                max=1,
            ),
            containers=[
                servo.connectors.kubernetes.ContainerConfiguration(
                    name=self.container,
                    alias="main",
                    cpu=self.cpu,
                    memory=self.memory,
                )
            ],
        )
        if self.deployment:
            main_arg = { 'deployments': [ main_config ] }
        elif self.rollout:
            main_arg = { 'rollouts': [ servo.connectors.kubernetes.RolloutConfiguration.parse_obj(
                main_config.dict(exclude_none=True)
            ) ] }

        return servo.connectors.kubernetes.KubernetesConfiguration(
            namespace=self.namespace,
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            timeout=self.timeout,
            settlement=self.settlement,
            **main_arg,
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
        return servo.connectors.prometheus.PrometheusConfiguration(
            description="A sidecar configuration for aggregating metrics from Envoy sidecar proxies.",
            base_url=PROMETHEUS_SIDECAR_BASE_URL,
            streaming_interval='10s',
            metrics=[
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
                    query='avg(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"2|3"}[1m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_success_rate",
                    servo.types.Unit.requests_per_second,
                    query='avg(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"2|3"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_error_rate",
                    servo.types.Unit.requests_per_second,
                    query='avg(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"4|5"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_error_rate",
                    servo.types.Unit.requests_per_second,
                    query='avg(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"4|5"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
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
            ],
            **kwargs,
        )

class BaseOpsaniDevChecks(servo.BaseChecks, abc.ABC):
    config: OpsaniDevConfiguration

    @property
    @abc.abstractmethod
    def controller_type_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def config_controller_name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def controller_class(self) -> Type[Union[
        servo.connectors.kubernetes.Deployment,
        servo.connectors.kubernetes.Rollout
    ]]:
        ...

    @property
    @abc.abstractmethod
    def required_permissions(self) -> List[servo.connectors.kubernetes.PermissionSet]:
        ...

    @abc.abstractmethod
    async def _get_port_forward_target(self) -> str:
        ...

    @abc.abstractmethod
    def _get_generated_controller_config(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> Union[
        servo.connectors.kubernetes.DeploymentConfiguration,
        servo.connectors.kubernetes.RolloutConfiguration
    ]:
        ...

    @abc.abstractmethod
    def _get_controller_service_selector(self, controller: Union[
        servo.connectors.kubernetes.Deployment,
        servo.connectors.kubernetes.Rollout
    ]) -> Dict[str, str]:
        ...

    ##
    # Kubernetes essentials

    @servo.checks.require("Connectivity to Kubernetes")
    async def check_connectivity(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            await v1.get_code()

    @servo.checks.warn("Kubernetes version")
    async def check_version(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            version = await v1.get_code()
            assert int(version.major) >= 1
            # EKS sets minor to "17+"
            assert int(int("".join(c for c in version.minor if c.isdigit()))) >= 16

    @servo.checks.require("Kubernetes permissions")
    async def check_permissions(self) -> None:
        async with kubernetes_asyncio.client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.AuthorizationV1Api(api)
            for permission in self.required_permissions:
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

    @servo.checks.require('Namespace "{self.config.namespace}" is readable')
    async def check_kubernetes_namespace(self) -> None:
        await servo.connectors.kubernetes.Namespace.read(self.config.namespace)

    @servo.checks.require('{self.controller_type_name} "{self.config_controller_name}" is readable')
    async def check_kubernetes_controller(self) -> None:
        await self.controller_class.read(self.config_controller_name, self.config.namespace)

    @servo.checks.require('Container "{self.config.container}" is readable')
    async def check_kubernetes_container(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        container = controller.find_container(self.config.container)
        assert (
            container
        ), f"failed reading Container '{self.config.container}' in {self.controller_type_name} '{self.config_controller_name}'"

    @servo.require('Container "{self.config.container}" has resource requirements')
    async def check_resource_requirements(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        container = controller.find_container(self.config.container)
        assert container
        assert container.resources, "missing container resources"

        # Apply any defaults/overrides for requests/limits from config
        servo.connectors.kubernetes.set_container_resource_defaults_from_config(container, self.config)

        for resource in servo.connectors.kubernetes.Resource.values():
            current_state = None
            container_requirements = container.get_resource_requirements(resource)
            get_requirements = getattr(self.config, resource).get
            for requirement in get_requirements:
                current_state = container_requirements.get(requirement)
                if current_state:
                    break

            assert current_state, (
                f"{self.controller_type_name} {self.config_controller_name} target container {self.config.container} spec does not define the resource {resource}. "
                f"At least one of the following must be specified: {', '.join(map(lambda req: req.resources_key, get_requirements))}"
            )

    @servo.checks.require("Target container resources fall within optimization range")
    async def check_target_container_resources_within_limits(self) -> None:
        # Load the Controller
        controller = await self.controller_class.read(
            self.config_controller_name,
            self.config.namespace
        )
        assert controller, f"failed to read {self.controller_type_name} '{self.config_controller_name}' in namespace '{self.config.namespace}'"

        # Find the target Container
        target_container = next(filter(lambda c: c.name == self.config.container, controller.containers), None)
        assert target_container, f"failed to find container '{self.config.container}' when verifying resource limits"

        # Apply any defaults/overrides from the config
        servo.connectors.kubernetes.set_container_resource_defaults_from_config(target_container, self.config)

        # Get resource requirements from container
        # TODO: This needs to reuse the logic from CanaryOptimization class (tuning_cpu, tuning_memory, etc properties)
        cpu_resource_requirements = target_container.get_resource_requirements('cpu')
        cpu_resource_value = cpu_resource_requirements.get(
            next(filter(lambda r: cpu_resource_requirements[r] is not None, self.config.cpu.get), None)
        )
        container_cpu_value = servo.connectors.kubernetes.Millicore.parse(cpu_resource_value)

        memory_resource_requirements = target_container.get_resource_requirements('memory')
        memory_resource_value = memory_resource_requirements.get(
            next(filter(lambda r: memory_resource_requirements[r] is not None, self.config.memory.get), None)
        )
        container_memory_value = servo.connectors.kubernetes.ShortByteSize.validate(memory_resource_value)

        # Get config values
        config_cpu_min = self.config.cpu.min
        config_cpu_max = self.config.cpu.max
        config_memory_min = self.config.memory.min
        config_memory_max = self.config.memory.max

        # Check values against config.
        assert container_cpu_value >= config_cpu_min, f"target container CPU value {container_cpu_value.human_readable()} must be greater than optimizable minimum {config_cpu_min.human_readable()}"
        assert container_cpu_value <= config_cpu_max, f"target container CPU value {container_cpu_value.human_readable()} must be less than optimizable maximum {config_cpu_max.human_readable()}"
        assert container_memory_value >= config_memory_min, f"target container Memory value {container_memory_value.human_readable()} must be greater than optimizable minimum {config_memory_min.human_readable()}"
        assert container_memory_value <= config_memory_max, f"target container Memory value {container_memory_value.human_readable()} must be less than optimizable maximum {config_memory_max.human_readable()}"

    @servo.require('{self.controller_type_name} "{self.config_controller_name}"  is ready')
    async def check_controller_readiness(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        if not await controller.is_ready():
            raise RuntimeError(f'{self.controller_type_name} "{controller.name}" is not ready')

    @servo.checks.require("service")
    async def check_kubernetes_service(self) -> None:
        await servo.connectors.kubernetes.Service.read(self.config.service, self.config.namespace)

    @servo.checks.warn("service type")
    async def check_kubernetes_service_type(self) -> None:
        service = await servo.connectors.kubernetes.Service.read(
            self.config.service, self.config.namespace
        )
        service_type = service.obj.spec.type
        if not service_type in ("ClusterIP", "LoadBalancer", "NodePort"):
            raise ValueError(
                f"expected service type of ClusterIP, LoadBalancer, or NodePort but found {service_type}"
            )

    @servo.checks.check("service port")
    async def check_kubernetes_service_port(self) -> None:
        service = await servo.connectors.kubernetes.Service.read(
            self.config.service, self.config.namespace
        )
        if len(service.ports) > 1:
            if not self.config.port:
                raise ValueError(
                    f"service defines more than one port: a `port` (name or number) must be specified in the configuration"
                )

            port = service.find_port(self.config.port)
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
            port = service.ports[0]

        return f"Service Port: {port.name} {port.port}:{port.target_port}/{port.protocol}"

    @servo.checks.check('Service routes traffic to {self.controller_type_name} Pods')
    async def check_service_routes_traffic_to_controller(self) -> None:
        service = await servo.connectors.kubernetes.Service.read(
            self.config.service, self.config.namespace
        )
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)

        # NOTE: The Service labels should be a subset of the controller labels
        controller_labels = self._get_controller_service_selector(controller)
        delta = dict(set(service.selector.items()) - set(controller_labels.items()))
        if delta:
            desc = ' '.join(map('='.join, delta.items()))
            raise RuntimeError(f"Service selector does not match {self.controller_type_name} labels. Missing labels: {desc}")

    ##
    # Prometheus sidecar

    @servo.checks.require("Prometheus ConfigMap exists")
    async def check_prometheus_config_map(self) -> None:
        namespace = os.getenv("POD_NAMESPACE", self.config.namespace)
        optimizer_subdomain = servo.connectors.kubernetes.dns_subdomainify(self.config.optimizer.name)

        # Read optimizer namespaced resources
        names = [f'servo.prometheus-{optimizer_subdomain}', 'prometheus-config']
        for name in names:
            try:
                config = await servo.connectors.kubernetes.ConfigMap.read(
                    name, namespace
                )
                if config:
                    break
            except kubernetes_asyncio.client.exceptions.ApiException as e:
                if e.status != 404 or e.reason != "Not Found":
                    raise

        self.logger.trace(f"read Prometheus ConfigMap: {repr(config)}")
        assert config, f"failed: no ConfigMap named '{names}' found in namespace '{namespace}'"

    @servo.checks.check("Prometheus sidecar is running")
    async def check_prometheus_sidecar_exists(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod is running in namespace '{self.config.namespace}'")

        if not pod.get_container("prometheus"):
            raise servo.checks.CheckError(
                f"no 'prometheus' container found in pod '{pod.name}' in namespace '{self.config.namespace}'"
            )

    @servo.checks.check("Prometheus sidecar is ready")
    async def check_prometheus_sidecar_is_ready(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        if not await pod.is_ready():
            raise servo.checks.CheckError(f"pod '{pod.name}' is not ready")

    @servo.checks.warn("Prometheus sidecar is stable")
    async def check_prometheus_restart_count(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        container = pod.get_container("prometheus")
        assert container, "could not find a Prometheus sidecar container"
        restart_count = await container.get_restart_count()
        assert (
            restart_count == 0
        ), f"container 'prometheus' in pod '{pod.name}' has restarted {restart_count} times"

    @servo.checks.require("Prometheus has container port on 9090")
    async def check_prometheus_container_port(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"failed: no servo pod was found")

        container = pod.get_container("prometheus")
        assert container, "could not find Prometheus sidecar container"

        assert (
            len(container.obj.ports) == 1
        ), f"expected 1 container port but found {len(container.obj.ports)}"
        port = container.obj.ports[0].container_port
        assert (
            port == 9090
        ), f"expected Prometheus container port on 9090 but found {port}"

    @servo.checks.require("Prometheus is accessible")
    async def check_prometheus_is_accessible(self) -> str:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        container = pod.get_container("prometheus")
        assert container, "could not find a Prometheus sidecar container"
        assert (
            len(container.obj.ports) == 1
        ), f"expected 1 container port but found {len(container.obj.ports)}"

        client = servo.connectors.prometheus.Client(base_url=self.config.prometheus_base_url)
        await client.list_targets()
        return f"Prometheus is accessible at {self.config.prometheus_base_url}"

    async def _read_servo_pod(self) -> Optional[servo.connectors.kubernetes.Pod]:
        return await self._read_servo_pod_from_env() or next(
            reversed(await self._list_servo_pods()), None
        )

    async def _read_servo_pod_from_env(self) -> Optional[servo.connectors.kubernetes.Pod]:
        """Reads the servo Pod from Kubernetes by referencing the `POD_NAME` and
        `POD_NAMESPACE` environment variables.

        Returns:
            The Pod object that was read or None if the Pod could not be read.
        """
        pod_name = os.getenv("POD_NAME")
        pod_namespace = os.getenv("POD_NAMESPACE")
        if None in (pod_name, pod_namespace):
            return None

        return await servo.connectors.kubernetes.Pod.read(pod_name, pod_namespace)

    async def _list_servo_pods(self) -> List[servo.connectors.kubernetes.Pod]:
        """Lists all servo pods in the configured namespace.

        Returns:
            A list of servo pods in the configured namespace.
        """
        async with servo.connectors.kubernetes.Pod.preferred_client() as api_client:
            label_selector = servo.connectors.kubernetes.selector_string(
                {"app.kubernetes.io/name": "servo"}
            )
            pod_list: servo.connectors.kubernetes.client.V1PodList = (
                await api_client.list_namespaced_pod(
                    namespace=self.config.namespace, label_selector=label_selector
                )
            )

        pods = [servo.connectors.kubernetes.Pod(p) for p in pod_list.items]
        return pods

    ##
    # Kubernetes Controller edits

    @servo.checks.require("{self.controller_type_name} PodSpec has expected annotations")
    async def check_controller_annotations(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        assert controller, f"failed to read {self.controller_type_name} '{self.config_controller_name}' in namespace '{self.config.namespace}'"

        # Add optimizer annotation to the static Prometheus values
        required_annotations = PROMETHEUS_ANNOTATION_DEFAULTS.copy()
        required_annotations['servo.opsani.com/optimizer'] = self.config.optimizer.id

        # NOTE: Only check for annotation keys
        annotations = controller.pod_template_spec.metadata.annotations or dict()
        actual_annotations = set(annotations.keys())
        delta = set(required_annotations.keys()).difference(actual_annotations)
        if delta:
            annotations = dict(map(lambda k: (k, required_annotations[k]), delta))
            patch = {"spec": {"template": {"metadata": {"annotations": annotations}}}}
            patch_json = json.dumps(patch, indent=None)
            # NOTE: custom resources don't support strategic merge type. json merge is acceptable for both cases because the patch json doesn't contain lists
            command = f"kubectl --namespace {self.config.namespace} patch {self.controller_type_name} {self.config_controller_name} --type='merge' -p '{patch_json}'"
            desc = ', '.join(sorted(delta))
            raise servo.checks.CheckError(
                f"{self.controller_type_name} '{controller.name}' is missing annotations: {desc}",
                hint=f"Patch annotations via: `{command}`",
                remedy=lambda: _stream_remedy_command(command)
            )

    @servo.checks.require("{self.controller_type_name} PodSpec has expected labels")
    async def check_controller_labels(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        assert controller, f"failed to read {self.controller_type_name} '{self.config_controller_name}' in namespace '{self.config.namespace}'"

        labels = controller.pod_template_spec.metadata.labels
        assert labels, f"{self.controller_type_name} '{controller.name}' does not have any labels"
        # Add optimizer label to the static values
        required_labels = ENVOY_SIDECAR_LABELS.copy()
        required_labels['servo.opsani.com/optimizer'] = servo.connectors.kubernetes.dns_labelize(self.config.optimizer.id)

        # NOTE: Check for exact labels as this isn't configurable
        delta = dict(set(required_labels.items()) - set(labels.items()))
        if delta:
            desc = ', '.join(sorted(map('='.join, delta.items())))
            patch = {"spec": {"template": {"metadata": {"labels": delta}}}}
            patch_json = json.dumps(patch, indent=None)
            # NOTE: custom resources don't support strategic merge type. json merge is acceptable for both cases because the patch json doesn't contain lists
            command = f"kubectl --namespace {self.config.namespace} patch {self.controller_type_name} {controller.name} --type='merge' -p '{patch_json}'"
            raise servo.checks.CheckError(
                f"{self.controller_type_name} '{controller.name}' is missing labels: {desc}",
                hint=f"Patch labels via: `{command}`",
                remedy=lambda: _stream_remedy_command(command)
            )

    @servo.checks.require("{self.controller_type_name} has Envoy sidecar container")
    async def check_controller_envoy_sidecars(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        assert controller, f"failed to read {self.controller_type_name} '{self.config_controller_name}' in namespace '{self.config.namespace}'"

        # Search the containers list for the sidecar
        for container in controller.containers:
            if container.name == "opsani-envoy":
                return

        port_switch = (
            f" --port {self.config.port}" if self.config.port is not None
            else ''
        )
        command = (
            f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- "
            f"servo --token-file /servo/opsani.token inject-sidecar --namespace {self.config.namespace} --service {self.config.service}{port_switch} "
            f"{self.controller_type_name.lower()}/{self.config_controller_name}"
        )
        raise servo.checks.CheckError(
            f"{self.controller_type_name} '{controller.name}' pod template spec does not include envoy sidecar container ('opsani-envoy')",
            hint=f"Inject Envoy sidecar container via: `{command}`",
            remedy=lambda: _stream_remedy_command(command)
        )

    @servo.checks.require("Pods have Envoy sidecar containers")
    async def check_pod_envoy_sidecars(self) -> None:
        controller = await self.controller_class.read(self.config_controller_name, self.config.namespace)
        assert controller, f"failed to read {self.controller_type_name} '{self.config_controller_name}' in namespace '{self.config.namespace}'"

        pods_without_sidecars = []
        for pod in await controller.get_pods():
            # Search the containers list for the sidecar
            if not pod.get_container('opsani-envoy'):
                pods_without_sidecars.append(pod)

        if pods_without_sidecars:
            desc = ', '.join(map(operator.attrgetter("name"), pods_without_sidecars))
            raise servo.checks.CheckError(f"pods '{desc}' do not have envoy sidecar container ('opsani-envoy')")

    ##
    # Connecting the dots

    @servo.require("Prometheus is discovering targets")
    async def check_prometheus_targets(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise servo.checks.CheckError(f"no servo pod was found")

        container = pod.get_container("prometheus")
        assert container, "could not find a Prometheus sidecar container"
        assert (
            len(container.obj.ports) == 1
        ), f"expected 1 container port but found {len(container.obj.ports)}"

        client = servo.connectors.prometheus.Client(base_url=self.config.prometheus_base_url)
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
        client = servo.connectors.prometheus.Client(base_url=self.config.prometheus_base_url)
        response = await client.query(metric)
        assert response.data, f"query returned no response data: '{metric.query}'"
        assert response.data.result_type == servo.connectors.prometheus.ResultType.vector, f"expected a vector result but found {response.data.result_type}"
        assert len(response.data) == 1, f"expected Prometheus API to return a single result for metric '{metric.name}' but found {len(response.data)}"
        result = response.data[0]
        _, value = result.value
        if value in {None, 0.0}:
            port_forward_target = await self._get_port_forward_target()
            command = (
                f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- "
                f"sh -c \"kubectl port-forward --namespace={self.config.namespace} {port_forward_target} 9980 & "
                f"echo 'GET http://localhost:9980/' | vegeta attack -duration 10s | vegeta report -every 3s\""
            )
            raise servo.checks.CheckError(
                f"Envoy is not reporting any traffic to Prometheus for metric '{metric.name}' ({metric.query})",
                hint=f"Send traffic to your application on port 9980. Try `{command}`",
                remedy=lambda: _stream_remedy_command(command)
            )
        return f"{metric.name}={value}{metric.unit}"

    @servo.check("Traffic is proxied through Envoy")
    async def check_service_proxy(self) -> str:
        proxy_service_port = ENVOY_SIDECAR_DEFAULT_PORT  # TODO: move to configuration
        service = await servo.connectors.kubernetes.Service.read(self.config.service, self.config.namespace)
        if self.config.port:
            port = service.find_port(self.config.port)
        else:
            port = service.ports[0]

        # return if we are already proxying to Envoy
        if port.target_port == proxy_service_port:
            return

        # patch the target port to pass traffic through Envoy
        patch = {"spec": { "type": service.obj.spec.type, "ports": [ {"protocol": "TCP", "name": port.name, "port": port.port, "targetPort": proxy_service_port }]}}
        patch_json = json.dumps(patch, indent=None)
        command = f"kubectl --namespace {self.config.namespace} patch service {self.config.service} -p '{patch_json}'"
        raise servo.checks.CheckError(
            f"service '{service.name}' is not routing traffic through Envoy sidecar on port {proxy_service_port}",
            hint=f"Update target port via: `{command}`",
            remedy=lambda: _stream_remedy_command(command)
        )

    @servo.check("Tuning pod is running")
    async def check_tuning_is_running(self) -> None:
        # Generate a KubernetesConfiguration to initialize the optimization class
        kubernetes_config = self.config.generate_kubernetes_config()
        controller_config = self._get_generated_controller_config(kubernetes_config)
        optimization = await servo.connectors.kubernetes.CanaryOptimization.create(
            controller_config, timeout=kubernetes_config.timeout
        )

        # Ensure the tuning pod is available
        try:
            if optimization.tuning_pod is None:
                servo.logger.info(f"Creating tuning pod '{optimization.tuning_pod_name}'")
                await optimization.create_tuning_pod()
            else:
                servo.logger.info(f"Found existing tuning pod '{optimization.tuning_pod_name}'")

        except Exception as error:
            servo.logger.exception("Failed creating tuning Pod: {error}")
            raise servo.checks.CheckError(
                f"could not find tuning pod '{optimization.tuning_pod_name}''"
            ) from error

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
                query=f'sum(envoy_cluster_upstream_rq_total{{opsani_role="tuning", kubernetes_namespace="{self.config.namespace}"}})'
            ),
        ]
        client = servo.connectors.prometheus.Client(base_url=self.config.prometheus_base_url)
        summaries = []
        for metric in metrics:
            response = await client.query(metric)
            assert response.data.result_type == servo.connectors.prometheus.ResultType.vector, f"expected a vector result but found {response.data.result_type}"

            assert len(response.data) == 1, f"expected Prometheus API to return a single result for metric '{metric.name}' but found {len(response.data)}"

            result = response.data[0]
            value = result.value[1]
            assert value is not None and value > 0.0, f"Envoy is reporting a value of {value} which is not greater than zero for metric '{metric.name}' ({metric.query})"

            summaries.append(f"{metric.name}={value}{metric.unit}")
            return ", ".join(summaries)

    @property
    def _servo_resource_target(self) -> str:
        if pod_name := os.environ.get('POD_NAME'):
            return f"pods/{pod_name}"
        else:
            return "deployment/servo"

class OpsaniDevChecks(BaseOpsaniDevChecks):
    """Opsani dev checks against standard kubernetes Deployments"""

    @property
    def controller_type_name(self) -> str:
        return "Deployment"

    @property
    def config_controller_name(self) -> str:
        return self.config.deployment

    @property
    def controller_class(self) -> Type[servo.connectors.kubernetes.Deployment]:
        return servo.connectors.kubernetes.Deployment

    @property
    def required_permissions(self) -> List[servo.connectors.kubernetes.PermissionSet]:
        return KUBERNETES_PERMISSIONS

    async def _get_port_forward_target(self) -> str:
        return f"deploy/{self.config.deployment}"

    def _get_generated_controller_config(
        self,
        config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> servo.connectors.kubernetes.DeploymentConfiguration:
        return config.deployments[0]

    def _get_controller_service_selector(self, controller: servo.connectors.kubernetes.Deployment) -> Dict[str, str]:
        return controller.obj.spec.selector.match_labels

class OpsaniDevRolloutChecks(BaseOpsaniDevChecks):
    """Opsani dev checks against argoproj.io Rollouts"""
    @property
    def controller_type_name(self) -> str:
        return "Rollout"

    @property
    def config_controller_name(self) -> str:
        return self.config.rollout

    @property
    def controller_class(self) -> Type[servo.connectors.kubernetes.Rollout]:
        return servo.connectors.kubernetes.Rollout

    @property
    def required_permissions(self) -> List[servo.connectors.kubernetes.PermissionSet]:
        return KUBERNETES_PERMISSIONS + [servo.connectors.kubernetes.PermissionSet(
            group="argoproj.io",
            resources=["rollouts", "rollouts/status"],
            verbs=["get", "list", "watch", "update", "patch"],
        )]

    async def _get_port_forward_target(self) -> str:
        # NOTE rollouts don't support kubectl port-forward, have to target the current replicaset instead
        rollout = await servo.connectors.kubernetes.Rollout.read(
            self.config.rollout,
            self.config.namespace
        )
        assert rollout, f"failed to read rollout '{self.config.rollout}' in namespace '{self.config.namespace}'"
        assert rollout.status, f"unable to verify envoy proxy. rollout '{self.config.rollout}' in namespace '{self.config.namespace}' has no status"
        assert rollout.status.current_pod_hash, f"unable to verify envoy proxy. rollout '{self.config.rollout}' in namespace '{self.config.namespace}' has no currentPodHash"
        return f"replicaset/{rollout.name}-{rollout.status.current_pod_hash}"

    def _get_generated_controller_config(
        self,
        config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> servo.connectors.kubernetes.RolloutConfiguration:
        return config.rollouts[0]

    def _get_controller_service_selector(self, controller: servo.connectors.kubernetes.Rollout) -> Dict[str, str]:
        match_labels = dict(controller.obj.spec.selector.match_labels)
        assert controller.status, f"unable to determine service selector. rollout '{self.config.rollout}' in namespace '{self.config.namespace}' has no status"
        assert controller.status.current_pod_hash, f"unable to determine service selector. rollout '{self.config.rollout}' in namespace '{self.config.namespace}' has no currentPodHash"
        match_labels['rollouts-pod-template-hash'] = controller.status.current_pod_hash
        return match_labels

    @servo.checks.require("Rollout Selector and PodSpec has opsani_role label")
    async def check_rollout_selector_labels(self) -> None:
        if os.environ.get("POD_NAME") and os.environ.get("POD_NAMESPACE"):
            return # Setting owner reference to servo should prevent tuning pod from being adopted by the rollout controller

        rollout = await servo.connectors.kubernetes.Rollout.read(self.config.rollout, self.config.namespace)
        assert rollout, f"failed to read Rollout '{self.config.rollout}' in namespace '{self.config.namespace}'"

        spec_patch = {}
        match_labels = rollout.obj.spec.selector.match_labels or dict()
        opsani_role_selector = match_labels.get("opsani_role")
        if opsani_role_selector is None or opsani_role_selector == "tuning":
            opsani_role_selector = "mainline"
            spec_patch["selector"] = {"matchLabels": {"opsani_role": opsani_role_selector}}

        labels = rollout.pod_template_spec.metadata.labels or dict()
        opsani_role_label = labels.get("opsani_role")
        if opsani_role_label is None or opsani_role_label == "tuning" or opsani_role_label != opsani_role_selector:
            spec_patch["template"] = {"metadata": {"labels": {"opsani_role": opsani_role_selector }}}

        if spec_patch: # Check failed if spec needs patching
            patch = {"spec": spec_patch}
            patch_json = json.dumps(patch, indent=None)
            # NOTE: custom resources don't support strategic merge type. json merge is acceptable because the patch json doesn't contain lists
            command = f"kubectl --namespace {self.config.namespace} patch rollout {self.config.rollout} --type='merge' -p '{patch_json}'"
            replicasets = [ f"rs/{rollout.name}-{rollout.status.current_pod_hash}" ]
            if rollout.status.stable_RS and rollout.status.stable_RS != rollout.status.current_pod_hash:
                replicasets.append(f"rs/{rollout.name}-{rollout.status.stable_RS}")
            raise servo.checks.CheckError(
                (
                    f"Rollout '{self.config.rollout}' has missing/mismatched opsani_role selector and/or label."
                    " Label opsani_role with value != \"tuning\" is required to prevent the rollout controller from adopting and destroying the tuning pod"
                ),
                hint=(
                    f"NOTE: Running this patch will require that you manually scale down or delete the replicaset(s) ({', '.join(replicasets)})"
                    f" orphaned by the selector update. Patch selector and labels via: `{command}`"
                )
            )

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

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
    ) -> List[servo.Check]:
        if self.config.deployment:
            checks_class = OpsaniDevChecks
        elif self.config.rollout:
            checks_class = OpsaniDevRolloutChecks

        return await checks_class.run(
            self.config, matching=matching, halt_on=halt_on
        )

async def _stream_remedy_command(command: str) -> None:
    await servo.utilities.subprocess.stream_subprocess_shell(
        command,
        stdout_callback=lambda msg: servo.logger.debug(f"[stdout] {msg}"),
        stderr_callback=lambda msg: servo.logger.warning(f"[stderr] {msg}"),
    )
