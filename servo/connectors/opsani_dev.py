import json
import operator
import os
from typing import List, Optional, Union

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
        verbs=["get", "list"],
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
    deployment: str
    container: str
    service: str
    port: Optional[Union[pydantic.StrictInt, str]] = None
    cpu: CPU
    memory: Memory
    prometheus_base_url: str = PROMETHEUS_SIDECAR_BASE_URL

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
        return servo.connectors.kubernetes.KubernetesConfiguration(
            namespace=self.namespace,
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                servo.connectors.kubernetes.DeploymentConfiguration(
                    name=self.deployment,
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
            ],
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
                    "main_pod_avg_request_rate",
                    servo.types.Unit.requests_per_second,
                    query='avg(rate(envoy_cluster_upstream_rq_total{opsani_role!="tuning"}[1m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "total_request_rate",
                    servo.types.Unit.requests_per_second,
                    query="sum(rate(envoy_cluster_upstream_rq_total[1m]))",
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_request_rate",
                    servo.types.Unit.requests_per_second,
                    query='sum(rate(envoy_cluster_upstream_rq_total{opsani_role!="tuning"}[1m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_request_rate",
                    servo.types.Unit.requests_per_second,
                    query='rate(envoy_cluster_upstream_rq_total{opsani_role="tuning"}[1m])',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_success_rate",
                    servo.types.Unit.requests_per_second,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"2|3"}[1m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_success_rate",
                    servo.types.Unit.requests_per_second,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"2|3"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_error_rate",
                    servo.types.Unit.requests_per_second,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"4|5"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_error_rate",
                    servo.types.Unit.requests_per_second,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"4|5"}[1m]))',
                    absent=servo.connectors.prometheus.AbsentMetricPolicy.zero
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_p99_latency",
                    servo.types.Unit.milliseconds,
                    query='avg(histogram_quantile(0.99,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role!="tuning"}[1m])))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_p99_latency",
                    servo.types.Unit.milliseconds,
                    query='avg(histogram_quantile(0.99,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role="tuning"}[1m])))',
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


class OpsaniDevChecks(servo.BaseChecks):
    config: OpsaniDevConfiguration

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

    @servo.checks.require('Namespace "{self.config.namespace}" is readable')
    async def check_kubernetes_namespace(self) -> None:
        await servo.connectors.kubernetes.Namespace.read(self.config.namespace)

    @servo.checks.require('Deployment "{self.config.deployment}" is readable')
    async def check_kubernetes_deployment(self) -> None:
        await servo.connectors.kubernetes.Deployment.read(self.config.deployment, self.config.namespace)

    @servo.checks.require('Container "{self.config.container}" is readable')
    async def check_kubernetes_container(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment, self.config.namespace
        )
        container = deployment.find_container(self.config.container)
        assert (
            container
        ), f"failed reading Container '{self.config.container}' in Deployment '{self.config.deployment}'"

    @servo.require('Container "{self.config.container}" has resource requirements')
    async def check_resource_requirements(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(self.config.deployment, self.config.namespace)
        container = deployment.find_container(self.config.container)
        assert container
        assert container.resources, "missing container resources"

        # Apply any defaults/overrides for requests/limits from config
        servo.connectors.kubernetes.set_container_resource_defaults_from_config(container, self.config)

        # TODO: How do we handle when you just don't have any requests?
        # assert container.resources.requests, "missing requests for container resources"
        # assert container.resources.requests.get("cpu"), "missing request for resource 'cpu'"
        # assert container.resources.requests.get("memory"), "missing request for resource 'memory'"

        # assert container.resources.limits, "missing limits for container resources"
        # assert container.resources.limits.get("cpu"), "missing limit for resource 'cpu'"
        # assert container.resources.limits.get("memory"), "missing limit for resource 'memory'"

    @servo.checks.require("Target container resources fall within optimization range")
    async def check_target_container_resources_within_limits(self) -> None:
        # Load the Deployment
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment,
            self.config.namespace
        )
        assert deployment, f"failed to read deployment '{self.config.deployment}' in namespace '{self.config.namespace}'"

        # Find the target Container
        target_container = next(filter(lambda c: c.name == self.config.container, deployment.containers), None)
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

    @servo.require('Deployment "{self.config.deployment}" is ready')
    async def check_deployment(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(self.config.deployment, self.config.namespace)
        if not await deployment.is_ready():
            raise RuntimeError(f'Deployment "{deployment.name}" is not ready')

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

    @servo.checks.check('Service routes traffic to Deployment Pods')
    async def check_service_routes_traffic_to_deployment(self) -> None:
        service = await servo.connectors.kubernetes.Service.read(
            self.config.service, self.config.namespace
        )
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment, self.config.namespace
        )

        # NOTE: The Service labels should be a subset of the Deployment labels
        deployment_labels = deployment.obj.spec.selector.match_labels
        delta = dict(set(service.selector.items()) - set(deployment_labels.items()))
        if delta:
            desc = ' '.join(map('='.join, delta.items()))
            raise RuntimeError(f"Service selector does not match Deployment labels. Missing labels: {desc}")

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
    # Kubernetes Deployment edits

    @servo.checks.require("Deployment PodSpec has expected annotations")
    async def check_deployment_annotations(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment,
            self.config.namespace
        )
        assert deployment, f"failed to read deployment '{self.config.deployment}' in namespace '{self.config.namespace}'"

        # Add optimizer annotation to the static Prometheus values
        required_annotations = PROMETHEUS_ANNOTATION_DEFAULTS.copy()
        required_annotations['servo.opsani.com/optimizer'] = self.config.optimizer.id

        # NOTE: Only check for annotation keys
        annotations = deployment.pod_template_spec.metadata.annotations or dict()
        actual_annotations = set(annotations.keys())
        delta = set(required_annotations.keys()).difference(actual_annotations)
        if delta:
            annotations = dict(map(lambda k: (k, required_annotations[k]), delta))
            patch = {"spec": {"template": {"metadata": {"annotations": annotations}}}}
            patch_json = json.dumps(patch, indent=None)
            command = f"kubectl --namespace {self.config.namespace} patch deployment {self.config.deployment} -p '{patch_json}'"
            desc = ', '.join(sorted(delta))
            raise servo.checks.CheckError(
                f"deployment '{deployment.name}' is missing annotations: {desc}",
                hint=f"Patch annotations via: `{command}`",
                remedy=lambda: _stream_remedy_command(command)
            )

    @servo.checks.require("Deployment PodSpec has expected labels")
    async def check_deployment_labels(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment,
            self.config.namespace
        )
        assert deployment, f"failed to read deployment '{self.config.deployment}' in namespace '{self.config.namespace}'"

        labels = deployment.pod_template_spec.metadata.labels
        assert labels, f"deployment '{deployment.name}' does not have any labels"

        # Add optimizer label to the static values
        required_labels = ENVOY_SIDECAR_LABELS.copy()
        required_labels['servo.opsani.com/optimizer'] = servo.connectors.kubernetes.dns_labelize(self.config.optimizer.id)

        # NOTE: Check for exact labels as this isn't configurable
        delta = dict(set(required_labels.items()) - set(labels.items()))
        if delta:
            desc = ', '.join(sorted(map('='.join, delta.items())))
            patch = {"spec": {"template": {"metadata": {"labels": delta}}}}
            patch_json = json.dumps(patch, indent=None)
            command = f"kubectl --namespace {self.config.namespace} patch deployment {self.config.deployment} -p '{patch_json}'"
            raise servo.checks.CheckError(
                f"deployment '{deployment.name}' is missing labels: {desc}",
                hint=f"Patch labels via: `{command}`",
                remedy=lambda: _stream_remedy_command(command)
            )

    @servo.checks.require("Deployment has Envoy sidecar container")
    async def check_deployment_envoy_sidecars(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment,
            self.config.namespace
        )
        assert deployment, f"failed to read deployment '{self.config.deployment}' in namespace '{self.config.namespace}'"

        # Search the containers list for the sidecar
        for container in deployment.containers:
            if container.name == "opsani-envoy":
                return

        command = f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- servo --token-file /servo/opsani.token inject-sidecar -n {self.config.namespace} -s {self.config.service} deployment/{self.config.deployment}"
        raise servo.checks.CheckError(
            f"deployment '{deployment.name}' pod template spec does not include envoy sidecar container ('opsani-envoy')",
            hint=f"Inject Envoy sidecar container via: `{command}`",
            remedy=lambda: _stream_remedy_command(command)
        )

    @servo.checks.require("Pods have Envoy sidecar containers")
    async def check_pod_envoy_sidecars(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment,
            self.config.namespace
        )
        assert deployment, f"failed to read deployment '{self.config.deployment}' in namespace '{self.config.namespace}'"

        pods_without_sidecars = []
        for pod in await deployment.get_pods():
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
        timestamp, value = result.value
        if value in {None, 0.0}:
            command = f"kubectl exec -n {self.config.namespace} -c servo {self._servo_resource_target} -- sh -c \"kubectl port-forward --namespace={self.config.namespace} deploy/{self.config.deployment} 9980 & echo 'GET http://localhost:9980/' | vegeta attack -duration 10s | vegeta report -every 3s\""
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
        deployment_config = kubernetes_config.deployments[0]
        optimization = await servo.connectors.kubernetes.CanaryOptimization.create(
            deployment_config, timeout=kubernetes_config.timeout
        )

        # Ensure the canary is available
        try:
            await optimization.create_or_recreate_tuning_pod()

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
        return await OpsaniDevChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

async def _stream_remedy_command(command: str) -> None:
    await servo.utilities.subprocess.stream_subprocess_shell(
        command,
        stdout_callback=lambda msg: servo.logger.debug(f"[stdout] {msg}"),
        stderr_callback=lambda msg: servo.logger.warning(f"[stderr] {msg}"),
    )
