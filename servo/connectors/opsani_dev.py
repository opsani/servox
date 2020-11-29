import os
from typing import List, Optional

import httpx

import servo
import servo.connectors.kubernetes
import servo.connectors.prometheus


PROMETHEUS_SIDECAR_BASE_URL = "http://localhost:9090"


class OpsaniDevConfiguration(servo.AbstractBaseConfiguration):
    namespace: str
    deployment: str
    container: str
    service: str
    config_maps: Optional[List[str]]

    @classmethod
    def generate(cls, **kwargs) -> "OpsaniDevConfiguration":
        return cls(
            namespace="default",
            deployment="app-deployment",
            container="main",
            service="app",
        )

    def generate_kubernetes_config(
        self, **kwargs
    ) -> servo.connectors.kubernetes.KubernetesConfiguration:
        """Generate a configuration for running an Opsani Dev optimization under servo.connectors.kubernetes.

        Returns:
            A tuple of connector name and a Kubernetes connector configuration object.
        """
        return servo.connectors.kubernetes.KubernetesConfiguration(
            namespace=self.namespace,
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                servo.connectors.kubernetes.DeploymentConfiguration(
                    name=self.deployment,
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        servo.connectors.kubernetes.ContainerConfiguration(
                            name=self.container,
                            cpu=servo.connectors.kubernetes.CPU(min="250m", max="4000m", step="125m"),
                            memory=servo.connectors.kubernetes.Memory(
                                min="256 MiB", max="4.0 GiB", step="128 MiB"
                            ),
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
            A tuple of connector name and a Prometheus connector configuration object.
        """
        return servo.connectors.prometheus.PrometheusConfiguration(
            description="A sidecar configuration for aggregating metrics from Envoy sidecar proxies.",
            base_url=PROMETHEUS_SIDECAR_BASE_URL,
            metrics=[
                servo.connectors.prometheus.PrometheusMetric(
                    "main_instance_count",
                    servo.types.Unit.COUNT,
                    query='sum(envoy_cluster_membership_healthy{opsani_role!="tuning"})',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_instance_count",
                    servo.types.Unit.COUNT,
                    query='envoy_cluster_membership_healthy{opsani_role="tuning"}',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_pod_avg_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='avg(rate(envoy_cluster_upstream_rq_total{opsani_role!="tuning"}[3m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "total_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_total[3m]))",
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='sum(rate(envoy_cluster_upstream_rq_total{opsani_role!="tuning"}[3m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='rate(envoy_cluster_upstream_rq_total{opsani_role="tuning"}[3m])',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_success_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class="2"}[3m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_success_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class="2"}[3m])',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_error_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!="tuning", envoy_response_code_class=~"4|5"}[3m]))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_error_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query='rate(envoy_cluster_upstream_rq_xx{opsani_role="tuning", envoy_response_code_class=~"4|5"}[3m])',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "main_p90_latency",
                    servo.types.Unit.MILLISECONDS,
                    query='avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role!="tuning"}[3m])))',
                ),
                servo.connectors.prometheus.PrometheusMetric(
                    "tuning_p90_latency",
                    servo.types.Unit.MILLISECONDS,
                    query='avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role="tuning"}[3m])))',
                ),
            ],
            **kwargs,
        )


# TODO: Support a decorator for setting class level tags
class OpsaniDevChecks(servo.BaseChecks):
    config: OpsaniDevConfiguration

    ##
    # Kubernetes essentials

    @servo.checks.require("namespace")
    async def check_kubernetes_namespace(self) -> None:
        await servo.connectors.kubernetes.Namespace.read(self.config.namespace)

    @servo.checks.require("deployment")
    async def check_kubernetes_deployment(self) -> None:
        await servo.connectors.kubernetes.Deployment.read(self.config.deployment, self.config.namespace)

    @servo.checks.require("container")
    async def check_kubernetes_container(self) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            self.config.deployment, self.config.namespace
        )
        container = deployment.find_container(self.config.container)
        assert (
            container
        ), f"failed reading Container '{self.config.container}' in Deployment '{self.config.deployment}'"

    @servo.checks.require("service")
    async def check_kubernetes_service(self) -> None:
        await servo.connectors.kubernetes.Service.read(self.config.service, self.config.namespace)

    @servo.checks.warn("service type")
    async def check_kubernetes_service_type(self) -> None:
        service = await servo.connectors.kubernetes.Service.read(
            self.config.service, self.config.namespace
        )
        if not service.obj.spec.type in ("ClusterIP", "LoadBalancer"):
            raise ValueError(
                f"expected service type of ClusterIP or LoadBalancer but found {service.spec.type}"
            )

    ##
    # Prometheus sidecar

    @servo.checks.require("Prometheus ConfigMap exists")
    async def check_prometheus_config_map(self) -> None:
        config = await servo.connectors.kubernetes.ConfigMap.read(
            "prometheus-config", self.config.namespace
        )
        self.logger.trace(f"read Prometheus ConfigMap: {repr(config)}")
        assert config, "failed: no config map named 'prometheus-config'"

    @servo.checks.check("Prometheus sidecar injected")
    async def check_prometheus_sidecar_exists(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise RuntimeError(f"failed: no servo pod was found")

        if not pod.get_container("prometheus"):
            raise RuntimeError(
                f"failed: no 'prometheus' container found in pod '{pod.name}'"
            )

    @servo.checks.check("Prometheus sidecar is ready")
    async def check_prometheus_sidecar_is_ready(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise RuntimeError(f"failed: no servo pod was found")

        if not await pod.is_ready():
            raise RuntimeError(f"failed: pod '{pod.name}' is not ready")

    @servo.checks.warn("Prometheus sidecar is stable")
    async def check_prometheus_restart_count(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise RuntimeError(f"failed: no servo pod was found")

        container = pod.get_container("prometheus")
        assert container
        restart_count = await container.get_restart_count()
        assert (
            restart_count == 0
        ), f"container 'prometheus' in pod '{pod.name}' has restarted {restart_count} times"

    @servo.checks.require("Prometheus has container port on 9090")
    async def check_prometheus_container_port(self) -> None:
        pod = await self._read_servo_pod()
        if pod is None:
            raise RuntimeError(f"failed: no servo pod was found")

        container = pod.get_container("prometheus")
        assert container

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
            raise RuntimeError(f"failed: no servo pod was found")

        container = pod.get_container("prometheus")
        assert container
        assert (
            len(container.obj.ports) == 1
        ), f"expected 1 container port but found {len(container.obj.ports)}"

        # NOTE: Prometheus sidecar will be up on localhost
        async with httpx.AsyncClient(
            base_url=PROMETHEUS_SIDECAR_BASE_URL
        ) as client:
            response = await client.get("/api/v1/targets")
            response.raise_for_status()
            result = response.json()

        target_count = len(result["data"]["activeTargets"])
        assert target_count > 0
        return f"found {target_count} targets"

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
                {"app.servo.connectors.kubernetes.io/name": "servo"}
            )
            pod_list: servo.connectors.kubernetes.client.V1PodList = (
                await api_client.list_namespaced_pod(
                    namespace=self.config.namespace, label_selector=label_selector
                )
            )

        pods = [servo.connectors.kubernetes.Pod(p) for p in pod_list.items]
        return pods

    # TODO: Trigger basic checks on Prometheus connector

    ##
    # Kubernetes Deployment edits

    @servo.checks.require("validate service")
    async def check_deployment_annotations(self) -> None:
        ...

    @servo.checks.require("validate service")
    async def check_deployment_labels(self) -> None:
        ...

    @servo.checks.require("validate service")
    async def check_deployment_envoy_sidecars(self) -> None:
        ...

    ##
    # Connecting the dots

    @servo.require("validate service")
    async def check_prometheus_scraping_envoys(self) -> None:
        ...

    @servo.require("validate service")
    async def check_prometheus_queries_make_sense(self) -> None:
        ...

    # TODO: Dispatch metrics and check the names/queries
    # @checks.multicheck("Prometheus queries")
    # async def check_metrics(self) -> None:
    #     ...
    # TODO: Fetch and look at the Prometheus targets
    @servo.check("Envoy proxies are being scraped")
    async def check_envoy_sidecar_metrics(self) -> str:
        # TODO: Ask Prometheus? Get its config or do I ask it to do something
        ...

    # TODO: Find the Envoy proxies and make sure they are all reporting metrics
    # TODO: Cycle the canary up and down and make sure that it gets traffic
    # # What we may want to do is run an adjust and then re-run all the checks.
    # # Actually we can just bring up the canary and then re-check...
    # @check("New canary Pods receive traffic")
    # async def check_pod_load_balancing(self) -> str:
    #     ...


@servo.metadata(
    description="Run connectors in a specific formation",
    version="0.0.1",
    homepage="https://github.com/opsani/servox",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.EXPERIMENTAL,
)
class OpsaniDevConnector(servo.BaseConnector):
    """Opsani Dev is a turnkey solution for optimizing a single service."""

    config: OpsaniDevConfiguration

    @servo.on_event()
    async def startup(self) -> None:
        servo_ = servo.Servo.current()
        await servo_.add_connector(
            "kubernetes",
            servo.connectors.kubernetes.KubernetesConnector(
                optimizer=self.optimizer,
                config=self.config.generate_kubernetes_config(),
            ),
        )
        await servo_.add_connector(
            "prometheus",
            servo.connectors.prometheus.PrometheusConnector(
                optimizer=self.optimizer,
                config=self.config.generate_prometheus_config(),
            ),
        )

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter],
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.CRITICAL,
    ) -> List[servo.Check]:
        return await OpsaniDevChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )
