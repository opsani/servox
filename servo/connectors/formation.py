from __future__ import annotations
import os
from servo.connectors.prometheus import PrometheusConnector
from servo.connectors.kubernetes import KubernetesConfiguration, KubernetesConnector
from servo.events import on_event
from typing import Generator, List, Optional, Tuple
import servo
from servo import connector, configuration, checks
from servo.connectors import kubernetes, prometheus
from servo import Check, Filter, Severity

# class Formations(str, enum.Enum):
#     opsani_dev = "opsani_dev"

class DevFormationChecks(checks.BaseChecks):
    @checks.warn("Prometheus sidecar")
    async def check_prometheus_sidecar(self) -> Tuple[bool, str]:
        if not os.environ.get("KUBERNETES_SERVICE_HOST", False):
            return False, "Not running under Kubernetes"

        # Read our own Pod
        pod_name = os.environ.get("POD_NAME", None)
        pod_namespace = os.environ.get("POD_NAMESPACE", None)
        if pod_name and pod_namespace:
            pod = await kubernetes.Pod.read(pod_name, pod_namespace)
            container = pod.get_container("prometheus")
            if container:
                return True, f"Found Prometheus sidecar running {container.obj.image} in Pod {pod_name}"
        else:
            return False, f"No Prometheus sidecar found in Pod {pod_name}"

    # TODO: Dispatch metrics and check the names/queries
    # @checks.multicheck("Prometheus queries")
    # async def check_metrics(self) -> None:
    #     ...

    # TODO: Fetch and look at the Prometheus targets
    @checks.check("Envoy proxies are being scraped")
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

class FormationConfiguration(servo.AbstractBaseConfiguration):
    @classmethod
    def _generate(cls, **kwargs) -> Generator[Tuple[str, servo.AbstractBaseConfiguration], None, None]:
        for method in (cls.generate_kubernetes_config, cls.generate_prometheus_config):
            yield method()
    
    @classmethod
    def generate_kubernetes_config(cls, **kwargs) -> Tuple[str, KubernetesConfiguration]:
        """Generates a configuration for running an Opsani Dev optimization under Kubernetes.

        Returns:
            A tuple of connector name and a Kubernetes connector configuration object.
        """
        return "kubernetes", KubernetesConfiguration(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                kubernetes.DeploymentConfiguration(
                    name="app",
                    replicas=kubernetes.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        kubernetes.ContainerConfiguration(
                            name="opsani/fiber-http:latest",
                            cpu=kubernetes.CPU(
                                min="250m",
                                max="4000m",
                                step="125m"
                            ),
                            memory=kubernetes.Memory(
                                min="256 MiB",
                                max="4.0 GiB",
                                step="128 MiB"
                            )
                        )
                    ]
                )
            ],
            **kwargs
        )
    
    @classmethod
    def generate_prometheus_config(cls, **kwargs) -> Tuple[str, KubernetesConfiguration]:
        """Generates a configuration for running an Opsani Dev optimization that utilizes
        Prometheus and Envoy sidecars to produce and aggregate the necessary metrics.

        Returns:
            A tuple of connector name and a Prometheus connector configuration object.
        """
        return "prometheus", prometheus.PrometheusConfiguration(
            description="A sidecar configuration for aggregating metrics from Envoy sidecar proxies.",
            base_url="http://localhost:9090",            
            metrics=[
                prometheus.PrometheusMetric(
                    "main_instance_count",
                    servo.types.Unit.COUNT,
                    query="sum(envoy_cluster_membership_healthy{opsani_role!=\"tuning\"})",
                ),
                prometheus.PrometheusMetric(
                    "tuning_instance_count",
                    servo.types.Unit.COUNT,
                    query="envoy_cluster_membership_healthy{opsani_role=\"tuning\"}",
                ),

                prometheus.PrometheusMetric(
                    "main_pod_avg_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="avg(rate(envoy_cluster_upstream_rq_total{opsani_role!=\"tuning\"}[3m]))",
                ),
                prometheus.PrometheusMetric(
                    "total_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_total[3m]))",
                ),
                prometheus.PrometheusMetric(
                    "main_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_total{opsani_role!=\"tuning\"}[3m]))",
                ),
                prometheus.PrometheusMetric(
                    "tuning_request_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="rate(envoy_cluster_upstream_rq_total{opsani_role=\"tuning\"}[3m])",
                ),
                
                prometheus.PrometheusMetric(
                    "main_success_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=\"2\"}[3m]))",
                ),
                prometheus.PrometheusMetric(
                    "tuning_success_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="rate(envoy_cluster_upstream_rq_xx{opsani_role=\"tuning\", envoy_response_code_class=\"2\"}[3m])",
                ),
                
                prometheus.PrometheusMetric(
                    "main_error_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[3m]))",
                ),
                prometheus.PrometheusMetric(
                    "tuning_error_rate",
                    servo.types.Unit.REQUESTS_PER_SECOND,
                    query="rate(envoy_cluster_upstream_rq_xx{opsani_role=\"tuning\", envoy_response_code_class=~\"4|5\"}[3m])",
                ),

                prometheus.PrometheusMetric(
                    "main_p90_latency",
                    servo.types.Unit.MILLISECONDS,
                    query="avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role!=\"tuning\"}[3m])))",
                ),
                prometheus.PrometheusMetric(
                    "tuning_p90_latency",
                    servo.types.Unit.MILLISECONDS,
                    query="avg(histogram_quantile(0.9,rate(envoy_cluster_upstream_rq_time_bucket{opsani_role=\"tuning\"}[3m])))",
                ),
            ],
            **kwargs,
        )

@connector.metadata(
    description="Run connectors in a specific formation",
    version="0.0.1",
    homepage="https://github.com/opsani/servox",
    license=connector.License.APACHE2,
    maturity=connector.Maturity.EXPERIMENTAL,
)
class FormationConnector(connector.BaseConnector):
    config: FormationConfiguration

    @servo.on_event()
    async def check(
        self,
        filter_: Optional[Filter],
        halt_on: Optional[Severity] = checks.Severity.critical
    ) -> List[Check]:
        return await DevFormationChecks.run(servo.BaseConfiguration(), filter_=filter_, halt_on=halt_on)

#     # TODO: require kubernetes -- can we depend on other connectors?
#     # TODO: inspect the prometheus targets
#     # TODO: dispatch event to get metrics from Prometheus, check thresholds???