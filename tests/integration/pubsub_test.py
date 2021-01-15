import asyncio
import pytest
import servo
import servo.pubsub
import servo.connectors.prometheus
import servo.connectors.vegeta

from typing import Callable, AsyncIterator

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

class TestVegeta:
    @pytest.fixture
    def connector(self) -> servo.connectors.vegeta.VegetaConnector:
        config = servo.connectors.vegeta.VegetaConfiguration(
            rate="50/1s",
            target="GET http://localhost:8080",
            reporting_interval="500ms"
        )
        return servo.connectors.vegeta.VegetaConnector(config=config)

    async def test_subscribe_via_exchange(self, connector) -> None:
        reports = []

        async def _callback(message, channel) -> None:
            debug("Vegeta Reported: ", message.json())
            reports.append(message.json())

        subscriber = connector.pubsub_exchange.create_subscriber("loadgen.vegeta", callback=_callback)
        connector.pubsub_exchange.start()
        measurement = await asyncio.wait_for(
            connector.measure(control=servo.Control(duration="5s")),
            timeout=7 # NOTE: Always make timeout exceed control duration
        )
        assert len(reports) > 5


    async def test_subscribe_via_connector(self, connector) -> None:
        ...

    async def test_subscribe_via_servo(self, connector) -> None:
        ...

# @pytest.mark.usefixtures("kubernetes_asyncio_config")
# @pytest.mark.applymanifests(
#     "../manifests",
#     files=[
#         "prometheus.yaml",
#     ],
# )
# @pytest.mark.clusterrolebinding('cluster-admin')
# class TestPrometheus:
#     def optimizer(self) -> servo.Optimizer:
#         return servo.Optimizer(
#             id="dev.opsani.com/blake-ignite",
#             token="bfcf94a6e302222eed3c73a5594badcfd53fef4b6d6a703ed32604",
#         )

#     @pytest.fixture(autouse=True)
#     def _wait_for_cluster(self, kube) -> None:
#         kube.wait_for_registered(timeout=60)

#     async def test_no_traffic(
#         self,
#         optimizer: servo.Optimizer,
#         kube,
#         kube_port_forward: Callable[[str, int], AsyncIterator[str]],
#         absent,
#         readings
#     ) -> None:
#         # NOTE: What we are going to do here is deploy Prometheus and fiber-http with no traffic source,
#         # port forward so we can talk to them, and then spark up the connector.
#         # The measurement duration will expire and report flatlined metrics.
#         servo.logging.set_level("DEBUG")
#         kube.wait_for_registered(timeout=30)

#         async with kube_port_forward("deploy/prometheus", 9090) as prometheus_url:
#             async with kube_port_forward("service/fiber-http", 80) as fiber_url:
#                 config = servo.connectors.prometheus.PrometheusConfiguration.generate(
#                     base_url=prometheus_url,
#                     metrics=[
#                         servo.connectors.prometheus.PrometheusMetric(
#                             "throughput",
#                             servo.Unit.requests_per_second,
#                             query="sum(rate(envoy_cluster_upstream_rq_total[5s]))",
#                             absent=absent,
#                             step="5s",
#                         ),
#                         servo.connectors.prometheus.PrometheusMetric(
#                             "error_rate",
#                             servo.Unit.percentage,
#                             query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[5s]))",
#                             absent=absent,
#                             step="5s",
#                         ),
#                     ],
#                 )
#                 connector = PrometheusConnector(config=config, optimizer=optimizer)
#                 measurement = await asyncio.wait_for(
#                     connector.measure(control=servo.Control(duration="15s")),
#                     timeout=25 # NOTE: Always make timeout exceed control duration
#                 )
#                 assert measurement is not None
#                 assert list(map(lambda r: (r.metric.name, str(r.duration), r.max.value), measurement.readings)) == readings
