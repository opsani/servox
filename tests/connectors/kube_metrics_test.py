import asyncio
import loguru
import pathlib

import freezegun
import kubetest.client
import pytest

from kubernetes_asyncio.client import V1Deployment

import servo
from servo.runner import ServoRunner

from servo.connectors.kubernetes_helpers import dict_to_selector, DeploymentHelper
from servo.connectors.kube_metrics import *
from servo.connectors.kube_metrics import (
    _append_data_point,
    _get_target_resource,
    _get_target_resource_container,
    _name_to_metric,
)


@pytest.fixture
def kube_metrics_config() -> KubeMetricsConfiguration:
    return KubeMetricsConfiguration.generate()


@pytest.fixture
def kube_metrics_connector(
    kube_metrics_config: KubeMetricsConfiguration,
) -> KubeMetricsConnector:
    return KubeMetricsConnector(config=kube_metrics_config)


async def test_attach(
    kube_metrics_connector: KubeMetricsConnector,
    servo_runner: ServoRunner,
    kubeconfig: pathlib.Path,
    minikube: str,
):
    kube_metrics_connector.config.kubeconfig = kubeconfig
    await servo_runner.servo.add_connector("kube_metrics", kube_metrics_connector)


def test_metrics(kube_metrics_connector: KubeMetricsConnector):
    kube_metrics_connector.metrics()


async def test_describe(kube_metrics_connector: KubeMetricsConnector):
    await kube_metrics_connector.describe()


MAIN_METRICS = [
    SupportedKubeMetrics.MAIN_CPU_USAGE,
    SupportedKubeMetrics.MAIN_CPU_REQUEST,
    SupportedKubeMetrics.MAIN_CPU_LIMIT,
    SupportedKubeMetrics.MAIN_CPU_SATURATION,
    SupportedKubeMetrics.MAIN_MEM_USAGE,
    SupportedKubeMetrics.MAIN_MEM_REQUEST,
    SupportedKubeMetrics.MAIN_MEM_LIMIT,
    SupportedKubeMetrics.MAIN_MEM_SATURATION,
    SupportedKubeMetrics.MAIN_POD_RESTART_COUNT,
]


async def _try_wait_for_scrape(namespace: str, deployment: V1Deployment) -> None:
    try:
        await asyncio.wait_for(
            _wait_for_scrape(namespace=namespace, deployment=deployment), timeout=60
        )
    except asyncio.TimeoutError as te:
        pytest.xfail("Metrics server scrape failed")


async def _wait_for_scrape(namespace: str, deployment: V1Deployment):
    async with kubernetes_asyncio.client.ApiClient() as api:
        cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api)
        while True:
            await asyncio.sleep(1)

            result = await cust_obj_api.list_namespaced_custom_object(
                label_selector=dict_to_selector(deployment.spec.selector.match_labels),
                namespace=namespace,
                **METRICS_CUSTOM_OJBECT_CONST_ARGS,
            )
            loguru.logger.info(f"Kubernetes metrics server result: {result}")
            if result.get("items"):  # metrics items present and non-empty
                if any(  # TODO same container name always used, abstract into param if needed
                    (
                        any(c["name"] == "fiber-http" for c in i["containers"])
                        and any(c["name"] == "opsani-envoy" for c in i["containers"])
                    )
                    for i in result["items"]
                ):
                    break

            loguru.logger.info("Coninuing wait for scrape")


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
async def test_periodic_measure(
    kube: kubetest.client.TestClient,
    servo_runner: ServoRunner,
):
    kube.wait_for_registered()
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    connector = KubeMetricsConnector(
        config=KubeMetricsConfiguration(
            name="fiber-http",
            namespace=kube.namespace,
            container="fiber-http",
        )
    )

    await connector.attach(servo_=servo_runner.servo)
    deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
    await _try_wait_for_scrape(namespace=kube.namespace, deployment=deployment)

    await connector.periodic_measure(
        target_metrics=MAIN_METRICS,
        datapoints_dicts=datapoints_dicts,
    )

    for m in MAIN_METRICS:
        assert m in datapoints_dicts


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests(
    "../manifests", files=["fiber-http-opsani-dev_no_resource_limits.yaml"]
)
async def test_periodic_measure_no_limits(
    kube: kubetest.client.TestClient,
    servo_runner: ServoRunner,
):
    kube.wait_for_registered()
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    connector = KubeMetricsConnector(
        config=KubeMetricsConfiguration(
            name="fiber-http",
            namespace=kube.namespace,
            container="fiber-http",
        )
    )

    await connector.attach(servo_=servo_runner.servo)
    deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
    await _try_wait_for_scrape(namespace=kube.namespace, deployment=deployment)

    await connector.periodic_measure(
        target_metrics=MAIN_METRICS,
        datapoints_dicts=datapoints_dicts,
    )

    for m in MAIN_METRICS:
        if "limit" in m:
            assert m not in datapoints_dicts


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests(
    "../manifests", files=["fiber-http-opsani-dev_no_resource_requests.yaml"]
)
async def test_periodic_measure_no_requests(
    kube: kubetest.client.TestClient,
    servo_runner: ServoRunner,
):
    kube.wait_for_registered()
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(
        lambda: defaultdict(list)
    )
    connector = KubeMetricsConnector(
        config=KubeMetricsConfiguration(
            name="fiber-http",
            namespace=kube.namespace,
            container="fiber-http",
        )
    )

    await connector.attach(servo_=servo_runner.servo)
    deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
    await _try_wait_for_scrape(namespace=kube.namespace, deployment=deployment)

    await connector.periodic_measure(
        target_metrics=MAIN_METRICS,
        datapoints_dicts=datapoints_dicts,
    )

    main_cpu_requests_readings = list(
        datapoints_dicts[SupportedKubeMetrics.MAIN_CPU_REQUEST].values()
    )
    main_cpu_limits_readings = list(
        datapoints_dicts[SupportedKubeMetrics.MAIN_CPU_LIMIT].values()
    )

    for requests_datapoints, limits_datapoints in zip(
        main_cpu_requests_readings, main_cpu_limits_readings
    ):
        assert requests_datapoints[0].value == limits_datapoints[0].value


@freezegun.freeze_time("2020-01-21 12:00:01")
def test_append_data_point():
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(
        lambda: defaultdict(list)
    )

    _append_data_point(
        datapoints_dicts=datapoints_dicts,
        pod_name="test_pod",
        metric_name="test_metric",
        time=datetime.now(),
        value=1,
    )

    assert datapoints_dicts == {
        "test_metric": {
            "test_pod": [
                DataPoint(
                    metric=Metric(
                        name="test_metric",
                        unit=servo.Unit.float,
                    ),
                    time=datetime(2020, 1, 21, 12, 0, 1),
                    value=1.0,
                ),
            ],
        },
    }


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
# async def test_periodic_measure(kubeconfig: str, minikube: str, kube: kubetest.client.TestClient, servo_runner: ServoRunner):
async def test_get_target_resource(kube: kubetest.client.TestClient):
    kube.wait_for_registered()
    await kubernetes_asyncio.config.load_kube_config()
    assert await _get_target_resource(
        KubeMetricsConfiguration(
            name="fiber-http",
            namespace=kube.namespace,
            container="fiber-http",
        )
    )


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
async def test_get_target_resource_container(kube: kubetest.client.TestClient):
    kube.wait_for_registered()
    await kubernetes_asyncio.config.load_kube_config()
    deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
    assert _get_target_resource_container(
        KubeMetricsConfiguration(
            name="fiber-http",
            namespace=kube.namespace,
            container="fiber-http",
        ),
        target_resource=deployment,
    )


def test_name_to_metric():
    assert _name_to_metric("tuning_cpu_usage") == servo.Metric(
        name="tuning_cpu_usage", unit=servo.Unit.float
    )


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
class TestKubeMetricsConnectorIntegration:
    @pytest.fixture
    def kube_metrics_config(self, kube: kubetest.client.TestClient):
        return KubeMetricsConfiguration(
            namespace=kube.namespace,
            name="fiber-http",
            container="fiber-http",
        )

    async def test_checks(self, kube_metrics_config: KubeMetricsConfiguration) -> None:
        checks = await KubeMetricsChecks.run(kube_metrics_config)
        assert all(c.success for c in checks), debug(checks)

    async def test_measure(
        self,
        kube: kubetest.client.TestClient,
        kube_metrics_connector: KubeMetricsConnector,
    ) -> None:
        kube.wait_for_registered()
        deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
        await _try_wait_for_scrape(namespace=kube.namespace, deployment=deployment)

        kube_metrics_connector.config.metric_collection_frequency = servo.Duration("1s")
        result = await kube_metrics_connector.measure()
        assert len(result) == 9, debug(result)
