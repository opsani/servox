import asyncio
from time import sleep

import freezegun
import kubetest.client
import pytest

import servo
from servo.runner import ServoRunner

from servo.connectors.kube_metrics import *
from servo.connectors.kube_metrics import _append_data_point, _get_target_resource, _get_target_resource_container, _name_to_metric
from tests.connectors.kubernetes_test import namespace

@pytest.fixture
def kubecontext() -> str:
    return "metrics-server"

@pytest.fixture
def kube_metrics_config() -> KubeMetricsConfiguration:
    return KubeMetricsConfiguration.generate()

@pytest.fixture
def kube_metrics_connector(kube_metrics_config: KubeMetricsConfiguration) -> KubeMetricsConnector:
    return KubeMetricsConnector(config=kube_metrics_config)

async def test_attach(kube_metrics_connector: KubeMetricsConnector, servo_runner: ServoRunner):
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

# TODO group minikube fixture into file scope when xdist supports fixture scoping
@pytest.mark.minikube_profile.with_args("metrics-server")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
async def test_periodic_measure(kubeconfig: str, minikube: str, kube: kubetest.client.TestClient, servo_runner: ServoRunner):
    kube.wait_for_registered()
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(lambda: defaultdict(list))
    connector = KubeMetricsConnector(config=KubeMetricsConfiguration(
        name="fiber-http",
        namespace=kube.namespace,
        container="fiber-http",
        context=minikube,
        kubeconfig=kubeconfig,
    ))

    await connector.attach(servo_=servo_runner.servo)
    deployment = await Deployment.read("fiber-http", kube.namespace)

    async def wait_for_scrape():
        async with kubernetes_asyncio.client.ApiClient() as api:
            cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api)
            while True:
                result = await cust_obj_api.list_namespaced_custom_object(
                    label_selector=deployment.label_selector,
                    namespace=kube.namespace,
                    **METRICS_CUSTOM_OJBECT_CONST_ARGS
                )
                if result.get('items'): # items present and non-empty
                    break
    await asyncio.wait_for(wait_for_scrape(), timeout=60)

    await connector.periodic_measure(
        target_resource=deployment,
        target_metrics=MAIN_METRICS,
        datapoints_dicts=datapoints_dicts,
    )

    for m in MAIN_METRICS:
        assert m in datapoints_dicts

@freezegun.freeze_time("2020-01-21 12:00:01")
def test_append_data_point():
    datapoints_dicts: Dict[str, Dict[str, List[DataPoint]]] = defaultdict(lambda: defaultdict(list))

    _append_data_point(
        datapoints_dicts=datapoints_dicts,
        pod_name="test_pod",
        metric_name="test_metric",
        time=datetime.now(),
        value=1,
    )

    assert datapoints_dicts == {
        'test_metric': {
            'test_pod': [
                DataPoint(
                    metric=Metric(
                        name='test_metric',
                        unit=servo.Unit.float,
                    ),
                    time=datetime(2020, 1, 21, 12, 0, 1),
                    value=1.0,
                ),
            ],
        },
    }

@pytest.mark.minikube_profile.with_args("metrics-server")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
# async def test_periodic_measure(kubeconfig: str, minikube: str, kube: kubetest.client.TestClient, servo_runner: ServoRunner):
async def test_get_target_resource(kubeconfig: str, kubecontext: str, minikube: str, kube: kubetest.client.TestClient):
    kube.wait_for_registered()
    await kubernetes_asyncio.config.load_kube_config(config_file=str(kubeconfig), context=kubecontext)
    assert await _get_target_resource(KubeMetricsConfiguration(
        name="fiber-http",
        namespace=kube.namespace,
        container="fiber-http",
        context=minikube,
        kubeconfig=kubeconfig,
    ))

@pytest.mark.minikube_profile.with_args("metrics-server")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
async def test_get_target_resource_container(kubeconfig: str, kubecontext: str, minikube: str, kube: kubetest.client.TestClient):
    kube.wait_for_registered()
    await kubernetes_asyncio.config.load_kube_config(config_file=str(kubeconfig), context=kubecontext)
    deployment = await Deployment.read("fiber-http", kube.namespace)
    assert _get_target_resource_container(KubeMetricsConfiguration(
        name="fiber-http",
        namespace=kube.namespace,
        container="fiber-http",
        context=minikube,
        kubeconfig=kubeconfig,
    ), target_resource=deployment)

def test_name_to_metric():
    assert _name_to_metric("tuning_cpu_usage") == servo.Metric(name='tuning_cpu_usage', unit=servo.Unit.float)

@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
# @pytest.mark.applymanifests("../manifests/kube_metrics", files=["role.yaml", "role-binding.yaml"])
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
class TestKubeMetricsConnectorIntegration:
    @pytest.fixture
    def kube_metrics_config(self, kube: kubetest.client.TestClient):
        return KubeMetricsConfiguration(
            namespace=kube.namespace,
            name="fiber-http",
            container="fiber-http",
        )

    @pytest.fixture
    def kubecontext(self) -> str:
        return None # override file level fixture for EKS

    async def test_checks(self, kube_metrics_config: KubeMetricsConfiguration) -> None:
        checks = await KubeMetricsChecks.run(kube_metrics_config)
        assert all(c.success for c in checks), debug(checks)

    async def test_measure(self, kube: kubetest.client.TestClient, kube_metrics_connector: KubeMetricsConnector) -> None:
        kube.wait_for_registered()
        deployment = await Deployment.read("fiber-http", kube.namespace)

        async def wait_for_scrape():
            async with kubernetes_asyncio.client.ApiClient() as api:
                cust_obj_api = kubernetes_asyncio.client.CustomObjectsApi(api)
                while True:
                    result = await cust_obj_api.list_namespaced_custom_object(
                        label_selector=deployment.label_selector,
                        namespace=kube.namespace,
                        **METRICS_CUSTOM_OJBECT_CONST_ARGS
                    )
                    if result.get('items'): # items present and non-empty
                        break
        await asyncio.wait_for(wait_for_scrape(), timeout=60)

        kube_metrics_connector.config.metric_collection_frequency = servo.Duration("1s")
        result = await kube_metrics_connector.measure()
        assert len(result) == 9, debug(result)
