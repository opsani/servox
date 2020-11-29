import os
import re

import httpx
import pytest
import respx

import servo
import servo.connectors.kubernetes
import servo.connectors.prometheus
import servo.connectors.opsani_dev


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
def config() -> servo.connectors.opsani_dev.OpsaniDevConfiguration:
    return servo.connectors.opsani_dev.OpsaniDevConfiguration(
        namespace="default",
        deployment="fiber-http",
        container="fiber-http",
        service="fiber-http",
    )


@pytest.fixture
def checks(config: servo.connectors.opsani_dev.OpsaniDevConfiguration) -> servo.connectors.opsani_dev.OpsaniDevChecks:
    return servo.connectors.opsani_dev.OpsaniDevChecks(config=config)


@pytest.mark.applymanifests(
    "opsani_dev",
    files=[
        "deployment.yaml",
        "service.yaml",
        "prometheus.yaml",
    ],
)
class TestChecksOriginalState:
    @pytest.fixture(autouse=True)
    async def load_manifests(
        self, kube, kubeconfig, kubernetes_asyncio_config, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        kube.wait_for_registered(timeout=30)
        checks.config.namespace = kube.namespace

        # Fake out the servo metadata in the environment
        # These env vars are set by our manifests
        pods = kube.get_pods(labels={ "app.kubernetes.io/name": "servo"})
        assert pods, "servo is not deployed"
        os.environ['POD_NAME'] = list(pods.keys())[0]
        os.environ["POD_NAMESPACE"] = kube.namespace

    @pytest.mark.parametrize(
        "resource", ["namespace", "deployment", "container", "service"]
    )
    async def test_resource_exists(
        self, resource: str, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_kubernetes_{resource}")
        assert result.success

    async def test_prometheus_configmap_exists(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_config_map")
        assert result.success

    async def test_prometheus_sidecar_exists(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_sidecar_exists")
        assert result.success

    async def test_prometheus_sidecar_is_ready(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_sidecar_is_ready")
        assert result.success

    async def test_check_prometheus_restart_count(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_restart_count")
        assert result.success

    async def test_check_prometheus_container_port(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_container_port")
        assert result.success

    @pytest.fixture
    def go_memstats_gc_sys_bytes(self) -> dict:
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "__name__": "go_memstats_gc_sys_bytes",
                            "instance": "localhost:9090",
                            "job": "prometheus",
                        },
                        "values": [
                            [1595142421.024, "3594504"],
                            [1595142481.024, "3594504"],
                        ],
                    }
                ],
            },
        }

    @pytest.fixture
    def mocked_api(self, go_memstats_gc_sys_bytes):
        with respx.mock(
            base_url=servo.connectors.opsani_dev.PROMETHEUS_SIDECAR_BASE_URL,
            assert_all_called=False
        ) as respx_mock:
            respx_mock.get(
                "/api/v1/targets",
                name="targets"
            ).mock(return_value=httpx.Response(200, json=[]))

            respx_mock.get(
                re.compile(r"/api/v1/query_range.+"),
                name="query",
            ).mock(return_value=httpx.Response(200, json=go_memstats_gc_sys_bytes))
            yield respx_mock

    # check_kubernetes_service_type
    async def test_check_prometheus_is_accessible(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        with respx.mock(base_url=servo.connectors.opsani_dev.PROMETHEUS_SIDECAR_BASE_URL) as respx_mock:
            request = respx_mock.get("/api/v1/targets").mock(return_value=httpx.Response(status_code=503))
            check = await checks.run_one(id=f"check_prometheus_is_accessible")
            assert request.called
            assert check
            assert check.name == 'Prometheus is accessible'
            assert check.id == "check_prometheus_is_accessible"
            assert check.critical
            assert not check.success
            assert check.message is not None
            assert isinstance(check.exception, httpx.HTTPStatusError)


class TestChecksServiceUpdated:
    ...


class TestChecksSidecarsInjected:
    ...


# Errors:
# Permissions, (Namespace, Deployment, Container, Service -> Doesnt Exist, Cant Read), ports don't match

# Warnings:
# 9980 port conflict
