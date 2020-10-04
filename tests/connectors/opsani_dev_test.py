import pytest
import os
import respx
import httpx
import servo
from servo.connectors import opsani_dev

class TestConfiguration:
    ...

class TestConnector:
    ...


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# @pytest.mark.integration
# class TestChecks:
#     ...

@pytest.fixture
def config() -> opsani_dev.OpsaniDevConfiguration:
    return opsani_dev.OpsaniDevConfiguration(
        namespace="default",
        deployment="fiber-http",
        container="fiber-http",
        service="fiber-http"
    )

@pytest.fixture
def checks(config: opsani_dev.OpsaniDevConfiguration) -> opsani_dev.OpsaniDevChecks:
    return opsani_dev.OpsaniDevChecks(config=config)

# TODO: Add doesn't exist, can't read...
@pytest.mark.applymanifests('opsani_dev', files=[
    'deployment.yaml',
    'service.yaml',
    'prometheus.yaml',
    # 'servo.yaml'
])
class TestChecksOriginalState:
    @pytest.fixture(autouse=True)
    async def load_manifests(self, kube, kubeconfig, checks: opsani_dev.OpsaniDevChecks) -> None:
        kube.wait_for_registered(timeout=30)
        checks.config.namespace = kube.namespace

        # Fake out the servo metadata in the environment
        # TODO: find the annotated servo pod
        # TODO: get the deployment
        # app.kubernetes.io/name: servo
        # pods = kube.get_pods(labels={ "app.kubernetes.io/name": "servo"})
        # debug(pods)
        # os.environ['POD_NAME'] = ""
        os.environ['POD_NAMESPACE'] = kube.namespace

    @pytest.mark.parametrize(
        "resource",
        [
            "namespace",
            "deployment",
            "container",
            "service"
        ]        
    )
    async def test_resource_exists(self, resource: str, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_kubernetes_{resource}")
        assert result.success
    
    async def test_prometheus_configmap_exists(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_prometheus_config_map")
        assert result.success
    
    async def test_prometheus_sidecar_exists(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_prometheus_sidecar_exists")
        assert result.success
    
    async def test_prometheus_sidecar_is_ready(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_prometheus_sidecar_is_ready")
        assert result.success
    
    async def test_check_prometheus_restart_count(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_prometheus_restart_count")
        assert result.success

    async def test_check_prometheus_container_port(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        result = await checks.run_one(id=f"check_prometheus_container_port")
        assert result.success

    
    @pytest.fixture
    def go_memstats_gc_sys_bytes(self) -> dict:
        return {'status': 'success', 'data': {'resultType': 'matrix', 'result': [{'metric': {'__name__': 'go_memstats_gc_sys_bytes', 'instance': 'localhost:9090', 'job': 'prometheus'}, 'values': [[1595142421.024, '3594504'], [1595142481.024, '3594504']]}]}}

    @pytest.fixture
    def mocked_api(self, go_memstats_gc_sys_bytes):
        with respx.mock(base_url="http://localhost:9090", assert_all_called=False) as respx_mock:
            respx_mock.get("/api/v1/targets", alias="targets", content=[])

            # re.compile(r"/api/v1/query_range/\w+")
            respx_mock.get(re.compile(r"/api/v1/query_range.+"), alias="query", content=go_memstats_gc_sys_bytes)
            # respx_mock.get("/api/v1/query_range", alias="query", content=go_memstats_gc_sys_bytes)
            yield respx_mock
    
    # check_kubernetes_service_type
    async def test_check_prometheus_is_accessible(self, kube, checks: opsani_dev.OpsaniDevChecks) -> None:
        with respx.mock(base_url="http://localhost:9090") as respx_mock:
            request = respx_mock.get("/api/v1/targets", status_code=503)
            check = await checks.run_one(id=f"check_prometheus_is_accessible")
            debug(check)
            assert request.called
            assert check
            assert check.name == 'Connect to "http://localhost:9090"'
            assert check.id == 'check_prometheus_is_accessible'
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