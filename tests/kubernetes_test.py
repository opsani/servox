import asyncio
import datetime
import hashlib

import kubernetes_asyncio
import kubernetes_asyncio.client
import kubetest.client
import pytest

import servo
import servo.connectors.kubernetes
import tests.helpers

import pydantic
import re
from servo.types import _suggest_step_aligned_values, _is_step_aligned


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.usefixtures("kubernetes_asyncio_config")
]

@pytest.mark.applymanifests("manifests", files=["nginx.yaml"])
def test_nginx(kube: kubetest.client.TestClient) -> None:
    # wait for the manifests loaded by the 'applymanifests' marker
    # to be ready on the cluster
    kube.wait_for_registered()

    deployments = kube.get_deployments()
    nginx_deploy = deployments.get("nginx-deployment")
    assert nginx_deploy is not None

    pods = nginx_deploy.get_pods()
    assert len(pods) == 1, "nginx should deploy with one replica"

    for pod in pods:
        containers = pod.get_containers()
        assert len(containers) == 1, "nginx pod should have one container"

        resp = pod.http_proxy_get("/")
        assert "<h1>Welcome to nginx!</h1>" in resp.data


@pytest.mark.applymanifests("manifests", files=["fiber-http-opsani-dev.yaml"])
def test_fiber_http_and_envoy(kube: kubetest.client.TestClient) -> None:
    kube.wait_for_registered()

    deployments = kube.get_deployments()
    web_deploy = deployments.get("fiber-http")
    assert web_deploy is not None

    pods = web_deploy.get_pods()
    assert len(pods) == 1, "fiber-http should deploy with one replica"

    pod = pods[0]
    pod.wait_until_ready(timeout=30)

    # Check containers
    containers = pod.get_containers()
    assert len(containers) == 2, "should have fiber-http and an envoy sidecar"
    assert containers[0].obj.name == "fiber-http"
    assert containers[1].obj.name == "opsani-envoy"

    # Check services
    response = pod.http_proxy_get("/")
    assert "move along, nothing to see here" in response.data

    # TODO: Ugly hack to control port number
    pod.name = pod.name + ":9901"
    response = pod.http_proxy_get("/stats/prometheus")
    assert "envoy_http_downstream_cx_length_ms_count" in response.data


@pytest.mark.applymanifests("manifests", files=["prometheus.yaml"])
@pytest.mark.xfail(reason="kubetest doesn't support the ClusterRole yet")
def test_prometheus(kube: kubetest.client.TestClient) -> None:
    kube.wait_for_registered()

    deployments = kube.get_deployments()
    prom_deploy = deployments.get("prometheus-core")
    assert prom_deploy is not None

    pods = prom_deploy.get_pods()
    assert len(pods) == 1, "prom_deploy should deploy with one replica"

    # Check that Prometheus is there by referencing string in the HTML body
    pod = pods[0]
    pod.name = pod.name + ":9090"
    response = pod.http_proxy_get("/")
    assert "Prometheus Time Series Collection and Processing Server" in response.data

def test_deploy_servo_fiberhttp_vegeta_measure() -> None:
    pass
    # Make servo load test fiber-http, report the outcome in JSON


def test_deploy_servo_fiberhttp_vegeta_adjust() -> None:
    pass
    # Make servo adjust fiber-http memory, report in JSON


# TODO: Tests to write...
# 1. Servo creates canary on start
# canary gets deleted on stop
# failed adjust (can't schedule)
# integration tests: ad-hoc adjust, ad-hoc measure, checks (generate config files in tmp)
# use ktunnel to bridge and return errors, garbage data
# k8s sizing tool

# Integration test k8s describe, adjust


def test_generate_outputs_human_readable_config() -> None:
    ...


def test_supports_nil_container_name() -> None:
    ...

@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
class TestSidecar:
    async def test_inject_sidecar_by_port_number(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment.containers) == 1
        await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', port=8181)

        deployment_ = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_port_number_string(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment.containers) == 1
        await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', port='8181')

        deployment_ = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_port_conflict(self, kube):
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        with pytest.raises(ValueError, match='Deployment already has a container port 8480'):
            await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', port=8481, service_port=8480)

    async def test_inject_sidecar_by_service(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)

        assert len(deployment.containers) == 1
        await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', service='fiber-http')

        deployment_ = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_service_and_port_number(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)

        assert len(deployment.containers) == 1
        await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', service='fiber-http', port=8480)

        deployment_ = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_service_and_port_name(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        # NOTE: This can generate a 409 Conflict failure under CI
        for _ in range(3):
            try:
                # change the container port so we don't conflict
                deployment.obj.spec.template.spec.containers[0].ports[0].container_port = 9999
                await deployment.replace()
                break

            except kubernetes_asyncio.client.exceptions.ApiException as e:
                if e.status == 409 and e.reason == 'Conflict':
                    # If we have a conflict, just load the existing object and continue
                    await deployment.refresh()


        assert len(deployment.containers) == 1
        await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', service='fiber-http', port='http')

        deployment_ = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_invalid_service_name(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        with pytest.raises(ValueError, match="Unknown Service 'invalid'"):
            await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', service='invalid')

    async def test_inject_sidecar_port_not_in_given_service(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        with pytest.raises(ValueError, match="Port 'invalid' does not exist in the Service 'fiber-http'"):
            await deployment.inject_sidecar('whatever', 'opsani/envoy-proxy:latest', service='fiber-http', port='invalid')

@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
class TestChecks:
    @pytest.fixture
    async def config(self, kube: kubetest.client.TestClient) -> servo.connectors.kubernetes.KubernetesConfiguration:
        config = servo.connectors.kubernetes.KubernetesConfiguration(
            namespace=kube.namespace,
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                servo.connectors.kubernetes.DeploymentConfiguration(
                    name="fiber-http",
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        servo.connectors.kubernetes.ContainerConfiguration(
                            name="opsani/fiber-http:latest",
                            cpu=servo.connectors.kubernetes.CPU(
                                min="250m", max="4000m", step="125m"
                            ),
                            memory=servo.connectors.kubernetes.Memory(
                                min="128MiB", max="4.0GiB", step="128MiB"
                            ),
                        )
                    ],
                )
            ],
        )
        return config

    @pytest.fixture(autouse=True)
    def wait_for_manifests(self, kube: kubetest.client.TestClient) -> None:
        kube.wait_for_registered()

    async def test_check_version(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_version")
        )
        assert results
        assert results[-1].success

    async def test_check_connectivity_success(self, config) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_connectivity")
        )
        assert len(results) == 1
        assert results[0].success

    async def test_check_connectivity_bad_hostname(self, config) -> None:
        async with tests.helpers.kubernetes_asyncio_client_overrides(host="https://localhost:4321"):
            checks = servo.connectors.kubernetes.KubernetesChecks(config)
            results = await checks.run_all(
                matching=servo.checks.CheckFilter(id="check_connectivity")
            )
            assert len(results) == 1
            result = results[0]
            assert not result.success
            assert "Cannot connect to host localhost:4321" in str(result.exception)

    async def test_check_permissions_success(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_permissions"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_permissions"
        assert result.success

    async def test_check_permissions_fails(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        # TODO: Delete the Role? Remove a permission?
        ...

    async def test_check_namespace_success(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_namespace"
        assert result.success, f"expected success but failed: {result}"

    async def test_check_namespace_doesnt_exist(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> None:
        config.namespace = "INVALID"
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_namespace"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_deployment(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> None:
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config, matching=servo.checks.CheckFilter(id="check_deployments_item_0")
        )
        assert results
        result = results[-1]
        assert result.id == "check_deployments_item_0"
        assert result.success

    async def test_check_deployment_doesnt_exist(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        config.deployments[0].name = "INVALID"
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_deployments_item_0"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_deployments_item_0"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_resource_requirements(self, config: servo.connectors.kubernetes.KubernetesConfiguration) -> None:
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config, matching=servo.checks.CheckFilter(id="check_resource_requirements_item_0")
        )
        assert results
        result = results[-1]
        assert result.id, "check_resource_requirements_item_0"
        assert result.success, f"Checking resource requirements \"{config.deployments[0].name}\" in namespace \"{config.namespace}\" failed: {result.exception or result.message or result}"

    async def test_check_resource_requirements_fail(self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube) -> None:
        # Zero out the CPU settings
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(limits={"cpu": None}, requests={"cpu": None})
        await deployment.patch()
        await deployment.wait_until_ready()

        # Fail the check because the CPU isn't limited
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config, matching=servo.checks.CheckFilter(id="check_resource_requirements_item_0")
        )
        assert results
        result = results[-1]
        assert result.id, "check_resource_requirements_item_0"
        assert not result.success, f"Checking resource requirements \"{config.deployments[0].name}\" in namespace \"{config.namespace}\" failed: {result.exception or result.message or result}"

    async def test_deployments_are_ready(self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube) -> None:
        # Set the CPU request implausibly high to force it into pending
        deployment = await servo.connectors.kubernetes.Deployment.read("fiber-http", kube.namespace)
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(limits={"cpu": None}, requests={"cpu": "500"})
        await deployment.patch()
        try:
            await asyncio.wait_for(deployment.wait_until_ready(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        # Fail because the Pod is stuck in pending
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config, matching=servo.checks.CheckFilter(id="check_resource_requirements_item_0")
        )
        assert results
        result = results[-1]
        assert result.id, "check_resource_requirements_item_0"
        assert not result.success, f"Checking resource requirements \"{config.deployments[0].name}\" in namespace \"{config.namespace}\" failed: {result.exception or result.message or result}"

@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
class TestService:
    @pytest.fixture(autouse=True)
    async def wait(self, kube) -> None:
        kube.wait_for_registered()
        await asyncio.sleep(0.0001)


    async def test_read_service(self, kube: kubetest.client.TestClient) -> None:
        svc = await servo.connectors.kubernetes.Service.read("fiber-http", kube.namespace)
        assert svc
        assert svc.obj.metadata.name == "fiber-http"
        assert svc.obj.metadata.namespace == kube.namespace


    async def test_patch_service(self, kube: kubetest.client.TestClient) -> None:
        svc = await servo.connectors.kubernetes.Service.read("fiber-http", kube.namespace)
        assert svc
        sentinel_value = hashlib.blake2b(
            str(datetime.datetime.now()).encode("utf-8"), digest_size=4
        ).hexdigest()
        svc.obj.metadata.labels["testing.opsani.com"] = sentinel_value
        await svc.patch()
        await svc.refresh()
        assert svc.obj.metadata.labels["testing.opsani.com"] == sentinel_value

@pytest.mark.parametrize(
    "value, step, expected_lower, expected_upper",
    [
        ('1.3GiB', '128MiB', '1.0GiB', '1.5GiB'),
        ('756MiB', '128MiB', '640.0MiB', '768.0MiB'),
        ('96MiB', '32MiB', '96.0MiB', '128.0MiB'),
        ('32MiB', '96MiB', '96.0MiB', '192.0MiB'),
        ('4.4GiB', '128MiB', '4.0GiB', '4.5GiB'),
        ('4.5GiB', '128MiB', '4.5GiB', '5.0GiB'),
        ('128MiB', '128MiB', '128.0MiB', '256.0MiB'),
    ]
)
def test_step_alignment_calculations_memory(value, step, expected_lower, expected_upper) -> None:
    value_bytes, step_bytes = servo.connectors.kubernetes.ShortByteSize.validate(value), servo.connectors.kubernetes.ShortByteSize.validate(step)
    lower, upper = _suggest_step_aligned_values(value_bytes, step_bytes, in_repr=servo.connectors.kubernetes.Memory.human_readable)
    assert lower == expected_lower
    assert upper == expected_upper
    assert _is_step_aligned(servo.connectors.kubernetes.ShortByteSize.validate(lower), step_bytes)
    assert _is_step_aligned(servo.connectors.kubernetes.ShortByteSize.validate(upper), step_bytes)

@pytest.mark.parametrize(
    "value, step, expected_lower, expected_upper",
    [
        ('250m', '64m', '192m', '256m'),
        ('4100m', '250m', '4', '4250m'),
        ('3', '100m', '3', '3100m'),
    ]
)
def test_step_alignment_calculations_cpu(value, step, expected_lower, expected_upper) -> None:
    value_millicores, step_millicores = servo.connectors.kubernetes.Millicore.parse(value), servo.connectors.kubernetes.Millicore.parse(step)
    lower, upper = _suggest_step_aligned_values(value_millicores, step_millicores, in_repr=servo.connectors.kubernetes.CPU.human_readable)
    assert lower == expected_lower
    assert upper == expected_upper
    assert _is_step_aligned(servo.connectors.kubernetes.Millicore.parse(lower), step_millicores)
    assert _is_step_aligned(servo.connectors.kubernetes.Millicore.parse(upper), step_millicores)

def test_cpu_not_step_aligned() -> None:
    with pytest.raises(pydantic.ValidationError, match=re.escape("CPU('cpu' 250m-4100m, 125m) max is not step aligned: 4100m is not a multiple of 125m (consider 4 or 4125m).")):
        servo.connectors.kubernetes.CPU(
            min="250m", max="4100m", step="125m"
        )

def test_memory_not_step_aligned() -> None:
    with pytest.raises(pydantic.ValidationError, match=re.escape("Memory('mem' 256.0MiB-4.1GiB, 128.0MiB) max is not step aligned: 4.1GiB is not a multiple of 128.0MiB (consider 4.0GiB or 4.5GiB).")):
        servo.connectors.kubernetes.Memory(
            min="256.0MiB", max="4.1GiB", step="128.0MiB"
        )
