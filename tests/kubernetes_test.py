import asyncio
import datetime
import hashlib
import re

import kubernetes_asyncio
import kubernetes_asyncio.client
import kubetest.client
import pydantic
import pytest

import servo
import servo.connectors.kubernetes
import tests.helpers
from servo.types.settings import _is_step_aligned, _suggest_step_aligned_values

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.usefixtures("kubernetes_asyncio_config"),
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
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment.containers) == 1
        await deployment.inject_sidecar(
            "whatever", "opsani/envoy-proxy:latest", port=8480
        )

        deployment_ = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_port_number_string(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment.containers) == 1
        await deployment.inject_sidecar(
            "whatever", "opsani/envoy-proxy:latest", port="8480"
        )

        deployment_ = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_port_conflict(self, kube):
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        with pytest.raises(
            ValueError,
            match="Port conflict: Deployment 'fiber-http' already exposes port 8480 through an existing container",
        ):
            await deployment.inject_sidecar(
                "whatever", "opsani/envoy-proxy:latest", port=8481, service_port=8480
            )

    async def test_inject_sidecar_by_service(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )

        assert len(deployment.containers) == 1
        await deployment.inject_sidecar(
            "whatever", "opsani/envoy-proxy:latest", service="fiber-http"
        )

        deployment_ = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_service_and_port_number(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )

        assert len(deployment.containers) == 1
        await deployment.inject_sidecar(
            "whatever", "opsani/envoy-proxy:latest", service="fiber-http", port=80
        )

        deployment_ = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_by_service_and_port_name(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        # NOTE: This can generate a 409 Conflict failure under CI
        for _ in range(3):
            try:
                # change the container port so we don't conflict
                deployment.obj.spec.template.spec.containers[0].ports[
                    0
                ].container_port = 9999
                await deployment.replace()
                break

            except kubernetes_asyncio.client.exceptions.ApiException as e:
                if e.status == 409 and e.reason == "Conflict":
                    # If we have a conflict, just load the existing object and continue
                    await deployment.refresh()

        assert len(deployment.containers) == 1
        await deployment.inject_sidecar(
            "whatever", "opsani/envoy-proxy:latest", service="fiber-http", port="http"
        )

        deployment_ = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert len(deployment_.containers) == 2

    async def test_inject_sidecar_invalid_service_name(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        with pytest.raises(ValueError, match="Unknown Service 'invalid'"):
            await deployment.inject_sidecar(
                "whatever", "opsani/envoy-proxy:latest", service="invalid"
            )

    async def test_inject_sidecar_port_not_in_given_service(self, kube) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        with pytest.raises(
            ValueError,
            match="Port 'invalid' does not exist in the Service 'fiber-http'",
        ):
            await deployment.inject_sidecar(
                "whatever",
                "opsani/envoy-proxy:latest",
                service="fiber-http",
                port="invalid",
            )


@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
class TestChecks:
    @pytest.fixture
    async def config(
        self, kube: kubetest.client.TestClient
    ) -> servo.connectors.kubernetes.KubernetesConfiguration:
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
                            name="fiber-http",
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

    async def test_check_version(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_kubernetes_version")
        )
        assert results
        assert results[-1].success

    async def test_check_connectivity_success(self, config) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_kubernetes_connectivity")
        )
        assert len(results) == 1
        assert results[0].success

    async def test_check_connectivity_bad_hostname(self, config) -> None:
        async with tests.helpers.kubernetes_asyncio_client_overrides(
            host="https://localhost:4321"
        ):
            checks = servo.connectors.kubernetes.KubernetesChecks(config)
            results = await checks.run_all(
                matching=servo.checks.CheckFilter(id="check_kubernetes_connectivity")
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
            matching=servo.checks.CheckFilter(id=["check_kubernetes_permissions"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_kubernetes_permissions"
        assert result.success

    async def test_check_permissions_fails(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        # TODO: Delete the Role? Remove a permission?
        ...

    async def test_check_namespace_success(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_kubernetes_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_kubernetes_namespace"
        assert result.success, f"expected success but failed: {result}"

    async def test_check_namespace_doesnt_exist(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        config.namespace = "INVALID"
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_kubernetes_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_kubernetes_namespace"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_deployment(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(id="check_kubernetes_deployments_item_0"),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_deployments_item_0"
        assert result.success

    async def test_check_deployment_doesnt_exist(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        config.deployments[0].name = "INVALID"
        checks = servo.connectors.kubernetes.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(
                id=["check_kubernetes_deployments_item_0"]
            )
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_kubernetes_deployments_item_0"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_resource_requirements(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration
    ) -> None:
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_resource_requirements_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_resource_requirements_item_0"
        assert (
            result.success
        ), f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'

    async def test_check_resource_requirements_configured_get(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube
    ) -> None:
        # Zero out the CPU setting for requests and Memory setting for limits
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(
            limits={"memory": None}, requests={"cpu": None}
        )
        await deployment.patch()
        await deployment.wait_until_ready()

        # Update resource config to require limits for CPU and requests for memory
        config.deployments[0].containers[0].cpu.get = [
            servo.connectors.kubernetes.ResourceRequirement.limit
        ]
        config.deployments[0].containers[0].memory.get = [
            servo.connectors.kubernetes.ResourceRequirement.request
        ]

        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_resource_requirements_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_resource_requirements_item_0"
        assert (
            result.success
        ), f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'

    async def test_check_resource_requirements_fail(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube
    ) -> None:
        # Zero out the CPU settings
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(
            limits={"cpu": None}, requests={"cpu": None}
        )
        await deployment.patch()
        await deployment.wait_until_ready()

        # Fail the check because the CPU isn't limited
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_resource_requirements_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_resource_requirements_item_0"
        failed_message = f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'
        assert not result.success, failed_message
        assert (
            str(result.exception)
            == "Deployment fiber-http target container fiber-http spec does not define the resource cpu. At least one of the following must be specified: requests, limits"
        ), failed_message

    async def test_check_resource_requirements_cpu_config_mismatch(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube
    ) -> None:
        # Zero out the CPU setting for requests
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(
            requests={"cpu": None}
        )
        await deployment.patch()
        await deployment.wait_until_ready()

        # Update resource config to require requests
        config.deployments[0].containers[0].cpu.get = [
            servo.connectors.kubernetes.ResourceRequirement.request
        ]

        # Fail the check because the CPU doesn't define requests
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_resource_requirements_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_resource_requirements_item_0"
        failed_message = f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'
        assert not result.success, failed_message
        assert (
            str(result.exception)
            == "Deployment fiber-http target container fiber-http spec does not define the resource cpu. At least one of the following must be specified: requests"
        ), failed_message

    async def test_check_resource_requirements_mem_config_mismatch(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube
    ) -> None:
        # Zero out the Memory setting for requests
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(
            requests={"memory": None}
        )
        await deployment.patch()
        await deployment.wait_until_ready()

        # Update resource config to require requests
        config.deployments[0].containers[0].memory.get = [
            servo.connectors.kubernetes.ResourceRequirement.request
        ]

        # Fail the check because the Memory doesn't define requests
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_resource_requirements_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_resource_requirements_item_0"
        failed_message = f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'
        assert not result.success, failed_message
        assert (
            str(result.exception)
            == "Deployment fiber-http target container fiber-http spec does not define the resource memory. At least one of the following must be specified: requests"
        ), failed_message

    async def test_deployments_are_ready(
        self, config: servo.connectors.kubernetes.KubernetesConfiguration, kube
    ) -> None:
        # Set the CPU request implausibly high to force it into pending
        deployment = await servo.connectors.kubernetes.Deployment.read(
            "fiber-http", kube.namespace
        )
        assert deployment
        container = deployment.containers[0]
        container.resources = kubernetes_asyncio.client.V1ResourceRequirements(
            limits={"cpu": None}, requests={"cpu": "500"}
        )
        await deployment.patch()
        try:
            await asyncio.wait_for(deployment.wait_until_ready(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        # Fail because the Pod is stuck in pending
        results = await servo.connectors.kubernetes.KubernetesChecks.run(
            config,
            matching=servo.checks.CheckFilter(
                id="check_kubernetes_deployments_are_ready_item_0"
            ),
        )
        assert results
        result = results[-1]
        assert result.id == "check_kubernetes_deployments_are_ready_item_0"
        failed_message = f'Checking resource requirements "{config.deployments[0].name}" in namespace "{config.namespace}" failed: {result.exception or result.message or result}'
        assert not result.success, failed_message
        assert (
            str(result.exception) == 'Deployment "fiber-http" is not ready'
        ), failed_message


@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
class TestService:
    @pytest.fixture(autouse=True)
    async def wait(self, kube) -> None:
        kube.wait_for_registered()
        await asyncio.sleep(0.0001)

    async def test_read_service(self, kube: kubetest.client.TestClient) -> None:
        svc = await servo.connectors.kubernetes.Service.read(
            "fiber-http", kube.namespace
        )
        assert svc
        assert svc.obj.metadata.name == "fiber-http"
        assert svc.obj.metadata.namespace == kube.namespace

    async def test_patch_service(self, kube: kubetest.client.TestClient) -> None:
        svc = await servo.connectors.kubernetes.Service.read(
            "fiber-http", kube.namespace
        )
        assert svc
        sentinel_value = hashlib.blake2b(
            str(datetime.datetime.now()).encode("utf-8"), digest_size=4
        ).hexdigest()
        svc.obj.metadata.labels["testing.opsani.com"] = sentinel_value
        await svc.patch()
        await svc.refresh()
        assert svc.obj.metadata.labels["testing.opsani.com"] == sentinel_value


@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
async def test_get_latest_pods(kube: kubetest.client.TestClient) -> None:
    kube.wait_for_registered()
    # Cache initially created replicaset
    _, old_rset = kube.get_replicasets().popitem()

    # Generate a new replicaset
    kube_dep = kube.get_deployments()["fiber-http"]
    kube_dep.obj.spec.template.spec.containers[0].resources.requests["memory"] = "128Mi"
    kube_dep.api_client.patch_namespaced_deployment(
        kube_dep.name, kube_dep.namespace, kube_dep.obj
    )

    servo_dep = await servo.connectors.kubernetes.Deployment.read(
        "fiber-http", kube.namespace
    )

    async def wait_for_new_replicaset():
        while len(kube.get_replicasets()) < 2:
            await asyncio.sleep(0.1)

    await asyncio.wait_for(wait_for_new_replicaset(), timeout=2)

    for _ in range(10):
        latest_pods = await servo_dep.get_latest_pods()
        # Check the latest pods aren't from the old replicaset
        for pod in latest_pods:
            for ow in pod.obj.metadata.owner_references:
                assert ow.name != old_rset.obj.metadata.name

        await asyncio.sleep(0.1)


@pytest.mark.parametrize(
    "value, step, expected_lower, expected_upper",
    [
        ("1.3Gi", "128Mi", "1.0Gi", "1.5Gi"),
        ("756Mi", "128Mi", "640.0Mi", "768.0Mi"),
        ("96Mi", "32Mi", "96.0Mi", "128.0Mi"),
        ("32Mi", "96Mi", "96.0Mi", "192.0Mi"),
        ("4.4Gi", "128Mi", "4.0Gi", "4.5Gi"),
        ("4.5Gi", "128Mi", "4.5Gi", "5.0Gi"),
        ("128Mi", "128Mi", "128.0Mi", "256.0Mi"),
    ],
)
def test_step_alignment_calculations_memory(
    value, step, expected_lower, expected_upper
) -> None:
    value_bytes, step_bytes = servo.connectors.kubernetes.ShortByteSize.validate(
        value
    ), servo.connectors.kubernetes.ShortByteSize.validate(step)
    lower, upper = _suggest_step_aligned_values(
        value_bytes,
        step_bytes,
        in_repr=servo.connectors.kubernetes.Memory.human_readable,
    )
    assert lower == expected_lower
    assert upper == expected_upper
    assert _is_step_aligned(
        servo.connectors.kubernetes.ShortByteSize.validate(lower), step_bytes
    )
    assert _is_step_aligned(
        servo.connectors.kubernetes.ShortByteSize.validate(upper), step_bytes
    )


@pytest.mark.parametrize(
    "value, step, expected_lower, expected_upper",
    [
        ("250m", "64m", "192m", "256m"),
        ("4100m", "250m", "4", "4.25"),
        ("3", "100m", "3", "3.1"),
    ],
)
def test_step_alignment_calculations_cpu(
    value, step, expected_lower, expected_upper
) -> None:
    value_cores, step_cores = servo.connectors.kubernetes.Core.parse(
        value
    ), servo.connectors.kubernetes.Core.parse(step)
    lower, upper = _suggest_step_aligned_values(
        value_cores, step_cores, in_repr=servo.connectors.kubernetes.CPU.human_readable
    )
    assert lower == expected_lower
    assert upper == expected_upper
    assert _is_step_aligned(servo.connectors.kubernetes.Core.parse(lower), step_cores)
    assert _is_step_aligned(servo.connectors.kubernetes.Core.parse(upper), step_cores)


def test_cpu_not_step_aligned() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match=re.escape(
            "CPU('cpu' 250m-4.1, 125m) min/max difference is not step aligned: 3.85 is not a multiple of 125m (consider min 350m or 225m, max 4 or 4.125)."
        ),
    ):
        servo.connectors.kubernetes.CPU(min="250m", max="4100m", step="125m")


def test_memory_not_step_aligned() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match=re.escape(
            "Memory('mem' 256.0Mi-4.1Gi, 128.0Mi) min/max difference is not step aligned: 3.8125Gi is not a multiple of 128Mi (consider min 320Mi or 192Mi, max 4Gi or 4.125Gi)."
        ),
    ):
        servo.connectors.kubernetes.Memory(
            min="256.0MiB", max="4.0625GiB", step="128.0MiB"
        )


def test_copying_cpu_with_invalid_value_does_not_raise() -> None:
    cpu = servo.connectors.kubernetes.CPU(min="250m", max="4", step="125m", value=None)

    # Trigger an error
    with pytest.raises(
        pydantic.ValidationError, match=re.escape("5 is outside of the range 250m-4")
    ):
        cpu.value = "5"

    # Use copy + update to hydrate the value
    cpu_copy = cpu.copy(update={"value": "5"})
    assert cpu_copy.value == "5"


@pytest.mark.parametrize(
    "value, resource_type, expected_autoset",
    [
        (
            servo.connectors.kubernetes.Core.parse("1"),
            "cpu",
            servo.connectors.kubernetes.CPU(min="250m", max="3000m", step="125m"),
        ),
        (
            servo.connectors.kubernetes.Core.parse("2"),
            "cpu",
            servo.connectors.kubernetes.CPU(min="500m", max="6000m", step="125m"),
        ),
        (
            servo.connectors.kubernetes.ShortByteSize.validate("2Gi"),
            "memory",
            servo.connectors.kubernetes.Memory(
                min="512.0MiB", max="6.0GiB", step="128.0MiB"
            ),
        ),
    ],
)
def test_autoset_resource_range(value, resource_type, expected_autoset):

    autoset_value = servo.connectors.kubernetes.autoset_resource_range(
        resource_type=resource_type, value=value
    )
    assert autoset_value == expected_autoset
