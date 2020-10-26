from datetime import datetime

import pytest

import servo
import servo.connectors.kubernetes as kubernetes_connector
from servo.connectors.kubernetes import KubernetesConfiguration

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

@pytest.mark.applymanifests("manifests", files=["nginx.yaml"])
def test_nginx(kube) -> None:
    # wait for the manifests loaded by the 'applymanifests' marker
    # to be ready on the cluster
    kube.wait_for_registered(timeout=30)

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


@pytest.mark.applymanifests("manifests", files=["fiber-http.yaml"])
def test_fiber_http_and_envoy(kube):
    kube.wait_for_registered(timeout=60)

    deployments = kube.get_deployments()
    web_deploy = deployments.get("fiber-http-deployment")
    assert web_deploy is not None

    pods = web_deploy.get_pods()
    assert len(pods) == 1, "fiber-http should deploy with one replica"

    pod = pods[0]
    pod.wait_until_ready(timeout=30)

    # Check containers
    containers = pod.get_containers()
    assert len(containers) == 2, "should have fiber-http and an envoy sidecar"
    assert containers[0].obj.name == "fiber-http"
    assert containers[1].obj.name == "envoy"

    # Check services
    response = pod.http_proxy_get("/")
    assert "move along, nothing to see here" in response.data

    # TODO: Ugly hack to control port number
    pod.name = pod.name + ":9901"
    response = pod.http_proxy_get("/stats/prometheus")
    assert "envoy_http_downstream_cx_length_ms_count" in response.data


@pytest.mark.applymanifests("manifests", files=["prometheus.yaml"])
def test_prometheus(kube) -> None:
    kube.wait_for_registered(timeout=30)

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


async def test_run_servo_on_docker(servo_image: str, subprocess) -> None:
    exit_code, stdout, stderr = await subprocess(
        f"docker run --rm -i {servo_image} servo --help", print_output=True
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "Operational Commands" in str(stdout)


async def test_run_servo_on_minikube(
    kube, minikube_servo_image: str, kubeconfig, subprocess
) -> None:
    command = (
        f'eval $(minikube -p minikube docker-env) && kubectl --kubeconfig={kubeconfig} --context=minikube run servo --attach --rm --wait --image-pull-policy=Never --restart=Never --image="{minikube_servo_image}" --'
        " servo --optimizer example.com/app --token 123456 version"
    )
    exit_code, stdout, stderr = await subprocess(command, print_output=True, timeout=20)
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout) # lgtm[py/incomplete-url-substring-sanitization]


async def test_run_servo_on_eks(servo_image: str, kubeconfig, subprocess) -> None:
    ecr_image = (
        "207598546954.dkr.ecr.us-west-2.amazonaws.com/servox-integration-tests:latest"
    )
    command = f"docker tag {servo_image} {ecr_image}" f" && docker push {ecr_image}"
    exit_code, stdout, stderr = await subprocess(command, print_output=True)
    assert exit_code == 0, f"image publishing failed: {stderr}"

    command = (
        f'kubectl --kubeconfig={kubeconfig} --context=servox-integration-tests run servo --attach --rm --wait --image-pull-policy=Always --restart=Never --image="{ecr_image}" --'
        " servo --optimizer example.com/app --token 123456 version"
    )
    exit_code, stdout, stderr = await subprocess(
        command,
        print_output=True,
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout) # lgtm[py/incomplete-url-substring-sanitization]


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


class TestChecks:
    @pytest.fixture
    async def config(self) -> kubernetes_connector.KubernetesConfiguration:
        config = kubernetes_connector.KubernetesConfiguration(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            deployments=[
                kubernetes_connector.DeploymentConfiguration(
                    name="app",
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        kubernetes_connector.ContainerConfiguration(
                            name="opsani/fiber-http:latest",
                            cpu=kubernetes_connector.CPU(
                                min="250m", max="4000m", step="125m"
                            ),
                            memory=kubernetes_connector.Memory(
                                min="256MiB", max="4.0GiB", step="128MiB"
                            ),
                        )
                    ],
                )
            ],
        )
        return config

    async def test_check_version(self, config) -> None:
        await config.load_kubeconfig()
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_version")
        )
        assert results
        assert results[-1].success

    async def test_check_connectivity_success(self, config) -> None:
        await config.load_kubeconfig()
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_connectivity")
        )
        assert len(results) == 1
        assert results[0].success

    async def test_check_connectivity_bad_hostname(self, config) -> None:
        # TODO: Generalize this into a helper
        from kubernetes_asyncio.client import Configuration as KubernetesAsyncioConfiguration

        await config.load_kubeconfig()

        loaded_config = KubernetesAsyncioConfiguration.get_default_copy()
        loaded_config.host = "https://localhost:4321"
        KubernetesAsyncioConfiguration.set_default(loaded_config)

        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id="check_connectivity")
        )
        assert len(results) == 1
        result = results[0]
        assert not result.success
        assert "Cannot connect to host localhost:4321" in str(result.exception)

    async def test_check_permissions_success(
        self, config: KubernetesConfiguration
    ) -> None:
        await config.load_kubeconfig()
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_permissions"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_permissions"
        assert result.success

    async def test_check_permissions_fails(
        self, config: KubernetesConfiguration
    ) -> None:
        ...

    async def test_check_namespace_success(self, config) -> None:
        await config.load_kubeconfig()
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_namespace"
        assert result.success

    async def test_check_namespace_doesnt_exist(self, config) -> None:
        await config.load_kubeconfig()
        config.namespace = "INVALID"
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_namespace"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_namespace"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_deployment(self, config) -> None:
        await config.load_kubeconfig()
        results = await kubernetes_connector.KubernetesChecks.run(
            config, matching=servo.checks.CheckFilter(id="check_deployments_item_0")
        )
        assert results
        result = results[-1]
        assert result.id == "check_deployments_item_0"
        assert result.success

    async def test_check_deployment_doesnt_exist(
        self, config: KubernetesConfiguration
    ) -> None:
        await config.load_kubeconfig()
        config.deployments[0].name = "INVALID"
        checks = kubernetes_connector.KubernetesChecks(config)
        results = await checks.run_all(
            matching=servo.checks.CheckFilter(id=["check_deployments_item_0"])
        )
        assert len(results)
        result = results[-1]
        assert result.id == "check_deployments_item_0"
        assert not result.success
        assert result.exception
        assert "Not Found" in str(result.exception)

    async def test_check_resource_requirements(self, config) -> None:
        await config.load_kubeconfig()
        results = await kubernetes_connector.KubernetesChecks.run(
            config, servo.checks.CheckFilter(id="check_resource_requirements_item_0")
        )
        assert results
        result = results[-1]
        assert result.matching, "check_resource_requirements_item_0"
        assert result.success

    # TODO: Create deployment without CPU, check it

    async def check_deployments_are_ready(self, config) -> None:
        ...
        # TODO: How do I force a deployment to be non-ready?


async def test_read_service(kubeconfig) -> None:
    svc = await kubernetes_connector.Service.read("fiber-http", "default")
    assert svc.obj.metadata.name == "fiber-http"
    assert svc.obj.metadata.namespace == "default"


async def test_patch_service(kubeconfig) -> None:
    from hashlib import blake2b

    svc = await kubernetes_connector.Service.read("fiber-http", "default")
    sentinel_value = blake2b(
        str(datetime.now()).encode("utf-8"), digest_size=4
    ).hexdigest()
    svc.obj.metadata.labels["testing.opsani.com"] = sentinel_value
    await svc.patch()
    await svc.refresh()
    assert svc.obj.metadata.labels["testing.opsani.com"] == sentinel_value
