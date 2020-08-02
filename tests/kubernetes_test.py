import sys

from pathlib import Path
from typing import Optional

import pytest
from kubernetes import client
from kubernetes.client.rest import ApiException

from kubetest import condition, response, utils
from servo.logging import logger
from servo.utilities import stream_subprocess_shell

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

@pytest.fixture
def kubeconfig() -> str:
    config_path = Path(__file__).parents[0] / 'kubeconfig'
    if not config_path.exists():
        raise FileNotFoundError(f"no kubeconfig file found at '{config_path}': configure a test cluster and add the kubeconfig file")
    return str(config_path)

@pytest.mark.applymanifests('manifests', files=[
    'nginx.yaml'
])
def test_nginx(kube):
    # wait for the manifests loaded by the 'applymanifests' marker
    # to be ready on the cluster
    kube.wait_for_registered(timeout=30)

    deployments = kube.get_deployments()
    nginx_deploy = deployments.get('nginx-deployment')
    assert nginx_deploy is not None

    pods = nginx_deploy.get_pods()
    assert len(pods) == 1, 'nginx should deploy with one replica'

    for pod in pods:
        containers = pod.get_containers()
        assert len(containers) == 1, 'nginx pod should have one container'

        resp = pod.http_proxy_get('/')
        assert '<h1>Welcome to nginx!</h1>' in resp.data


@pytest.mark.applymanifests('manifests', files=[
    'co-http.yaml'
])
def test_co_http_and_envoy(kube):
    kube.wait_for_registered(timeout=60)

    deployments = kube.get_deployments()
    web_deploy = deployments.get('co-http-deployment')
    assert web_deploy is not None

    pods = web_deploy.get_pods()
    assert len(pods) == 1, 'co-http should deploy with one replica'

    pod = pods[0]
    pod.wait_until_ready(timeout=30)

    # Check containers
    containers = pod.get_containers()
    assert len(containers) == 2, "should have co-http and an envoy sidecar"
    assert containers[0].obj.name == "co-http"
    assert containers[1].obj.name == "envoy"

    # Check services
    response = pod.http_proxy_get('/')
    assert "busy for " in response.data

    # TODO: Ugly hack to control port number
    pod.name = pod.name + ":9901"
    response = pod.http_proxy_get('/stats/prometheus')
    assert "envoy_http_downstream_cx_length_ms_count" in response.data


@pytest.mark.applymanifests('manifests', files=["prometheus.yaml"])
def test_prometheus(kube) -> None:
    kube.wait_for_registered(timeout=30)

    deployments = kube.get_deployments()
    prom_deploy = deployments.get('prometheus-core')
    assert prom_deploy is not None

    pods = prom_deploy.get_pods()
    assert len(pods) == 1, 'prom_deploy should deploy with one replica'

    # Check that Prometheus is there by referencing string in the HTML body
    pod = pods[0]
    pod.name = pod.name + ":9090"
    response = pod.http_proxy_get('/')
    assert "Prometheus Time Series Collection and Processing Server" in response.data

# TODO: Move into test helpers and conftest
from typing import Awaitable, Callable, List
from servo.utilities import SubprocessResult, Timeout

class SubprocessTestHelper:    
    async def shell(
        self,
        cmd: str,
        *,
        timeout: Timeout = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,        
    ) -> SubprocessResult:
        stdout: List[str] = []
        stderr: List[str] = []

        def create_output_callback(name: str, output: List[str]) -> Callable[[str], Awaitable[None]]:
            async def output_callback(msg: str) -> None:
                output.append(msg)
                m = f"[{name}] {msg}"
                if print_output:
                    print(m)
                if log_output:
                    logger.debug(m)

            return output_callback
        
        print(f"\n❯ Executing `{cmd}`")
        return_code = await stream_subprocess_shell(
            cmd,
            timeout=timeout,
            stdout_callback=create_output_callback("stdout", stdout),
            stderr_callback=create_output_callback("stderr", stderr),
        )
        return SubprocessResult(return_code, stdout, stderr)

    async def __call__(
        self,
        cmd: str,
        *,
        timeout: Timeout = None,
        print_output: bool = False,
        log_output: bool = True,
        **kwargs,        
    ) -> SubprocessResult:
        return await self.shell(cmd, timeout=timeout, print_output=print_output, log_output=log_output, **kwargs)
    
@pytest.fixture()
async def subprocess() -> SubprocessTestHelper:
    return SubprocessTestHelper()

async def build_docker_image(tag: str = "servox:latest", *, preamble: Optional[str] = None, **kwargs) -> str:
    root_path = Path(__file__).parents[1]
    subprocess = SubprocessTestHelper()
    exit_code, stdout, stderr = await subprocess(
        f"{preamble or 'true'} && DOCKER_BUILDKIT=1 docker build -t {tag} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:latest {root_path}",        
        **kwargs,
    )
    if exit_code != 0:
        error = '\n'.join(stderr)
        raise RuntimeError(f"Docker build failed with exit code {exit_code}: error: {error}")
    
    return tag

@pytest.fixture()
async def servo_image() -> str:
    return await build_docker_image()

@pytest.fixture()
async def minikube_servo_image(servo_image: str) -> str:
    return await build_docker_image(preamble="eval $(minikube -p minikube docker-env)")

async def test_run_servo_on_docker(servo_image: str, subprocess) -> None:
    exit_code, stdout, stderr = await subprocess(
        f"docker run --rm -i {servo_image} servo --help",
        print_output=True
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "Operational Commands" in str(stdout)

async def test_run_servo_on_minikube(kube, minikube_servo_image: str, kubeconfig, subprocess) -> None:
    command = (f'eval $(minikube -p minikube docker-env) && kubectl --kubeconfig={kubeconfig} --context=minikube run servo --attach --rm --wait --image-pull-policy=Never --restart=Never --image="{minikube_servo_image}" --'
               ' servo --optimizer example.com/app --token 123456 version')
    exit_code, stdout, stderr = await subprocess(
        command,
        print_output=True,
        timeout=20
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout)

async def test_run_servo_on_eks(servo_image: str, kubeconfig, subprocess) -> None:
    ecr_image = "207598546954.dkr.ecr.us-west-2.amazonaws.com/servox-integration-tests:latest"
    command = (
        f"docker tag {servo_image} {ecr_image}"
        f" && docker push {ecr_image}"
    )
    exit_code, stdout, stderr = await subprocess(command, print_output=True)
    assert exit_code == 0, f"image publishing failed: {stderr}"

    command = (f'kubectl --kubeconfig={kubeconfig} --context=servox-integration-tests run servo --attach --rm --wait --image-pull-policy=Always --restart=Never --image="{ecr_image}" --'
               ' servo --optimizer example.com/app --token 123456 version')
    exit_code, stdout, stderr = await subprocess(
        command,
        print_output=True,
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout)

def test_deploy_servo_cohttp_vegeta_measure() -> None:
    pass
    # Make servo load test co-http, report the outcome in JSON

def test_deploy_servo_cohttp_vegeta_adjust() -> None:
    pass
    # Make servo adjust co-http memory, report in JSON
# TODO: Tests to write...
# 1. Servo creates canary on start
# canary gets deleted on stop
# failed adjust (can't schedule)
# integration tests: ad-hoc adjust, ad-hoc measure, checks (generate config files in tmp)
# use ktunnel to bridge and return errors, garbage data
# k8s sizing tool

# Integration test k8s describe, adjust