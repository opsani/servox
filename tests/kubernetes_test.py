import asyncio
from asyncio.streams import StreamReader
from pathlib import Path
import pytest

import logging
from typing import Dict, List, Optional, Tuple, Union

from kubernetes import client
from kubernetes.client.rest import ApiException

from kubetest import condition, response, utils
from servo.logging import logger

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

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

async def stream_subprocess_shell(
    cmd: str,
    *,
    cwd: Path = Path.cwd(), 
    env: Optional[Dict[str, str]] = None,
    print_output: bool = True,
    log_output: bool = True,
) -> Tuple[Optional[int], List[str], List[str]]:
    """
    Create an asynchronous subprocess shell and stream its output.

    Returns the exit code and two lists of strings containing output 
    from stdout and stderr, respectively.
    """
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_list: List[str] = []
    stderr_list: List[str] = []
    try:
        await asyncio.wait([
            _process_stream(proc.stdout, stdout_list, print_output=print_output, log_output=log_output), 
            _process_stream(proc.stderr, stderr_list, print_output=print_output, log_output=log_output)
        ])
        return_code = proc.returncode
    except Exception:
        proc.terminate()
        return_code = -1
    
    return return_code, stdout_list, stderr_list

async def _process_stream(
    stream: StreamReader, 
    output_list: Optional[List[str]] = None,
    *,
    print_output: bool = True,
    log_output: bool = True,
) -> None:
    while True:
        line = await stream.readline()
        if line:
            line = line.decode('utf-8')
            if output_list is not None:
                output_list.append(line)
            if log_output:
                logger.debug(f"command output: {line}")
            if print_output:
                print(line, end='')
        else:
            break

async def build_docker_image(tag: str = "servox:latest", *, preamble: Optional[str] = None) -> str:
    root_path = Path(__file__).parents[1]
    exit_code, stdout, stderr = await stream_subprocess_shell(
        f"{preamble or 'true'} && DOCKER_BUILDKIT=1 docker build -t {tag} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:latest {root_path}",
        print_output=False
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

async def test_build_and_run_servo_image(servo_image: str) -> None:
    exit_code, stdout, stderr = await stream_subprocess_shell(
        f"docker run --rm -i {servo_image} servo --help",
        print_output=False
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "Operational Commands" in str(stdout)


async def test_run_servo_on_kubernetes(kube, minikube_servo_image: str) -> None:
    command = (f'eval $(minikube -p minikube docker-env) && kubectl --context=minikube run servo --rm --attach --image-pull-policy=Never --restart=Never --image="{minikube_servo_image}" --'
               ' servo --optimizer example.com/app --token 123456 version')
    exit_code, stdout, stderr = await stream_subprocess_shell(
        command,
        print_output=False,
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