import sys

from pathlib import Path
from typing import Optional

import pytest
from kubernetes import client
from kubernetes.client.rest import ApiException

from kubetest import condition, response, utils

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
    'fiber-http.yaml'
])
def test_fiber_http_and_envoy(kube):
    kube.wait_for_registered(timeout=60)

    deployments = kube.get_deployments()
    web_deploy = deployments.get('fiber-http-deployment')
    assert web_deploy is not None

    pods = web_deploy.get_pods()
    assert len(pods) == 1, 'fiber-http should deploy with one replica'

    pod = pods[0]
    pod.wait_until_ready(timeout=30)

    # Check containers
    containers = pod.get_containers()
    assert len(containers) == 2, "should have fiber-http and an envoy sidecar"
    assert containers[0].obj.name == "fiber-http"
    assert containers[1].obj.name == "envoy"

    # Check services
    response = pod.http_proxy_get('/')
    assert "move along, nothing to see here" in response.data

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
    # Make servo load test fiber-http, report the outcome in JSON

def test_deploy_servo_cohttp_vegeta_adjust() -> None:
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