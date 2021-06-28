from tests.connectors.kubernetes_test import namespace
import loguru
import pytest

from typing import Iterator, Optional

import servo.checks
import servo.connectors.kubernetes

MINIKUBE_PROFILE_CONTEXT = "servox-minikube"

pytestmark = [
    pytest.mark.system,
    pytest.mark.minikube_profile.with_args(MINIKUBE_PROFILE_CONTEXT)
]

# override kubetest context
@pytest.fixture
def kubecontext() -> Optional[str]:
    return MINIKUBE_PROFILE_CONTEXT

# TODO: change to kind cluster if VPA supported
@pytest.fixture
async def vertical_pod_autoscaler(kubeconfig, kubecontext, minikube, subprocess) -> Iterator[None]:
    # enable metrics server addon
    exit_code, _, stderr = await subprocess(f"KUBECONFIG={kubeconfig} minikube addons -p {MINIKUBE_PROFILE_CONTEXT} enable metrics-server", print_output=True)
    assert exit_code == 0, f"enable metrics server failed: {stderr}"

    # clone autoscaler repo
    exit_code, _, stderr = await subprocess("git clone https://github.com/kubernetes/autoscaler.git")
    assert exit_code == 0, f"clone autoscaler repo failed: {stderr}"

    # cache current context
    restore_context = None
    exit_code, stdout, stderr = await subprocess("kubectl --kubeconfig={kubeconfig} config current-context")
    if exit_code == 0:
        restore_context = stdout[0]
    else:
        loguru.logger.warning(f"get current-context failed: {stderr}")

    # set current context to minikube for VPA installation
    exit_code, _, stderr = await subprocess(f"kubectl --kubeconfig={kubeconfig} config use-context {kubecontext}")
    assert exit_code == 0, f"set default context to minikube failed: {stderr}"

    # install vertical pod autoscaler
    exit_code, _, stderr = await subprocess(f"KUBECONFIG={kubeconfig} ./hack/vpa-up.sh", print_output=True, cwd="autoscaler/vertical-pod-autoscaler")
    assert exit_code == 0, f"set default context to minikube failed: {stderr}"

    yield

    # restore previous context
    if restore_context:
        exit_code, stdout, stderr = await subprocess(f"kubectl --kubeconfig={kubeconfig} config use-context {restore_context}")
        assert exit_code == 0, f"restoring default context previous value failed: {stderr}"

async def test_check_for_vpa_failure(vertical_pod_autoscaler, kubernetes_asyncio_config, kubeconfig, kubecontext, kube, subprocess) -> None:
    # NOTE: have to use subprocess as kubetest client does not recognize VPA objects
    exit_code, _, stderr = await subprocess(
        f"kubectl --kubeconfig={kubeconfig} --context={kubecontext} apply -n {kube.namespace} -f autoscaler/vertical-pod-autoscaler/examples/hamster.yaml",
        print_output=True
    )
    assert exit_code == 0, f"apply VPA example deployment failed: {stderr}"

    exit_code, _, stderr = await subprocess(
        f"kubectl --kubeconfig={kubeconfig} --context={kubecontext} wait --for=condition=available --timeout=60s -n {kube.namespace} deployment hamster",
        print_output=True
    )
    assert exit_code == 0, f"wait for VPA example deployment failed: {stderr}"


    checks = servo.connectors.kubernetes.KubernetesChecks(
        servo.connectors.kubernetes.KubernetesConfiguration(
            namespace=kube.namespace,
            deployments=[
                servo.connectors.kubernetes.DeploymentConfiguration(
                    name="hamster",
                    replicas=servo.Replicas(
                        min=1,
                        max=4,
                    ),
                    containers=[
                        servo.connectors.kubernetes.ContainerConfiguration(
                            name="hamster",
                            cpu=servo.connectors.kubernetes.CPU(min="100m", max="500m", step="100m"),
                            memory=servo.connectors.kubernetes.Memory(min="50MiB", max="500MiB", step="50MiB"),
                        )
                    ],
                )
            ],
        )
    )
    results = await checks.run_all(
        matching=servo.checks.CheckFilter(id=["check_deployments_for_vpa_item_0"])
    )
    assert len(results)
    result = results[-1]
    assert result.id == "check_deployments_for_vpa_item_0"
    assert not result.success
    assert result.exception
    assert str(result.exception) == 'Deployment "hamster" is managed by VPA "hamster-vpa"'
