import pytest

pytestmark = pytest.mark.system

async def test_run_servo_on_docker(servo_image: str, subprocess) -> None:
    exit_code, stdout, stderr = await subprocess(
        f"docker run --rm -i {servo_image} --help", print_output=True
    )
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "Operational Commands" in str(stdout)


@pytest.mark.skip(reason="moving away from minikube")
async def test_run_servo_on_minikube(
    minikube_servo_image: str, subprocess, kubeconfig: str,
) -> None:
    command = (
        f'kubectl --kubeconfig={kubeconfig} run servo --attach --rm --wait --image-pull-policy=Never --restart=Never --image="{minikube_servo_image}" --'
        " --optimizer example.com/app --token 123456 version"
    )
    exit_code, stdout, stderr = await subprocess(command, print_output=True, timeout=None)
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout) # lgtm[py/incomplete-url-substring-sanitization]


async def test_run_servo_on_kind(
    kind: str,
    kind_servo_image: str,
    subprocess,
    kubeconfig: str,
) -> None:
    await subprocess(f"kubectl --kubeconfig={kubeconfig} config view", print_output=True)
    command = (
        f'kubectl --kubeconfig={kubeconfig} --context kind-{kind} run servo --attach --rm --wait --image-pull-policy=Never --restart=Never --image="{kind_servo_image}" --'
        " --optimizer example.com/app --token 123456 version"
    )
    exit_code, stdout, stderr = await subprocess(command, print_output=True, timeout=None)
    assert exit_code == 0, f"servo image execution failed: {stderr}"
    assert "https://opsani.com/" in "".join(stdout) # lgtm[py/incomplete-url-substring-sanitization]
