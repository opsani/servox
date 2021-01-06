import os

import pytest

import tests.helpers


@pytest.fixture
async def servo_image(request) -> str:
    """Asynchronously build a Docker image from the current working copy and return its tag."""
    image_key = f"servo_image/{os.getpid()}"
    image = request.config.cache.get(image_key, None)
    if image is None:
        image = await tests.helpers.build_docker_image()
        request.config.cache.set(image_key, image)
    return image

@pytest.fixture
async def minikube(request, subprocess) -> str:
    """Run tests within a local minikube profile.

    The profile name is determined using the parametrized `minikube_profile` marker
    or else uses "default".
    """
    marker = request.node.get_closest_marker("minikube_profile")
    if marker:
        assert len(marker.args) == 1, f"minikube_profile marker accepts a single argument but received: {repr(marker.args)}"
        profile = marker.args[0]
    else:
        profile = "servox"

    # Start minikube and configure environment
    exit_code, _, _ = await subprocess(f"minikube start -p {profile} --interactive=false --keep-context=true --wait=true", print_output=True)
    if exit_code != 0:
        raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")

    # Yield the profile name
    try:
        yield profile

    finally:
        exit_code, _, _ = await subprocess(f"minikube stop -p {profile}", print_output=True)
        if exit_code != 0:
            raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")

@pytest.fixture
async def minikube_servo_image(minikube: str, servo_image: str, subprocess) -> str:
    """Asynchronously build a Docker image from the current working copy and cache it into the minikube repository."""
    exit_code, _, _ = await subprocess(f"minikube cache add -p {minikube} {servo_image}", print_output=True)
    if exit_code != 0:
        raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")

    yield servo_image

@pytest.fixture
async def kind(request, subprocess, kubeconfig: str, kubecontext: str) -> str:
    """Run tests within a local kind cluster.

    The cluster name is determined using the parametrized `kind_cluster` marker
    or else uses "default".
    """
    cluster = "kind"
    marker = request.node.get_closest_marker("kind_cluster")
    if marker:
        assert len(marker.args) == 1, f"kind_cluster marker accepts a single argument but received: {repr(marker.args)}"
        cluster = marker.args[0]

    # Start kind and configure environment
    # TODO: if we create it, we should delete it (with kubernetes_cluster() as foo:)
    exit_code, _, _ = await subprocess(f"kind get clusters | grep {cluster} || kind create cluster --name {cluster} --kubeconfig {kubeconfig}", print_output=True)
    if exit_code != 0:
        raise RuntimeError(f"failed running kind: exited with status code {exit_code}")

    # Yield the cluster name
    try:
        # FIXME: note sure what is up with this but kind is prefixing the cluster name
        yield cluster

    finally:
        # ensure default context is respected
        await subprocess(f"kubectl config --kubeconfig {kubeconfig} use-context {kubecontext}", print_output=True)

        # TODO: add an option to not tear down the cluster
        if not os.getenv("GITHUB_ACTIONS"):
            exit_code, _, _ = await subprocess(f"kind delete cluster --name {cluster} --kubeconfig {kubeconfig}", print_output=True)
            if exit_code != 0:
                raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")

# TODO: Replace this with a callable like: `kind.create(), kind.delete(), with kind.cluster() as ...`
# TODO: add markers for the image, cluster name.
@pytest.fixture
async def kind_servo_image(kind: str, servo_image: str, subprocess, kubeconfig: str) -> str:
    """Asynchronously build a Docker image from the current working copy and load it into kind."""
    # TODO: Figure out how to checksum this and skip it if possible
    exit_code, _, _ = await subprocess(f"kind load docker-image --name {kind} {servo_image}", print_output=True)
    if exit_code != 0:
        raise RuntimeError(f"failed running kind: exited with status code {exit_code}")

    yield servo_image
