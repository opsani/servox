import asyncio
import builtins
import contextlib
import json
import os
import pathlib
import random
import socket
import string
import pathlib
from typing import AsyncGenerator, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

import devtools
import fastapi
import httpx
import kubetest
import pytest
import typer.testing
import uvloop
import yaml

import servo
import servo.cli
import servo.connectors.kubernetes
import tests.helpers

# Add the devtools debug() function globally in tests
builtins.debug = devtools.debug


def pytest_report_header(config) -> str:
    try:
        for connector in servo.connector.ConnectorLoader().load():
            servo.logger.debug(f"Loaded {connector.__qualname__}")
    except Exception:
        servo.logger.exception(
            "failed loading connectors via discovery", backtrace=True, diagnose=True
        )

    names = list(
        map(
            lambda c: f"{c.__default_name__}-{c.version}",
                servo.Assembly.all_connector_types())
    )
    return "servo connectors: " + ", ".join(names)

@pytest.fixture
def event_loop_policy(request) -> str:
    """Return the active event loop policy for the test.

    Valid values are "default" and "uvloop".

    The default implementation uses the parametrized `event_loop_policy` marker
    to select the effective policy.
    """
    marker = request.node.get_closest_marker("event_loop_policy")
    if marker:
        assert len(marker.args) == 1, f"event_loop_policy marker accepts a single argument but received: {repr(marker.args)}"
        event_loop_policy = marker.args[0]
    else:
        event_loop_policy = "uvloop"

    valid_policies = ("default", "uvloop")
    assert event_loop_policy in valid_policies, f"invalid event_loop_policy marker: \"{event_loop_policy}\" is not in {repr(valid_policies)}"

    return event_loop_policy


@pytest.fixture
def event_loop(event_loop_policy: str) -> Iterator[asyncio.AbstractEventLoop]:
    """Yield an instance of the event loop for each test case.

    The effective event loop policy is determined by the `event_loop_policy` fixture.
    """
    if event_loop_policy == "default":
        asyncio.set_event_loop_policy(None)
    elif event_loop_policy == "uvloop":
        uvloop.install()
    else:
        raise ValueError(f"invalid event loop policy: \"{event_loop_policy}\"")

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_addoption(parser) -> None:
    """Add pytest options for enabling execution of integration tests."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_configure(config) -> None:
    """Register custom markers for use in the test suite."""
    config.addinivalue_line(
        "markers", "integration: marks integration tests with outside dependencies"
    )
    config.addinivalue_line(
        "markers", "system: marks system tests with end to end dependencies"
    )
    config.addinivalue_line(
        "markers", "event_loop_policy: marks async tests to run under a parametrized asyncio runloop policy (e.g., default or uvloop)"
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Modify the discovered pytest nodes to configure default markers.

    This methods sets asyncio as the async backend, configures a default event loop
    policy of uvloop, and configures integration tests to not be run by default.
    """
    skip_itegration = pytest.mark.skip(
        reason="add --integration option to run integration tests"
    )
    skip_system = pytest.mark.skip(reason="add --system to run system tests")

    for item in items:
        # Set asyncio + uvloop default markers as defaults
        item.add_marker(pytest.mark.asyncio)
        if not item.get_closest_marker("event_loop_policy"):
            item.add_marker(pytest.mark.event_loop_policy("uvloop"))

        # Skip slow/sensitive integration & system tests by default
        if "integration" in item.keywords and not config.getoption("--integration"):
            item.add_marker(skip_itegration)
        if "system" in item.keywords and not config.getoption("--system"):
            item.add_marker(skip_system)


@pytest.fixture()
def cli_runner() -> typer.testing.CliRunner:
    """Return a runner for testing Typer CLI applications."""
    return typer.testing.CliRunner(mix_stderr=False)


@pytest.fixture()
def servo_cli() -> servo.cli.ServoCLI:
    """Return an instance of the servo CLI Typer application."""
    return servo.cli.ServoCLI()


@pytest.fixture()
def optimizer_env() -> Iterator[None]:
    """Add values for the OPSANI_OPTIMIZER and OPSANI_TOKEN variables to the environment."""
    os.environ.update(
        {"OPSANI_OPTIMIZER": "dev.opsani.com/servox", "OPSANI_TOKEN": "123456789"}
    )
    try:
        yield
    finally:
        os.environ.pop("OPSANI_OPTIMIZER", None)
        os.environ.pop("OPSANI_TOKEN", None)


@pytest.fixture()
def optimizer() -> servo.Optimizer:
    """Return a generated optimizer instance."""
    return servo.Optimizer(id="dev.opsani.com/servox", token="123456789")


@pytest.fixture()
def servo_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to an empty servo config file."""
    config_path: pathlib.Path = tmp_path / "servo.yaml"
    config_path.touch()
    return config_path


@pytest.fixture()
def stub_servo_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a servo config file set up for running stub connectors from the test helpers."""
    config_path: pathlib.Path = tmp_path / "servo.yaml"
    settings = tests.helpers.StubBaseConfiguration(name="stub")
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    config = {"connectors": ["measure", "adjust"], "measure": measure_config_json, "adjust": {}}
    config = yaml.dump(config)
    config_path.write_text(config)
    return config_path

@pytest.fixture()
def stub_multiservo_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a servo config file set up for multi-servo execution."""
    config_path: pathlib.Path = tmp_path / "servo.yaml"
    settings = tests.helpers.StubBaseConfiguration(name="stub")
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    optimizer1 = servo.Optimizer(id="dev.opsani.com/multi-servox-1", token="123456789")
    optimizer1_config_json = json.loads(
        json.dumps(
            optimizer1.dict(
                by_alias=True,
            )
        )
    )
    config1 = {
        "optimizer": optimizer1_config_json,
        "connectors": ["measure", "adjust"],
        "measure": measure_config_json,
        "adjust": {}
    }
    optimizer2 = servo.Optimizer(id="dev.opsani.com/multi-servox-2", token="987654321")
    optimizer2_config_json = json.loads(
        json.dumps(
            optimizer2.dict(
                by_alias=True,
            )
        )
    )
    config2 = {
        "optimizer": optimizer2_config_json,
        "connectors": ["measure", "adjust"],
        "measure": measure_config_json,
        "adjust": {}
    }
    config_yaml = yaml.dump_all([config1, config2])
    config_path.write_text(config_yaml)
    return config_path


# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: pathlib.Path) -> None:
    """Change the working directory to a temporary path to help isolate the test suite."""
    os.chdir(tmp_path)


@pytest.fixture(autouse=True)
def clean_environment() -> Callable[[None], None]:
    """Discard environment variables prefixed with `SERVO_` or `OPSANI`.

    This fixture helps ensure test suite isolation from local development
    configuration (often set via a .env file).

    Returns:
        A callable that can be used to clean the environment on-demand.
    """
    def _clean_environment():
        for key, value in os.environ.copy().items():
            if key.startswith("SERVO_") or key.startswith("OPSANI_"):
                os.environ.pop(key)

    _clean_environment()
    return _clean_environment


@pytest.fixture
def random_string() -> str:
    """Return a random string of characters."""
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(32))


@pytest.fixture
async def kubeconfig(request) -> str:
    """Return the path to a kubeconfig file to use when running integraion tests.

    To avoid inadvertantly interacting with clusters not explicitly configured
    for development, we suppress the kubetest default of using ~/.kube/kubeconfig.
    """
    config_opt = request.session.config.getoption('kube_config') or "tests/kubeconfig"
    path = pathlib.Path(config_opt).expanduser()
    config_path = (
        path if path.is_absolute()
        else request.session.config.rootpath.joinpath(path)
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"kubeconfig file not found: configure a test cluster and create kubeconfig at: {config_path}"
        )

    return str(config_path)


@pytest.fixture
async def kubernetes_asyncio_config(request, kubeconfig: str, kubecontext: Optional[str]) -> None:
    """Initialize the kubernetes_asyncio config module with the kubeconfig fixture path."""
    import logging

    import kubernetes_asyncio.config

    if request.session.config.getoption('in_cluster') or os.getenv("KUBERNETES_SERVICE_HOST"):
        kubernetes_asyncio.config.load_incluster_config()
    else:
        kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        if kubeconfig:
            kubeconfig_path = pathlib.Path(os.path.expanduser(kubeconfig))
            await kubernetes_asyncio.config.load_kube_config(
                config_file=os.path.expandvars(kubeconfig_path),
                context=kubecontext,
            )
        else:
            log = logging.getLogger('kubetest')
            log.error(
                'unable to interact with cluster: kube fixture used without kube config '
                'set. the config may be set with the flags --kube-config or --in-cluster or by'
                'an env var KUBECONFIG or custom kubeconfig fixture definition.'
            )
            raise FileNotFoundError(
                f"kubeconfig file not found: configure a test cluster and add kubeconfig: {kubeconfig}"
            )


@pytest.fixture()
async def subprocess() -> tests.helpers.Subprocess:
    """Return an asynchronous executor for testing subprocesses."""
    return tests.helpers.Subprocess()

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


### Kind
# Depends on subprocessx. Libraries in the key of x. kowabunga. kareem. krush. ktest

@pytest.fixture
async def kind(request, subprocess, kubeconfig: str, kube_context: str) -> str:
    """Run tests within a local kind cluster.

    The cluster name is determined using the parametrized `kind_cluster` marker
    or else uses "default".
    """
    cluster = "pytest-k8s"
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
        # TODO: add an option to not tear down the cluster
        if not os.getenv("GITHUB_ACTIONS"):
            exit_code, _, _ = await subprocess(f"kind delete cluster --name {cluster} --kubeconfig {kubeconfig}", print_output=True)
            if exit_code != 0:
                raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")
            await subprocess(f"kubectl config --kubeconfig {kubeconfig} use-context {kube_context}", print_output=True)

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

@pytest.fixture()
def random_duration() -> servo.Duration:
    seconds = random.randrange(30, 600)
    return servo.Duration(seconds=seconds)

@pytest.fixture
def fastapi_app() -> fastapi.FastAPI:
    """Return a FastAPI instance for testing in the current scope.

    To utilize the FakeAPI fixtures, define a module local FastAPI object
    that implements the API interface that you want to work with and return it
    from an override implementation of the `fastapi_app` fixture.

    The default implementation is abstract and raises a NotImplementedError.

    To interact from the FastAPI app within your tests, invoke the `fakeapi_url`
    fixture to obtain the base URL for a running instance of your fastapi app.
    """
    raise NotImplementedError(f"incomplete fixture implementation: build a FastAPI fixture modeling the system you want to fake")

@pytest.fixture
async def fakeapi_url(fastapi_app: fastapi.FastAPI, unused_tcp_port: int) -> AsyncGenerator[str, None]:
    """Run a FakeAPI server as a pytest fixture and yield the base URL for accessing it."""
    server = tests.helpers.FakeAPI(app=fastapi_app, port=unused_tcp_port)
    await server.start()
    yield server.base_url
    await server.stop()

@pytest.fixture
async def fakeapi_client(fakeapi_url: str) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an httpx client configured to interact with a FakeAPI server."""
    async with httpx.AsyncClient(
        headers={
            'Content-Type': 'application/json',
        },
        base_url=fakeapi_url,
    ) as client:
        yield client



######

@pytest.fixture()
def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    optimizer = servo.Optimizer(
        id="dev.opsani.com/blake-ignite",
        token="bfcf94a6e302222eed3c73a5594badcfd53fef4b6d6a703ed32604",

    )
    assembly_ = servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly_


@pytest.fixture
def assembly_runner(assembly: servo.Assembly) -> servo.runner.AssemblyRunner:
    """Return an unstarted assembly runner."""
    return servo.runner.AssemblyRunner(assembly)

@pytest.fixture
async def servo_runner(assembly: servo.Assembly) -> servo.runner.ServoRunner:
    """Return an unstarted servo runner."""
    return servo.runner.ServoRunner(assembly.servos[0])

@pytest.fixture
def fastapi_app() -> fastapi.FastAPI:
    return tests.fake.api

## Kubernetes Port Forwarding

ForwardingTarget = Union[
    str,
    kubetest.objects.Pod,
    servo.connectors.kubernetes.Pod,
    kubetest.objects.Deployment,
    servo.connectors.kubernetes.Deployment,
    kubetest.objects.Service,
    servo.connectors.kubernetes.Service,
]

@contextlib.asynccontextmanager
async def kubectl_ports_forwarded(
    target: ForwardingTarget,
    *ports: List[Tuple[int, int]],
    kubeconfig: str,
    context: Optional[str],
    namespace: str,
) -> AsyncIterator[Union[str, Dict[int, str]]]:
    """An async context manager that establishes a port-forward to remote targets in a Kubernetes cluster and yields URLs for connecting to them.

    When a single port is forwarded, a single URL is yielded. When multiple ports are forwarded, a mapping is yielded from
    the remote target port to a URL for reaching it.

    Args:
        target: The deployment, pod, or service to open a forward to.
        ports: A list of integer tuples where the first item is the local port and the second is the remote.
        kubeconfig: Path to the kubeconfig file to use when establishing the port forward.
        namespace: The namespace that the target is running in.

    Returns:
        A URL if a single port was forwarded else a mapping of destination ports to URLs.

    The `target` argument accepts the following syntaxes:
        - [POD NAME]
        - pod/[NAME]
        - deployment/[NAME]
        - deploy/[NAME]
        - service/[NAME]
        - svc/[NAME]
    """
    try:
        def _identifier_for_target(target: ForwardingTarget) -> str:
            if isinstance(target, str):
                return target
            elif isinstance(target, (kubetest.objects.Pod, servo.connectors.kubernetes.Pod)):
                return f"pod/{target.name}"
            elif isinstance(target, (kubetest.objects.Deployment, servo.connectors.kubernetes.Deployment)):
                return f"deployment/{target.name}"
            elif isinstance(target, (kubetest.objects.Service, servo.connectors.kubernetes.Service)):
                return f"service/{target.name}"            
            else:
                raise TypeError(f"unknown target: {repr(target)}")

        identifier = _identifier_for_target(target)
        ports_arg = " ".join(list(map(lambda pair: f"{pair[0]}:{pair[1]}", ports)))
        context_arg = f"--context {context}" if context else ""
        event = asyncio.Event()
        task = asyncio.create_task(
            tests.helpers.Subprocess.shell(
                f"kubectl --kubeconfig={kubeconfig} {context_arg} port-forward --namespace {namespace} {identifier} {ports_arg}",
                event=event,
                print_output=True
        ))

        await event.wait()

        # Check if the sockets are open
        for local_port, _ in ports:
            a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if a_socket.connect_ex(("localhost", local_port)) != 0:
                raise RuntimeError(f"port forwarding failed")

        if len(ports) == 1:
            url = f"http://localhost:{ports[0][0]}"
            yield url
        else:
            # Build a mapping of from target ports to the forwarded URL
            ports_to_urls = dict(map(lambda p: (p[1], f"http://localhost:{p[0]}"), ports))
            yield ports_to_urls
    finally:
        task.cancel()

        # Cancel outstanding tasks
        tasks = [t for t in asyncio.all_tasks() if t not in [asyncio.current_task()]]
        [task.cancel() for task in tasks]

        await asyncio.gather(*tasks, return_exceptions=True)

# TODO: This one takes a dict of servi
# @pytest.fixture()
# async def kube_port_forwards(
#     kube,
#     unused_tcp_port_factory: Callable[[], int],
#     kubeconfig,
# ) -> Callable[[ForwardingTarget, List[int]], AsyncIterator[str]]:
#     """A pytest fixture that returns an async generator for port forwarding to a remote kubernetes deployment, pod, or service."""
#     def _port_forwarder(target: ForwardingTarget, *remote_ports: int):
#         kube.wait_for_registered(timeout=10)
#         return kubectl_ports_forwarded(
#             target,
#             unused_tcp_port,
#             remote_port,
#             namespace=kube.namespace,
#             kubeconfig=kubeconfig
#         )

#     return _port_forwarder

# @pytest.fixture()
# async def kube_port_forward(
#     kube,
#     unused_tcp_port: int,
#     kubeconfig,
# ) -> Callable[[ForwardingTarget, List[int]], AsyncIterator[str]]:
#     """A pytest fixture that returns an async generator for port forwarding to a remote kubernetes deployment, pod, or service."""
#     def _port_forwarder(target: ForwardingTarget, *remote_ports: int):
#         kube.wait_for_registered(timeout=10)
#         ports = list(map(lambda port: (unused_tcp_port_factory(), port), remote_ports))
#         return kubectl_ports_forwarded(
#             target,
#             *ports,
#             namespace=kube.namespace,
#             kubeconfig=kubeconfig
#         )

#     return _port_forwarder

@pytest.fixture()
async def kube_port_forward(
    kube,
    unused_tcp_port_factory: Callable[[], int],
    kubeconfig,
    kube_context: Optional[str],
) -> Callable[[ForwardingTarget, List[int]], AsyncIterator[str]]:
    """A pytest fixture that returns an async generator for port forwarding to a remote kubernetes deployment, pod, or service."""
    def _port_forwarder(target: ForwardingTarget, *remote_ports: int):
        kube.wait_for_registered(timeout=10)
        ports = list(map(lambda port: (unused_tcp_port_factory(), port), remote_ports))
        return kubectl_ports_forwarded(
            target,
            *ports,
            namespace=kube.namespace,
            kubeconfig=kubeconfig,
            context=kube_context
        )

    return _port_forwarder

@pytest.fixture
def pod_loader(kube: kubetest.client.TestClient) -> Callable[[str], kubetest.objects.Pod]:
    """A pytest fixture that returns a callable for loading a kubernetes pod reference."""
    def _pod_loader(deployment: str) -> kubetest.objects.Pod:
        kube.wait_for_registered(timeout=10)

        deployments = kube.get_deployments()
        prometheus = deployments.get(deployment)
        assert prometheus is not None

        pods = prometheus.get_pods()
        assert len(pods) == 1, "prometheus should deploy with one replica"

        pod = pods[0]
        pod.wait_until_ready(timeout=30)

        return pod

    return _pod_loader
