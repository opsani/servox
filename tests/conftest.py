import asyncio
import builtins
import contextlib
import enum
import json
import os
import pathlib
import random
import socket
import string
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import chevron
import devtools
import fastapi
import httpx
import kubetest
import kubetest.client
import kubetest.objects
import loguru
import pytest
import typer.testing
import uvloop
import yaml

import servo
import servo.runner
import servo.cli
import servo.connectors.kubernetes
import tests.helpers

# Add the devtools debug() function globally in tests
from devtools import debug

builtins.debug = debug

# Render all manifests as Mustache templates by default
kubetest.manifest.__render__ = chevron.render


def pytest_report_header(config) -> str:
    for connector in servo.connector.ConnectorLoader().load():
        servo.logger.debug(f"Loaded {connector.__qualname__}")

    names = list(
        map(
            lambda c: f"{c.__default_name__}-{c.version}",
            servo.Assembly.all_connector_types(),
        )
    )
    return "servo connectors: " + ", ".join(names)


@pytest.fixture()
def event_loop_policy():
    return uvloop.EventLoopPolicy()


# FIXME: Below logic causes "OSError: [Errno 24] Too many open files". Fix if really needed
# @pytest.fixture
# def event_loop_policy(request) -> str:
#     """Return the active event loop policy for the test.

#     Valid values are "default" and "uvloop".

#     The default implementation defers to the `event_loop_policy` marker
#     when it is set and otherwise selects a default policy based on the
#     characteristics of the test being run.
#     """
#     marker = request.node.get_closest_marker("event_loop_policy")
#     if marker:
#         assert (
#             len(marker.args) == 1
#         ), f"event_loop_policy marker accepts a single argument but received: {repr(marker.args)}"
#         event_loop_policy = marker.args[0]
#     else:
#         event_loop_policy = "uvloop"

#     valid_policies = ("default", "uvloop")
#     assert (
#         event_loop_policy in valid_policies
#     ), f'invalid event_loop_policy marker: "{event_loop_policy}" is not in {repr(valid_policies)}'

#     return event_loop_policy


# @pytest.fixture
# def event_loop(event_loop_policy: str) -> Iterator[asyncio.AbstractEventLoop]:
#     """Yield an instance of the event loop for each test case.

#     The effective event loop policy is determined by the `event_loop_policy` fixture.
#     """
#     if event_loop_policy == "default":
#         asyncio.set_event_loop_policy(None)
#     elif event_loop_policy == "uvloop":
#         uvloop.install()
#     else:
#         raise ValueError(f'invalid event loop policy: "{event_loop_policy}"')

#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     yield loop
#     loop.close()


def pytest_addoption(parser) -> None:
    """Add pytest options for running tests of various types."""
    parser.addoption(
        "-I",
        "--integration",
        action="store_true",
        default=False,
        help="enable integration tests",
    )
    parser.addoption(
        "-S",
        "--system",
        action="store_true",
        default=False,
        help="enable system tests",
    )
    parser.addoption(
        "-T",
        "--type",
        action="store",
        metavar="TYPE",
        help="only run tests of the type TYPE.",
    )


class TestType(enum.StrEnum):
    unit = "unit"
    integration = "integration"
    system = "system"

    @classmethod
    def names(cls) -> List[str]:
        return cls.__members__.keys()


class Environment(enum.StrEnum):
    docker = "Docker"
    compose = "Docker Compose"
    kind = "Kind"
    minikube = "Minikube"
    kubernetes = "kubernetes"
    eks = "EKS"
    gke = "GKE"
    aks = "AKS"
    ecs = "ECS"

    @classmethod
    def ids(cls) -> List[str]:
        return cls.__members__.keys()

    @property
    def parents(self) -> List["Environment"]:
        if self in {Environment.compose, Environment.kind, Environment.minikube}:
            return [Environment.docker]
        elif self in {Environment.eks, Environment.gke, Environment.aks}:
            return [Environment.kubernetes]
        else:
            return []


UNIT_INI = (
    "unit: marks the test as a unit test. Unit tests are fast, highly localized, and "
    "have no external dependencies. Tests without an explicit type mark are considered "
    "unit tests for convenience."
)
INTEGRATION_INI = (
    "integration: marks the test as an integration test. Integration tests have external "
    "dependencies that can be orchestrated by the test suite. They are much slower than "
    "unit tests but provide interaction with external components. "
)
SYSTEM_INI = (
    "system: marks the test as system test. System tests are run in specific environments "
    "and execute functionality end to end. They are very slow and resource intensive, but "
    "are capable of verifying that the product meets requirements as specified from a user "
    "perspective."
)
EVENT_LOOP_POLICY_INI = (
    "event_loop_policy: marks async tests to run under a parametrized asyncio "
    "runloop policy. There are two event loop policies available: default and uvloop. "
    "The `default` policy is the standard event loop behavior provided with asyncio. "
    "The `uvloop` policy is a high performance event loop. Certain tests may fail "
    "due to interactions between uvloop and the pytest output capture mechanism. "
    "The `event_loop_policy` fixture determines what event loop policy is registered "
    "at runtime and respects the value of this marker."
)
MINIKUBE_PROFILE_INI = (
    "minikube_profile: marks tests using minikube to run under the profile specified"
    "in the first marker argument. Eg. pytest.mark.minikube_profile.with_args(MINIKUBE_PROFILE)"
)
ROLLOUT_MANIFEST_INI = (
    "rollout_manifest: mark tests using argo rollouts to apply the manifest from"
    "the specified path in the first marker argument. Eg."
    '@pytest.mark.rollout_manifest.with_args("tests/manifests/opsani_dev/argo_rollouts/rollout.yaml")'
)


def pytest_configure(config) -> None:
    """Register custom markers for use in the test suite."""
    config.addinivalue_line("markers", UNIT_INI)
    config.addinivalue_line("markers", INTEGRATION_INI)
    config.addinivalue_line("markers", SYSTEM_INI)
    config.addinivalue_line("markers", EVENT_LOOP_POLICY_INI)
    config.addinivalue_line("markers", MINIKUBE_PROFILE_INI)
    config.addinivalue_line("markers", ROLLOUT_MANIFEST_INI)

    # Add generic description for all environments
    for key, value in Environment.__members__.items():
        config.addinivalue_line(
            "markers", f"{key}: marks the test as runnable on {value}."
        )


def pytest_runtest_setup(item):
    # NOTE: If integration is selected but theres no kubeconfig, fail them clearly
    type_mark, _ = gather_marks_for_item(item)
    assert type_mark, "test should have a type"
    if type_mark.name in {TestType.integration, TestType.system}:
        config_path = kubeconfig_path_from_config(item.config)
        if not config_path.exists():
            pytest.fail(
                f"Cannot run {type_mark.name} tests: "
                f"kubeconfig file not found: configure a test cluster and create kubeconfig at: {config_path}\n "
                "Hint: See README.md and run `make test-kubeconfig`"
            )


def selected_types_for_item(item) -> Optional[List[TestType]]:
    type_option = item.config.getoption("-T")
    if not type_option:
        return None

    matches = list(filter(lambda t: t.startswith(type_option), TestType.names()))
    if matches:
        if len(matches) > 1:
            item.warn(
                pytest.PytestWarning(
                    f"--type argument '{type_option}' matched multiple types ({matches})"
                )
            )
    else:
        type_names = ", ".join(list(TestType.names()))
        item.warn(
            pytest.PytestWarning(
                f"--type argument '{type_option}' does not match any test type: {type_names}"
            )
        )

    return matches


def gather_marks_for_item(item) -> tuple:
    type_mark, env_marks = None, []
    for mark in item.iter_markers():
        if type_mark is None and mark.name in TestType.names():
            # NOTE: Only the closest marker is relevant
            type_mark = mark
        elif mark.name in Environment.ids():
            env_marks.append(mark)

    return (type_mark, env_marks)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify the discovered pytest nodes to configure default markers.

    This methods sets asyncio as the async backend and skips
    integration and system tests unless opted in.
    """

    selected_items = []
    deselected_items = []
    for item in items:
        # Consider any unmarked item as a unit test
        type_mark, env_marks = gather_marks_for_item(item)
        if not type_mark:
            type_mark = pytest.mark.unit
            item.add_marker(type_mark)

        # Add missing parent marks for environment selectors
        if env_marks:
            for name, member in Environment.__members__.items():
                if next(filter(lambda m: m.name == name, env_marks), None):
                    for env in member.parents:
                        mark = getattr(pytest.mark, env.name)
                        item.add_marker(mark)
                        env_marks.append(mark)

        # Handle CLI switches
        selected_types = selected_types_for_item(item)
        if selected_types is not None:
            if type_mark.name not in selected_types:
                deselected_items.append(item)
        else:
            if (
                type_mark.name == TestType.integration
                and not config.getoption("--integration")
            ) or (
                type_mark.name == TestType.system and not config.getoption("--system")
            ):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{type_mark.name} tests not enabled. Run with --{type_mark.name} to enable"
                    )
                )

        if item not in deselected_items:
            selected_items.append(item)

    # Deselect any items accumulated. The items input array must be mutated in place
    items[:] = selected_items
    config.hook.pytest_deselected(items=deselected_items)


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
def optimizer() -> servo.OpsaniOptimizer:
    """Return a generated optimizer instance."""
    return servo.OpsaniOptimizer(id="dev.opsani.com/servox", token="123456789")


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
    settings = servo.BaseConfiguration()
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    config = {
        "connectors": ["measure", "adjust"],
        "measure": measure_config_json,
        "adjust": {},
    }
    config = yaml.dump(config)
    config_path.write_text(config)
    return config_path


@pytest.fixture()
def stub_multiservo_yaml(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a servo config file set up for multi-servo execution."""
    config_path: pathlib.Path = tmp_path / "servo.yaml"
    settings = tests.helpers.BaseConfiguration()
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    optimizer1 = servo.OpsaniOptimizer(
        id="dev.opsani.com/multi-servox-1", token="123456789"
    )
    optimizer1_config_json = json.loads(
        optimizer1.json(
            by_alias=True,
        )
    )
    config1 = {
        "optimizer": optimizer1_config_json,
        "connectors": ["measure", "adjust"],
        "measure": measure_config_json,
        "adjust": {},
    }
    optimizer2 = servo.OpsaniOptimizer(
        id="dev.opsani.com/multi-servox-2", token="987654321"
    )
    optimizer2_config_json = json.loads(
        optimizer2.json(
            by_alias=True,
        )
    )
    config2 = {
        "optimizer": optimizer2_config_json,
        "connectors": ["measure", "adjust"],
        "measure": measure_config_json,
        "adjust": {},
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
def captured_logs() -> Generator[None, list["loguru.Message"], None]:
    messages = []
    temp_sink_id = servo.logger.add(lambda m: messages.append(m), level=0)
    yield messages
    servo.logger.remove(temp_sink_id)


@pytest.fixture
def rootpath(pytestconfig) -> pathlib.Path:
    return pytestconfig.rootpath


def kubeconfig_path_from_config(config) -> Union[pathlib.Path, pathlib.PurePath]:
    config_opt = config.getoption("kube_config") or "tests/kubeconfig"
    path = pathlib.Path(config_opt).expanduser()
    config_path = path if path.is_absolute() else config.rootpath.joinpath(path)
    return config_path


@pytest.fixture
def kubeconfig(request) -> Union[pathlib.Path, pathlib.PurePath]:
    """Return the path to a kubeconfig file to use when running tests. (an empty file will be created if none exists)

    To avoid inadvertantly interacting with clusters not explicitly configured
    for development, we suppress the kubetest default of using ~/.kube/kubeconfig.
    """
    retval = kubeconfig_path_from_config(request.session.config)
    if not retval.exists():
        retval.write_text("")
    return retval


@pytest.fixture
async def kubernetes_asyncio_config(
    request, kubeconfig: str, kubecontext: Optional[str]
) -> None:
    """Initialize the kubernetes_asyncio config module with the kubeconfig fixture path."""
    import logging

    import kubernetes_asyncio.config

    if request.session.config.getoption("in_cluster"):
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
            log = logging.getLogger("kubetest")
            log.error(
                "unable to interact with cluster: kube fixture used without kube config "
                "set. the config may be set with the flags --kube-config or --in-cluster or by"
                "an env var KUBECONFIG or custom kubeconfig fixture definition."
            )
            raise FileNotFoundError(
                f"kubeconfig file not found: configure a test cluster and add kubeconfig: {kubeconfig}"
            )


@pytest.fixture
async def minikube(
    request: pytest.FixtureRequest,
    subprocess: tests.helpers.Subprocess,
    kubeconfig: pathlib.Path,
) -> AsyncGenerator[str, None]:
    """Run tests within a local minikube profile.

    The profile name is determined using the parametrized `minikube_profile` marker
    or else uses "default".
    """
    marker = request.node.get_closest_marker("minikube_profile")
    addons = ""
    if marker:
        assert (
            len(marker.args) == 1
        ), f"minikube_profile marker accepts a single argument but received: {repr(marker.args)}"
        profile = marker.args[0]
        if profile == "metrics-server":
            addons = "--addons metrics-server "
    else:
        profile = "servox"

    # Start minikube and configure environment
    exit_code, _, stderr = await subprocess(
        f"KUBECONFIG={kubeconfig} minikube start {addons}-p {profile} --interactive=false --wait=true",
        print_output=True,
    )
    if exit_code != 0:
        if exit_code == 50 and any(
            "Exiting due to DRV_CP_ENDPOINT" in line for line in stderr
        ):
            # https://github.com/kubernetes/minikube/issues/10357
            # NOTE no point in even trying to recover from this due to asyncio xdist parallelization coordination hell
            pytest.xfail("Minikube failed start")

        if exit_code == 80 and any(
            "Exiting due to GUEST_START" in line for line in stderr
        ):
            # https://github.com/kubernetes/minikube/issues/13621
            pytest.xfail("Minikube failed start (CA)")

        if exit_code == 127 and any("minikube: not found" in line for line in stderr):
            pytest.xfail("Minikube not installed")

        raise RuntimeError(
            f"failed running minikube: exited with status code {exit_code}: {stderr}"
        )

    # Yield the profile name
    try:
        yield profile

    finally:
        # TODO: add an option to not tear down the cluster
        # Skip teardown on parallelized runs due to session scope being non-functional with xdist and asyncio
        # https://github.com/pytest-dev/pytest-asyncio/issues/75
        # https://github.com/pytest-dev/pytest-xdist/issues/271
        if (
            not hasattr(request.config, "workerinput")
            or request.config.workerinput["workercount"] <= 1
        ):
            exit_code, _, stderr = await subprocess(
                f"KUBECONFIG={kubeconfig} minikube stop -p {profile}", print_output=True
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"failed to stop minikube: exited with status code {exit_code}: {stderr}"
                )


@pytest.fixture()
async def subprocess() -> tests.helpers.Subprocess:
    """Return an asynchronous executor for testing subprocesses."""
    return tests.helpers.Subprocess()


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
    raise NotImplementedError(
        f"incomplete fixture implementation: build a FastAPI fixture modeling the system you want to fake"
    )


@pytest.fixture
async def fakeapi_url(
    fastapi_app: fastapi.FastAPI, unused_tcp_port: int
) -> AsyncGenerator[str, None]:
    """Run a FakeAPI server as a pytest fixture and yield the base URL for accessing it."""
    async with asyncio.TaskGroup() as tg:
        server = tests.helpers.FakeAPI(
            app=fastapi_app, port=unused_tcp_port, task_group=tg
        )
        await server.start()
        yield server.base_url
        await server.stop()


@pytest.fixture
async def fakeapi_client(fakeapi_url: str) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an httpx client configured to interact with a FakeAPI server."""
    async with httpx.AsyncClient(
        headers={
            "Content-Type": "application/json",
        },
        base_url=fakeapi_url,
    ) as client:
        yield client


######


@pytest.fixture()
async def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    assembly_ = await servo.assembly.Assembly.assemble(config_file=servo_yaml)
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
    kubetest.objects.Deployment,
    kubetest.objects.Service,
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

    def _identifier_for_target(target: ForwardingTarget) -> str:
        if isinstance(target, str):
            return target
        elif isinstance(target, kubetest.objects.Pod):
            return f"pod/{target.name}"
        elif isinstance(target, kubetest.objects.Deployment):
            return f"deployment/{target.name}"
        elif isinstance(target, kubetest.objects.Service):
            return f"service/{target.name}"
        else:
            raise TypeError(f"unknown target: {repr(target)}")

    identifier = _identifier_for_target(target)
    ports_arg = " ".join(list(map(lambda pair: f"{pair[0]}:{pair[1]}", ports)))
    context_arg = f"--context {context}" if context else ""
    event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        task = tg.create_task(
            tests.helpers.Subprocess.shell(
                f"kubectl --kubeconfig={kubeconfig} {context_arg} port-forward --namespace {namespace} {identifier} {ports_arg}",
                event=event,
                print_output=True,
            )
        )

        await event.wait()

        # Check the sockets can be connected to
        # TODO/FIXME add fault tolerance for error upgrading connection: unable to upgrade connection: pod does not exist
        for local_port, _ in ports:
            a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if (h_errno := a_socket.connect_ex(("localhost", local_port))) != 0:
                if task.done():
                    debug(task.result())
                raise RuntimeError(
                    f"port forwarding failed: port {local_port} connect failed (errno {h_errno})"
                )

        if len(ports) == 1:
            url = f"http://localhost:{ports[0][0]}"
            yield url
        else:
            # Build a mapping of from target ports to the forwarded URL
            ports_to_urls = dict(
                map(lambda p: (p[1], f"http://localhost:{p[0]}"), ports)
            )
            yield ports_to_urls

        task.cancel()


@pytest.fixture()
async def kube_port_forward(
    kube: kubetest.client.TestClient,
    unused_tcp_port_factory: Callable[[], int],
    kubeconfig,
    kubecontext: Optional[str],
) -> Callable[[ForwardingTarget, List[int]], AsyncIterator[str]]:
    """A pytest fixture that returns an async generator for port forwarding to a remote kubernetes deployment, pod, or service."""

    def _port_forwarder(target: ForwardingTarget, *remote_ports: int):
        kube.wait_for_registered()
        ports = list(map(lambda port: (unused_tcp_port_factory(), port), remote_ports))
        return kubectl_ports_forwarded(
            target,
            *ports,
            namespace=kube.namespace,
            kubeconfig=kubeconfig,
            context=kubecontext,
        )

    return _port_forwarder


@pytest.fixture
def pod_loader(
    kube: kubetest.client.TestClient,
) -> Callable[[str], kubetest.objects.Pod]:
    """A pytest fixture that returns a callable for loading a kubernetes pod reference."""

    def _pod_loader(deployment: str) -> kubetest.objects.Pod:
        kube.wait_for_registered()

        deployments = kube.get_deployments()
        prometheus = deployments.get(deployment)
        assert prometheus is not None

        pods = prometheus.get_pods()
        assert len(pods) == 1, "prometheus should deploy with one replica"

        pod = pods[0]
        pod.wait_until_ready(timeout=30)

        return pod

    return _pod_loader


@pytest.fixture()
def kubetest_teardown(
    kube: kubetest.client.TestClient,
) -> Generator[kubetest.client.TestClient, None, None]:
    """
    Kubetest's default teardown assumes the namespace was created and applied manifests will be cleaned up when the ns is deleted

    This teardown method only needs to be used if a pre-existing namespace was used
    """
    # this is teardown only, yield for test run
    yield
    for obj in kube.pre_registered:
        obj.delete()


# NOTE: kubetest does not support CRD or CR objects, rollout fixtures utilize kubectl for needed setup
@pytest.fixture
async def verify_rollout_crd(subprocess, kubeconfig):
    """Verify applied CRDs are the latest version"""

    rollouts_version_cmd = [
        "kubectl",
        f"--kubeconfig={kubeconfig}",
        "get",
        "deployment",
        "argo-rollouts",
        "-n",
        "argo-rollouts",
        "-o",
        "jsonpath='{.spec.template.spec.containers[0].image}'",
    ]
    exit_code, stdout, stderr = await subprocess(
        " ".join(rollouts_version_cmd), print_output=True, timeout=None
    )
    assert (
        exit_code == 0
    ), f"Unable to get argo-rollouts controller Deployment: {stderr}"

    async with httpx.AsyncClient() as client:
        latest_version = (
            (
                await client.get(
                    "https://api.github.com/repos/argoproj/argo-rollouts/releases/latest"
                )
            )
            .json()
            .get("tag_name")
        )

    assert (
        latest_version in stdout[0]
    ), f"Installed rollout controller image ({stdout[0]}) out of date (latest={latest_version})"


@pytest.fixture
async def manage_rollout(
    request, kube, rootpath, kubeconfig, subprocess, verify_rollout_crd
):
    """
    Apply the manifest of the target rollout being tested against

    The applied manifest is determined using the parametrized `rollout_manifest` marker
    or else uses "tests/manifests/argo_rollouts/fiber-http-opsani-dev.yaml".
    """
    marker = request.node.get_closest_marker("rollout_manifest")
    if marker:
        assert (
            len(marker.args) == 1
        ), f"rollout_manifest marker accepts a single argument but received: {repr(marker.args)}"
        manifest = marker.args[0]
    else:
        manifest = "tests/manifests/argo_rollouts/fiber-http-opsani-dev.yaml"

    rollout_cmd = [
        "kubectl",
        f"--kubeconfig={kubeconfig}",
        "apply",
        "-n",
        kube.namespace,
        "-f",
        str(rootpath / manifest),
    ]
    exit_code, _, stderr = await subprocess(
        " ".join(rollout_cmd), print_output=True, timeout=None
    )
    assert exit_code == 0, f"argo-rollouts CR manifest apply failed: {stderr}"

    wait_cmd = [
        "kubectl",
        f"--kubeconfig={kubeconfig}",
        "wait",
        "--for=condition=completed",
        "--timeout=60s",
        "-n",
        kube.namespace,
        "rollout",
        "fiber-http",
    ]
    exit_code, _, stderr = await subprocess(
        " ".join(wait_cmd), print_output=True, timeout=None
    )
    assert (
        exit_code == 0
    ), f"argo-rollouts CR manifest wait for available failed: {stderr}"

    # NOTE: it is assumed kubetest will handle the needed teardown of the rollout
