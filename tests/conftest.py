import asyncio
import builtins
import json
import os
import random
import string
import pathlib
from typing import AsyncGenerator, Iterator, Optional

import devtools
import fastapi
import httpx
import pytest
import yaml
import uvloop
import typer.testing

import servo
import servo.cli
import tests.helpers

# Add the devtools debug() function globally in tests
builtins.debug = devtools.debug

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
    """Add pytest options for enabling execution of integration and system tests."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--system", action="store_true", default=False, help="run system tests"
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
def run_in_clean_environment() -> None:
    """Discard environment variables prefixed with `SERVO_` or `OPSANI`.
    
    This fixture helps ensure test suite isolation from local development configuration.
    """
    for key, value in os.environ.copy().items():
        if key.startswith("SERVO_") or key.startswith("OPSANI_"):
            os.environ.pop(key)


@pytest.fixture
def random_string() -> str:
    """Return a random string of characters."""
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(32))


@pytest.fixture
async def kubeconfig() -> str:
    """Return the path to a kubeconfig file to use when running integraion tests."""
    config_path = pathlib.Path(__file__).parents[0] / "kubeconfig"
    if not config_path.exists():
        raise FileNotFoundError(
            f"kubeconfig file not found: configure a test cluster and create kubeconfig at: {config_path}"
        )

    return str(config_path)

@pytest.fixture
def kube_context(request) -> Optional[str]:
    """Return the context to be used within the kubeconfig file or None to use the default."""
    return request.session.config.getoption('kube_context')

@pytest.fixture
async def kubernetes_asyncio_config(request, kubeconfig: str, kube_context: Optional[str]) -> None:
    """Initialize the kubernetes_asyncio config module with the kubeconfig fixture path."""
    import kubernetes_asyncio.config
    import logging
    
    if request.session.config.getoption('in_cluster') or os.getenv("KUBERNETES_SERVICE_HOST"):
        kubernetes_asyncio.config.load_incluster_config()
    else:
        kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        if kubeconfig:
            await kubernetes_asyncio.config.load_kube_config(
                config_file=os.path.expandvars(os.path.expanduser(kubeconfig)),
                context=kube_context,
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


@pytest.fixture()
async def servo_image() -> str:
    """Asynchronously build a Docker image from the current working copy and return its tag."""
    return await tests.helpers.build_docker_image()

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
    exit_code, _, _ = await subprocess(f"minikube start -p {profile} --interactive=false --keep-context=true --wait=true")
    if exit_code != 0:
        raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")
    
    # Yield the profile name
    try:
        yield profile
    
    finally:
        exit_code, _, _ = await subprocess(f"minikube stop -p {profile}")
        if exit_code != 0:
            raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")

@pytest.fixture()
async def minikube_servo_image(minikube: str, servo_image: str, subprocess) -> str:
    """Asynchronously build a Docker image from the current working copy and cache it into the minikube repository."""
    exit_code, _, _ = await subprocess(f"minikube cache add -p {minikube} {servo_image}")
    if exit_code != 0:
        raise RuntimeError(f"failed running minikube: exited with status code {exit_code}")
    
    yield servo_image
    # return await tests.helpers.build_docker_image(preamble="eval $(minikube -p minikube docker-env)")

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
async def fakeapi_client(fakeapi_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Yield an httpx client configured to interact with a FakeAPI server."""
    async with httpx.AsyncClient(
        headers={
            'Content-Type': 'application/json',
        },
        base_url=fakeapi_url,
    ) as client:
        yield client
