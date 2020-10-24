import json
import os
import random
import string
from pathlib import Path
from typing import Iterator, Optional

import pytest
import yaml
from typer.testing import CliRunner

# Add the devtools debug() function globally in tests
try:
    import builtins

    from devtools import debug

    builtins.debug = debug
except ImportError:
    pass

from kubernetes_asyncio import config as kubernetes_asyncio_config

from servo.cli import ServoCLI
from servo.configuration import Optimizer
from tests.test_helpers import StubBaseConfiguration, SubprocessTestHelper


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--system", action="store_true", default=False, help="run system tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks integration tests with outside dependencies"
    )
    config.addinivalue_line(
        "markers", "system: marks system tests with end to end dependencies"
    )


def pytest_collection_modifyitems(config, items):
    skip_itegration = pytest.mark.skip(
        reason="add --integration option to run integration tests"
    )
    skip_system = pytest.mark.skip(reason="add --system to run system tests")

    for item in items:
        # Set asyncio as a default marker across the suite
        item.add_marker("asyncio")

        # Skip slow/sensitive integration & system tests by default
        if "integration" in item.keywords and not config.getoption("--integration"):
            item.add_marker(skip_itegration)
        if "system" in item.keywords and not config.getoption("--system"):
            item.add_marker(skip_system)


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def servo_cli() -> ServoCLI:
    return ServoCLI()


@pytest.fixture()
def optimizer_env() -> Iterator[None]:
    os.environ.update(
        {"OPSANI_OPTIMIZER": "dev.opsani.com/servox", "OPSANI_TOKEN": "123456789"}
    )
    yield
    os.environ.pop("OPSANI_OPTIMIZER", None)
    os.environ.pop("OPSANI_TOKEN", None)


@pytest.fixture()
def optimizer() -> Optimizer:
    return Optimizer(id="dev.opsani.com/servox", token="123456789")


@pytest.fixture()
def servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    config_path.touch()
    return config_path


@pytest.fixture()
def stub_servo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    settings = StubBaseConfiguration(name="stub")
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    config = {"connectors": ["measure"], "measure": measure_config_json}
    config = yaml.dump(config)
    config_path.write_text(config)
    return config_path

@pytest.fixture()
def stub_multiservo_yaml(tmp_path: Path) -> Path:
    config_path: Path = tmp_path / "servo.yaml"
    settings = StubBaseConfiguration(name="stub")
    measure_config_json = json.loads(
        json.dumps(
            settings.dict(
                by_alias=True,
            )
        )
    )
    optimizer1 = Optimizer(id="dev.opsani.com/multi-servox-1", token="123456789")
    optimizer1_config_json = json.loads(
        json.dumps(
            optimizer1.dict(
                by_alias=True,
            )
        )
    )
    config1 = {
        "optimizer": optimizer1_config_json,
        "connectors": ["measure"], 
        "measure": measure_config_json
    }
    optimizer2 = Optimizer(id="dev.opsani.com/multi-servox-2", token="987654321")
    optimizer2_config_json = json.loads(
        json.dumps(
            optimizer2.dict(
                by_alias=True,
            )
        )
    )
    config2 = {
        "optimizer": optimizer2_config_json,
        "connectors": ["measure"], 
        "measure": measure_config_json
    }
    config_yaml = yaml.dump_all([config1, config2])
    config_path.write_text(config_yaml)
    return config_path


# Ensure no files from the working copy and found
@pytest.fixture(autouse=True)
def run_from_tmp_path(tmp_path: Path) -> None:
    os.chdir(tmp_path)


# Ensure that we don't have configuration bleeding into tests
@pytest.fixture(autouse=True)
def run_in_clean_environment() -> None:
    for key, value in os.environ.copy().items():
        if key.startswith("SERVO_") or key.startswith("OPSANI_"):
            os.environ.pop(key)


@pytest.fixture(scope='function')
def random_string() -> str:
    letters = string.ascii_letters
    return "".join(random.choice(letters) for i in range(32))


@pytest.fixture
async def kubeconfig() -> str:
    config_path = Path(__file__).parents[0] / "kubeconfig"
    if not config_path.exists():
        raise FileNotFoundError(
            f"no kubeconfig file found at '{config_path}': configure a test cluster and add the kubeconfig file"
        )

    # Load the test config into async kubernetes
    await kubernetes_asyncio_config.load_kube_config(
        config_file=str(config_path),
    )

    return str(config_path)


@pytest.fixture()
async def subprocess() -> SubprocessTestHelper:
    return SubprocessTestHelper()


async def build_docker_image(
    tag: str = "servox:latest",
    *,
    preamble: Optional[str] = None,
    print_output: bool = True,
    **kwargs,
) -> str:
    root_path = Path(__file__).parents[1]
    subprocess = SubprocessTestHelper()
    exit_code, stdout, stderr = await subprocess(
        f"{preamble or 'true'} && DOCKER_BUILDKIT=1 docker build -t {tag} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:latest {root_path}",
        print_output=print_output,
        **kwargs,
    )
    if exit_code != 0:
        error = "\n".join(stderr)
        raise RuntimeError(
            f"Docker build failed with exit code {exit_code}: error: {error}"
        )

    return tag


@pytest.fixture()
async def servo_image() -> str:
    return await build_docker_image()


@pytest.fixture()
async def minikube_servo_image(servo_image: str) -> str:
    return await build_docker_image(preamble="eval $(minikube -p minikube docker-env)")
