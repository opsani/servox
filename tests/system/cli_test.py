import asyncio
import os
import pathlib
import shutil

import pytest

import tests.helpers

# FIXME: This file should be getting marked as integration automatically
pytestmark = [pytest.mark.asyncio, pytest.mark.system]

# NOTE: Use module level functions so that we don't have to install multiple times


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def project_path(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("servox_app")


@pytest.fixture(scope="module", autouse=True)
async def install_servox(project_path: pathlib.Path, pytestconfig) -> None:
    os.chdir(project_path)

    await tests.helpers.Subprocess.shell(
        f"poetry init --name servox_app --dependency servox:{pytestconfig.rootpath} --no-interaction",
        print_output=True,
    )
    await tests.helpers.Subprocess.shell(
        "poetry config --local virtualenvs.in-project true", print_output=True
    )

    # Copy the lock file so we don't get unexpected package upgrades
    lock_file = pytestconfig.rootpath / "poetry.lock"
    shutil.copy(lock_file, project_path)

    await tests.helpers.Subprocess.shell(
        "poetry install --no-dev --no-root --no-interaction", print_output=True
    )


@pytest.mark.xfail(reason="poetry init --dependency localpath results in error")
async def test_generate_opsani_dev(project_path: pathlib.Path, subprocess) -> None:
    # FIXME: Should be unnecessary but another fixture is thrashing us
    os.chdir(project_path)

    await subprocess("poetry run servo generate opsani_dev", print_output=True)

    config_path = project_path / "servo.yaml"
    yaml_text = config_path.read_text()
    assert yaml_text == (
        "optimizer:\n"
        "  id: generated-id.test/generated\n"
        "  token: generated-token\n"
        "  url: https://api.opsani.com/accounts/generated-id.test/applications/generated/\n"
        "connectors:\n"
        "- opsani_dev\n"
        "opsani_dev:\n"
        "  namespace: default\n"
        "  deployment: app-deployment\n"
        "  container: main\n"
        "  service: app\n"
        "  cpu:\n"
        "    unit: cores\n"
        "    min: 250m\n"
        "    max: '4'\n"
        "  memory:\n"
        "    unit: GiB\n"
        "    min: 256.0Mi\n"
        "    max: 4.0Gi\n"
    )
