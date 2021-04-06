
import asyncio
import fastapi
import multiprocessing
import pathlib
import threading
import time
from typing import AsyncGenerator

import kubetest.client
import pytest
import uvicorn

import servo
import servo.connectors.prometheus
import servo.runner
import tests.fake
import tests.helpers

test_optimizer_config = {
    "id": "dev.opsani.com/servox-integration-tests",
    "token": "179eddc9-20e2-4096-b064-824b72a83b7d",
}

tests.fake.api.optimizer = tests.fake.SequencedOptimizer(**test_optimizer_config)

@pytest.fixture
def isolated_fakeapi_url(fastapi_app: fastapi.FastAPI, unused_tcp_port: int) -> AsyncGenerator[str, None]:
    """Run a fake OpsaniApi uvicorn server as a pytest fixture and yield the base URL for accessing it.
    Unlike the base definition in conftest, this override runs in its own process to eliminate the need for a running
    event loop as assembly expects to start and close its own loop
    """
    server = uvicorn.Server(config=uvicorn.Config(fastapi_app, host="127.0.0.1", port=unused_tcp_port))
    proc = multiprocessing.Process(target=server.serve)
    proc.start()
    
    yield server.base_url

    # teardown
    proc.kill()
    # join process?

@pytest.fixture()
def assembly(event_loop: asyncio.AbstractEventLoop, servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "prometheus": servo.connectors.prometheus.PrometheusConnector,
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    # TODO: This needs a real optimizer ID
    optimizer = servo.configuration.Optimizer(**test_optimizer_config)
    assembly_ = event_loop.run_until_complete( servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    ))
    return assembly_

# Expose private shutdown method for test teardown
class TestAssemblyRunner(servo.runner.AssemblyRunner):
    async def shutdown(self) -> None:
        await self._shutdown(loop=asyncio.get_event_loop())

@pytest.fixture
def assembly_runner(assembly: servo.Assembly) -> TestAssemblyRunner:
    """Return an unstarted assembly runner."""
    return TestAssemblyRunner(assembly)

def test_file_config_update(
    event_loop: asyncio.AbstractEventLoop,
    assembly_runner: TestAssemblyRunner,
    fakeapi_url: str,
    servo_yaml: pathlib.Path
) -> None:
    servo.logging.set_level("DEBUG")
    asyncio.set_event_loop(event_loop)
    servo_ = assembly_runner.assembly.servos[0]
    servo_.optimizer.base_url = fakeapi_url
    for connector in servo_.connectors:
        connector.optimizer.base_url = fakeapi_url
        # below is nop due to api_client_options being a @property decorator. An anonymous dict is being updated but is not stored anywhere
        # connector.api_client_options.update(servo_.api_client_options)
    messages = []
    assembly_runner.logger.add(lambda m: messages.append(m), level=0)

    async def update_config():
        await asyncio.sleep(3)
        servo_yaml.write_text("test update, won't be loaded")

    event_loop.create_task(update_config())
    assembly_runner.run()

    assert assembly_runner.running == False
    assert "Config file change detected (Change.modified), shutting down active Servo(s) for config reload" in messages
