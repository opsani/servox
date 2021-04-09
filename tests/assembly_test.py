
import asyncio
import fastapi
import pathlib
import threading
from typing import Generator
import uvicorn

import pytest

import servo
import servo.connectors.prometheus
import servo.connectors.vegeta
import servo.runner
import tests.fake
import tests.helpers

test_optimizer_config = {
    "id": "dev.opsani.com/servox-integration-tests",
    "token": "179eddc9-20e2-4096-b064-824b72a83b7d",
}

@pytest.fixture
def isolated_fakeapi_url(event_loop: asyncio.AbstractEventLoop, fastapi_app: fastapi.FastAPI, unused_tcp_port: int) -> Generator[str, None, None]:
    """Run a fake OpsaniApi uvicorn server as a pytest fixture and yield the base URL for accessing it.
    Unlike the definition in conftest, this override runs in its own thread with its own
    event loop as assembly expects to start and close the loop its being run on
    """
    fastapi_app.optimizer = tests.fake.SequencedOptimizer(**test_optimizer_config)
    event_loop.run_until_complete(fastapi_app.optimizer.enter_state(tests.fake.StateMachine.States.awaiting_description))
    fastapi_app.optimizer.sequence(
        fastapi_app.optimizer.done(),
        fastapi_app.optimizer.reset(),
        fastapi_app.optimizer.request_description(),
    )
    # event_loop.run_until_complete(fastapi_app.optimizer.request_description())
    # Note: config of loop=none is necessary to prevent 'No event loop for Thread' errors. Server still runs on uvloop
    server = uvicorn.Server(config=uvicorn.Config(fastapi_app, host="127.0.0.1", loop="none", port=unused_tcp_port))
    def run_server_in_thread():
        asyncio.set_event_loop(asyncio.new_event_loop())
        server.run()

    thread = threading.Thread(target=run_server_in_thread)
    thread.start()

    yield f"http://{server.config.host}:{server.config.port}/"

    server.should_exit = True
    thread.join()

@pytest.fixture()
def assembly(event_loop: asyncio.AbstractEventLoop, isolated_fakeapi_url: str, servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "prometheus": servo.connectors.prometheus.PrometheusConnector,
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    # TODO: This needs a real optimizer ID
    optimizer = servo.configuration.Optimizer(base_url=isolated_fakeapi_url, **test_optimizer_config)
    assembly_ = event_loop.run_until_complete( servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    ))
    return assembly_

@pytest.fixture
def assembly_runner(assembly: servo.Assembly) -> servo.runner.AssemblyRunner:
    """Return an unstarted assembly runner."""
    return servo.runner.AssemblyRunner(assembly)

# TODO:
# def test_file_config_update_none_strategy():
#     pass

@pytest.mark.timeout(10)
def test_file_config_update_shutdown(
    event_loop: asyncio.AbstractEventLoop,
    assembly_runner: servo.runner.AssemblyRunner,
    isolated_fakeapi_url: str,
    servo_yaml: pathlib.Path
) -> None:
    asyncio.set_event_loop(event_loop)
    assembly_runner.assembly.config_change_strategy = servo.ConfigurationFileChangedStrategy.shutdown

    # Capture critical logs
    messages = []
    assembly_runner.logger.add(lambda m: messages.append(m), level=50)

    async def update_config():
        await asyncio.sleep(2)
        servo_yaml.write_text("test update, won't be loaded")

    event_loop.create_task(update_config())

    # Blocks until servo shutdown is triggered by above coro. Test marked with aggressive timeout accordingly
    assembly_runner.run()

    assert assembly_runner.running == False
    assert "Config file change detected (Change.modified), shutting down Servox for config reload" in messages[0]

@pytest.mark.timeout(10)
def test_file_config_update_reload(
    event_loop: asyncio.AbstractEventLoop,
    assembly_runner: servo.runner.AssemblyRunner,
    servo_yaml: pathlib.Path
) -> None:
    servo.logging.set_level("TRACE")
    asyncio.set_event_loop(event_loop)
    assembly_runner.assembly.config_change_strategy = servo.ConfigurationFileChangedStrategy.reload

    # Capture critical logs
    messages = []
    assembly_runner.logger.add(lambda m: messages.append(m), level=50)

    new_config_model = servo.assembly._create_config_model_from_routes(
        {
            "prometheus": servo.connectors.prometheus.PrometheusConnector,
            "adjust": tests.helpers.AdjustConnector,
            "vegeta": servo.connectors.vegeta.VegetaConnector,
        }
    )
    new_config = new_config_model.generate()

    async def update_config():
        await asyncio.sleep(2)
        servo_yaml.write_text(new_config.yaml())

    event_loop.create_task(update_config())

    # Blocks until servo shutdown is triggered by above coro. Test marked with aggressive timeout accordingly
    assembly_runner.run()

    assert assembly_runner.running == False
    assert "Config file change detected (Change.modified), shutting down active Servo(s) for config reload" in messages[0]
