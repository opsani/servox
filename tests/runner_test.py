
import asyncio
import contextlib
import pathlib
from typing import Callable

import pytest
import unittest.mock

import servo
import servo.runner
import servo.connectors.prometheus
import tests.fake
import tests.helpers

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

test_optimizer_config = {
    "id": "dev.opsani.com/servox-integration-tests",
    "token": "179eddc9-20e2-4096-b064-824b72a83b7d",
}

@pytest.fixture()
async def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
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
    assembly_ = await servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly_


@pytest.fixture
def assembly_runner(assembly: servo.Assembly) -> servo.runner.AssemblyRunner:
    """Return an unstarted assembly runner."""
    return servo.runner.AssemblyRunner(assembly)

@contextlib.contextmanager
def mock_event_loop_run_close_stop(event_loop: asyncio.AbstractEventLoop) -> asyncio.AbstractEventLoop:
    # NOTE: defining this as a fixture causes RuntimeError: Event loop stopped before Future completed.
    # NOTE: using the pytest_mocker fixture for mocking the event loop can cause side effects with pytest-asyncio
    #   (eg. when fixture mocking the `stop` method, the test will run forever).
    #   By using unittest.mock, we can ensure the event_loop is restored before exiting the test method

    # Event loop is already running from pytest setup, runner trying to run the loop again produces an error
    with unittest.mock.patch.object(event_loop, 'run_forever', return_value=None):
        # run_forever no longer blocks causing loop.close() to be called immediately, stop assembly runner from closing it to prevent errors
        with unittest.mock.patch.object(event_loop, 'close', return_value=None):
            # stopping the loop causes test errors when fakeapi is running, don't let runner stop the loop
            with unittest.mock.patch.object(event_loop, 'stop', return_value=None):
                yield event_loop

@contextlib.contextmanager
def mock_all_tasks_exclude_test_tasks() -> None:
    test_tasks = asyncio.all_tasks()
    original_all_tasks = asyncio.all_tasks
    def all_tasks_side_effect():
        return set(filter(lambda t: t not in test_tasks, original_all_tasks()))

    with unittest.mock.patch('asyncio.all_tasks', side_effect=all_tasks_side_effect):
        yield

async def safe_wait(condition: Callable[[], bool], max_checks: int = 5, sleep_duration: float = 0.01) -> None:
    num_checks = 0
    while not condition() and num_checks < max_checks:
        num_checks += 1
        await asyncio.sleep(sleep_duration)

    if num_checks >= max_checks:
        raise RuntimeError(f"Infinite loop guard expired (max_checks: {max_checks}, sleep_duration: {sleep_duration})")

async def test_assembly_shutdown_with_non_running_servo(assembly_runner: servo.runner.AssemblyRunner):
    event_loop = asyncio.get_event_loop()
    with mock_event_loop_run_close_stop(event_loop) as mocked_loop:
        assembly_runner.run()
        await safe_wait(lambda: assembly_runner.assembly.servos[0].is_running)

        # Shutdown the servo to produce edge case error
        await assembly_runner.assembly.servos[0].shutdown()
        try:
            await assembly_runner.assembly.shutdown()
        except:
            raise
        finally:
            # Teardown runner asyncio tasks so they don't raise errors when the loop is closed by pytest
            await assembly_runner._shutdown(mocked_loop)

async def test_file_config_update(
    assembly_runner: servo.runner.AssemblyRunner,
    fakeapi_url: str,
    fastapi_app: 'tests.OpsaniAPI',
    servo_yaml: pathlib.Path
) -> None:
    # Test Setup
    assembly_runner.assembly.watch_config_file = True
    fastapi_app.optimizer = tests.fake.SequencedOptimizer(**test_optimizer_config)
    await fastapi_app.optimizer.request_description()

    servo_ = assembly_runner.assembly.servos[0]
    servo_.optimizer.base_url = fakeapi_url
    for connector in servo_.connectors:
        connector.optimizer.base_url = fakeapi_url

    # Capture critical logs
    messages = []
    assembly_runner.logger.add(lambda m: messages.append(m), level=50)

    # Mock event loop methods to prevent errors from servo trying to start/close it
    event_loop = asyncio.get_event_loop()
    with mock_event_loop_run_close_stop(event_loop) as mocked_loop:
        with mock_all_tasks_exclude_test_tasks():
            assembly_runner.run()
            await asyncio.sleep(1)
            servo_yaml.write_text("test update, won't be loaded")

            await safe_wait(lambda: not assembly_runner.running, sleep_duration=0.1)

    assert assembly_runner.running == False
    assert "Config file change detected (Change.modified), shutting down active Servo(s) for config reload" in messages[0]


@pytest.fixture
async def servo_runner(assembly: servo.Assembly) -> servo.runner.ServoRunner:
    """Return an unstarted servo runner."""
    return servo.runner.ServoRunner(assembly.servos[0])

@pytest.fixture
async def running_servo(
    event_loop: asyncio.AbstractEventLoop,
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str
) -> servo.runner.ServoRunner:
    """Start, run, and yield a servo runner.

    Lifecycle of the servo is managed on your behalf. When yielded, the servo will have its main
    runloop scheduled and will begin interacting with the optimizer API.
    """
    try:
        servo_runner.servo.optimizer.base_url = fakeapi_url
        for connector in servo_runner.servo.connectors:
            connector.optimizer.base_url = fakeapi_url
            connector.api_client_options.update(servo_runner.servo.api_client_options)
        event_loop.create_task(servo_runner.run())
        async with servo_runner.servo.current():
            yield servo_runner

    finally:
        await servo_runner.shutdown()

        # Cancel outstanding tasks
        # tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        # [task.cancel() for task in tasks]

        # await asyncio.gather(*tasks, return_exceptions=True)

# TODO: Switch this over to using a FakeAPI
@pytest.mark.xfail(reason="too brittle.")
async def test_out_of_order_operations(servo_runner: servo.runner.ServoRunner) -> None:
    await servo_runner.servo.startup()
    response = await servo_runner._post_event(
        servo.api.Events.hello, dict(agent=servo.api.user_agent())
    )
    debug(response)
    assert response.status == "ok"

    response = await servo_runner._post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command in (servo.api.Commands.describe, servo.api.Commands.sleep)

    description = await servo_runner.describe()

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    debug(param)
    response = await servo_runner._post_event(servo.api.Events.describe, param)
    debug(response)

    response = await servo_runner._post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command == servo.api.Commands.measure

    # Send out of order adjust
    reply = {"status": "ok"}
    response = await servo_runner._post_event(servo.api.Events.adjust, reply)
    debug(response)

    assert response.status == "unexpected-event"
    assert (
        response.reason
        == 'Out of order event "ADJUSTMENT", expected "MEASUREMENT"; ignoring'
    )

    servo_runner.logger.info("test logging", operation="ADJUST", progress=55)

async def test_hello(
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str,
    fastapi_app: 'tests.OpsaniAPI',
) -> None:
    static_optimizer = tests.fake.StaticOptimizer(id='dev.opsani.com/big-in-japan', token='31337')
    await static_optimizer.request_description()
    fastapi_app.optimizer = static_optimizer
    servo_runner.servo.optimizer.base_url = fakeapi_url
    response = await servo_runner._post_event(
        servo.api.Events.hello, dict(agent=servo.api.user_agent())
    )
    assert response.status == "ok"

    description = await servo_runner.describe()

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    response = await servo_runner._post_event(servo.api.Events.describe, param)

# async def test_describe() -> None:
#     pass

# async def test_measure() -> None:
#     pass

# async def test_adjust() -> None:
#     pass

# async def test_whats_next() -> None:
#     pass

# async def test_sleep() -> None:
#     pass

# async def test_goodbye() -> None:
#     pass

# @pytest.mark.integration
# @pytest.mark.parametrize(
#     ("proxies"),
#     [
#         "http://localhost:1234",
#         {"all://": "http://localhost:1234"},
#         {"https://": "http://localhost:1234"},
#         {"https://api.opsani.com": "http://localhost:1234"},
#         {"https://*.opsani.com": "http://localhost:1234"},
#     ]
# )
# async def test_proxies_support() -> None:
#     ...
#     # fire up runner.run and check .run, etc.


# TODO: This doesn't need to be integration test
async def test_adjustment_rejected(mocker, servo_runner: servo.runner.ServoRunner) -> None:
    connector = servo_runner.servo.get_connector("adjust")
    with servo.utilities.pydantic.extra(connector):
        on_handler = connector.get_event_handlers("adjust", servo.events.Preposition.on)[0]
        mock = mocker.patch.object(on_handler, "handler")
        mock.side_effect = servo.errors.AdjustmentRejectedError()
        await servo_runner.servo.startup()
        with pytest.raises(servo.errors.AdjustmentRejectedError):
            await servo_runner.adjust([], servo.Control())
