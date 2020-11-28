
import asyncio
import pathlib
import yaml
from typing import Dict, List, Optional, Union

import fastapi
import httpx
import pydantic
import pytest

import servo
import servo.connectors.prometheus
import tests.helpers
import tests.fake

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

@pytest.fixture()
def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "prometheus": servo.connectors.prometheus.PrometheusConnector,
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    optimizer = servo.configuration.Optimizer(
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
        servo.Servo.set_current(servo_runner.servo)
        yield servo_runner

    finally:
        await servo_runner.shutdown()

        # Cancel outstanding tasks
        # tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        # [task.cancel() for task in tasks]

        # await asyncio.gather(*tasks, return_exceptions=True)

# TODO: Switch this over to using a FakeAPI
async def test_test_out_of_order_operations(servo_runner: servo.runner.ServoRunner) -> None:
    await servo_runner.servo.startup()
    response = await servo_runner._post_event(
        servo.api.Events.hello, dict(agent=servo.api.USER_AGENT)
    )
    debug(response)
    assert response.status == "ok"

    response = await servo_runner._post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command == servo.api.Command.DESCRIBE

    description = await servo_runner.describe()

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    debug(param)
    response = await servo_runner._post_event(servo.api.Events.describe, param)
    debug(response)

    response = await servo_runner._post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command == servo.api.Command.MEASURE

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

    await asyncio.sleep(5)


async def test_hello(servo_runner: servo.runner.ServoRunner, fakeapi_url: str) -> None:
    servo_runner.servo.optimizer.base_url = fakeapi_url
    response = await servo_runner._post_event(
        servo.api.Events.hello, dict(agent=servo.api.USER_AGENT)
    )
    assert response.status == "ok"

    description = await servo_runner.describe()

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    response = await servo_runner._post_event(servo.api.Events.describe, param)
    debug(response)
    await asyncio.sleep(10)


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
#
async def test_adjustment_rejected(mocker, runner) -> None:
    connector = runner.servo.get_connector("adjust")
    with servo.utilities.pydantic.extra(connector):
        on_handler = connector.get_event_handlers("adjust", servo.events.Preposition.ON)[0]
        mock = mocker.patch.object(on_handler, "handler")
        mock.side_effect = servo.errors.AdjustmentRejectedError()
        await runner.servo.startup()
        with pytest.raises(servo.errors.AdjustmentRejectedError):
            await runner.adjust([], servo.Control())
async def test_fakeapi(fakeapi_client: httpx.AsyncClient) -> None:
    response = await fakeapi_client.get("/")
    debug(response.json())


@pytest.fixture
async def progress_logging() -> None:

    # Setup logging
    # TODO: encapsulate all this shit
    async def report_progress(**kwargs) -> None:
        # Forward to the active servo...
        await servo.Servo.current().report_progress(**kwargs)

    def handle_progress_exception(error: Exception) -> None:
        # FIXME: This needs to be made multi-servo aware
        # Restart the main event loop if we get out of sync with the server
        if isinstance(error, servo.api.UnexpectedEventError):
            servo.logging.logger.error(
                "servo has lost synchronization with the optimizer: restarting operations"
            )

            tasks = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task()
            ]
            servo.logging.logger.info(f"Canceling {len(tasks)} outstanding tasks")
            [task.cancel() for task in tasks]

            # Restart a fresh main loop
            # runner = self._runner_for_servo(servo.Servo.current())
            # asyncio.create_task(runner.main_loop(), name="main loop")

    progress_handler = servo.logging.ProgressHandler(
        report_progress, servo.logging.logger.warning, handle_progress_exception
    )
    servo.logging.logger.add(progress_handler.sink, catch=True)
    yield progress_handler
    await progress_handler.shutdown()

async def test_run(progress_logging, running_servo: servo.runner.ServoRunner) -> None:
    await asyncio.sleep(1200)

# -----------------------------------------------

# TODO: Replace these models...






class TestOptimizerStateMachine:
    async def test_idle_to_hello(self) -> None:
        ...

    async def test_setting_initial_state(self) -> None:
        ...

    async def test_progress_transitions(self) -> None:
        state_machine = OptimizerStateMachine('self', initial=States.awaiting_measurement)
        debug(state_machine.state)
        await state_machine.measurement()
        debug(state_machine.state)
