import asyncio
from contextlib import contextmanager
import pathlib
from typing import AsyncGenerator
import typing

import devtools
import pytest
import pytest_mock
import unittest.mock

import servo
import servo.events
import servo.runner
import servo.connectors.prometheus
import servo.types
import tests.fake
import tests.helpers


@pytest.fixture()
async def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            "prometheus": servo.connectors.prometheus.PrometheusConnector,
            "adjust": tests.helpers.AdjustConnector,
        }
    )
    # TODO: This needs a real optimizer ID
    optimizer = servo.configuration.OpsaniOptimizer(
        id="dev.opsani.com/servox",
        token="00000000-0000-0000-0000-000000000000",
    )
    config = config_model.generate(optimizer=optimizer)
    servo_yaml.write_text(config.yaml())

    assembly_ = await servo.assembly.Assembly.assemble(config_file=servo_yaml)
    servo_config: servo.BaseServoConfiguration = assembly_.servos[0].config
    servo_config.settings.backoff.__root__["__default__"].max_time = servo.Duration(
        "30s"
    )  # override default 10m timeout
    return assembly_


@pytest.fixture
async def assembly_runner(
    assembly: servo.Assembly,
) -> AsyncGenerator[servo.runner.AssemblyRunner, None]:
    """Return an unstarted assembly runner."""
    runner = servo.runner.AssemblyRunner(assembly)
    yield runner
    if runner.progress_handler is not None:
        await runner.progress_handler.shutdown()


@contextmanager
def patch_event_loop() -> typing.Iterator[asyncio.AbstractEventLoop]:
    event_loop = asyncio.get_event_loop()
    # NOTE: using the pytest_mocker fixture for mocking the event loop can cause side effects with pytest-asyncio
    #   (eg. when fixture mocking the `stop` method, the test will run forever).
    #   By using unittest.mock, we can ensure the event_loop is restored before exiting this method

    # Event loop is already running from pytest setup, runner trying to run the loop again produces an error
    with unittest.mock.patch.object(
        event_loop, "run_until_complete", return_value=None
    ):
        with unittest.mock.patch.object(event_loop, "run_forever", return_value=None):
            # run_forever no longer blocks causing loop.close() to be called immediately, stop runner from closing it to prevent errors
            with unittest.mock.patch.object(event_loop, "close", return_value=None):
                yield event_loop


@pytest.mark.asyncio
@tests.helpers.api_mock
async def test_assembly_shutdown_with_non_running_servo(
    assembly_runner: servo.runner.AssemblyRunner,
):
    with patch_event_loop() as event_loop:

        async def wait_for_servo_running():
            while not assembly_runner.assembly.servos[0].is_running:
                await asyncio.sleep(0.01)

        try:
            assembly_runner.run()
        except ValueError as e:
            if "add_signal_handler() can only be called from the main thread" in str(e):
                # https://github.com/pytest-dev/pytest-xdist/issues/620
                pytest.xfail("not running in the main thread")
            else:
                raise

        await asyncio.wait_for(wait_for_servo_running(), timeout=2)

        # Shutdown the servo to produce edge case error
        await assembly_runner.assembly.servos[0].shutdown()
        try:
            await assembly_runner.assembly.shutdown()
        except:
            raise
        finally:
            # Teardown runner asyncio tasks so they don't raise errors when the loop is closed by pytest
            await assembly_runner.shutdown(event_loop)


@pytest.mark.timeout(5)
@tests.helpers.api_mock
def test_assembly_run_raises_task_errors(
    mocker: pytest_mock.MockerFixture, assembly_runner: servo.runner.AssemblyRunner
):
    # with patch_event_loop() as event_loop:
    mock = mocker.patch.object(servo.runner.ServoRunner, "run_main_loop")
    mock.side_effect = RuntimeError("KABOOM")
    with pytest.raises(ExceptionGroup) as eg:
        assembly_runner.run()
        # event_loop.run_until_complete(assembly_runner._root_task)
        print(mock.called)

    assert tests.helpers.unwrap_exception_group(
        eg.value, RuntimeError, 1
    ), devtools.pformat(eg.value)


@pytest.fixture
async def servo_runner(assembly: servo.Assembly) -> servo.runner.ServoRunner:
    """Return an unstarted servo runner."""
    return servo.runner.ServoRunner(assembly.servos[0])


@pytest.mark.asyncio
@pytest.fixture
async def running_servo(
    event_loop: asyncio.AbstractEventLoop,
    servo_runner: servo.runner.ServoRunner,
) -> servo.runner.ServoRunner:
    """Start, run, and yield a servo runner.

    Lifecycle of the servo is managed on your behalf. When yielded, the servo will have its main
    runloop scheduled and will begin interacting with the optimizer API.
    """
    try:
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
@pytest.mark.asyncio
@pytest.mark.xfail(reason="too brittle.")
async def test_out_of_order_operations(servo_runner: servo.runner.ServoRunner) -> None:
    await servo_runner.servo.startup()
    response = await servo_runner.servo.post_event(
        servo.api.Events.hello, dict(agent=servo.api.user_agent())
    )
    debug(response)
    assert response.status == "ok"

    response = await servo_runner.servo.post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command in (servo.api.Commands.describe, servo.api.Commands.sleep)

    description = await servo_runner.describe(servo.types.Control())

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    debug(param)
    response = await servo_runner.servo.post_event(servo.api.Events.describe, param)
    debug(response)

    response = await servo_runner.servo.post_event(servo.api.Events.whats_next, None)
    debug(response)
    assert response.command == servo.api.Commands.measure

    # Send out of order adjust
    reply = {"status": "ok"}
    response = await servo_runner.servo.post_event(servo.api.Events.adjust, reply)
    debug(response)

    assert response.status == "unexpected-event"
    assert (
        response.reason
        == 'Out of order event "ADJUSTMENT", expected "MEASUREMENT"; ignoring'
    )

    servo_runner.logger.info("test logging", operation="ADJUST", progress=55)


@pytest.mark.asyncio
async def test_hello(
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str,
    fastapi_app: "tests.OpsaniAPI",
) -> None:
    static_optimizer = tests.fake.StaticOptimizer(
        id="dev.opsani.com/big-in-japan", token="31337"
    )
    await static_optimizer.request_description()
    fastapi_app.optimizer = static_optimizer
    servo_runner.servo.optimizer.base_url = fakeapi_url
    servo_runner.servo._api_client.base_url = servo_runner.servo.optimizer.default_url
    response = await servo_runner.servo.post_event(
        servo.api.Events.hello, dict(agent=servo.api.user_agent())
    )
    assert response.status == "ok"

    description = await servo_runner.describe(servo.types.Control())

    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    response = await servo_runner.servo.post_event(servo.api.Events.describe, param)


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


@pytest.mark.asyncio
async def test_authorization_redacted(
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str,
    fastapi_app: "tests.OpsaniAPI",
) -> None:
    static_optimizer = tests.fake.StaticOptimizer(
        id="dev.opsani.com/servox-integration-tests",
        token="00000000-0000-0000-0000-000000000000",
    )
    fastapi_app.optimizer = static_optimizer
    servo_runner.servo.optimizer.base_url = fakeapi_url
    servo_runner.servo._api_client.base_url = servo_runner.servo.optimizer.default_url

    # Capture TRACE logs
    messages = []
    servo_runner.logger.add(lambda m: messages.append(m), level=5)

    await servo_runner.servo.post_event(
        servo.api.Events.hello, dict(agent=servo.api.user_agent())
    )

    curlify_log = next(filter(lambda m: "curl" in m, messages))
    assert servo_runner.optimizer.token.get_secret_value() not in curlify_log


@pytest.mark.asyncio
async def test_control_sent_on_adjust(
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str,
    fastapi_app: "tests.OpsaniAPI",
    mocker: pytest_mock.MockFixture,
) -> None:
    sequenced_optimizer = tests.fake.SequencedOptimizer(
        id="dev.opsani.com/big-in-japan", token="31337"
    )
    control = servo.Control(settlement="10s")
    await sequenced_optimizer.recommend_adjustments(adjustments=[], control=control),
    sequenced_optimizer.sequence(sequenced_optimizer.done())
    fastapi_app.optimizer = sequenced_optimizer
    servo_runner.servo.optimizer.base_url = fakeapi_url
    servo_runner.servo._api_client.base_url = servo_runner.servo.optimizer.default_url

    adjust_connector = servo_runner.servo.get_connector("adjust")
    event_handler = adjust_connector.get_event_handlers(
        "adjust", servo.events.Preposition.on
    )[0]
    spy = mocker.spy(event_handler, "handler")

    async def wait_for_optimizer_done():
        while fastapi_app.optimizer.state.name != "done":
            await asyncio.sleep(0.01)

    async with asyncio.TaskGroup() as tg:
        _ = tg.create_task(servo_runner.run())
        async with asyncio.timeout(2):
            await tg.create_task(wait_for_optimizer_done())
        await tg.create_task(servo_runner.shutdown())

    spy.assert_called_once_with(adjust_connector, [], control)


# TODO: This doesn't need to be integration test
@pytest.mark.asyncio
@tests.helpers.api_mock
async def test_adjustment_rejected(
    mocker, servo_runner: servo.runner.ServoRunner
) -> None:
    connector = servo_runner.servo.get_connector("adjust")
    with servo.utilities.pydantic.extra(connector):
        on_handler = connector.get_event_handlers(
            "adjust", servo.events.Preposition.on
        )[0]
        mock = mocker.patch.object(on_handler, "handler")
        mock.side_effect = servo.errors.AdjustmentRejectedError()
        await servo_runner.servo.startup()
        with pytest.raises(ExceptionGroup) as excg:
            await servo_runner.adjust([], servo.Control())

        _ = tests.helpers.unwrap_exception_group(
            excg.value, servo.errors.AdjustmentRejectedError, 1
        )
