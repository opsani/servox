import asyncio

import pytest
from pydantic import Extra

from servo import BaseConfiguration, BaseConnector, Duration, Optimizer
from servo.repeating import Mixin, repeating

pytestmark = pytest.mark.asyncio


class RepeatingConnector(BaseConnector):
    def run_me(self) -> None:
        pass

    class Config:
        # Needed for mocking
        extra = Extra.allow


@pytest.fixture(autouse=True)
async def cleanup_tasks() -> None:
    yield
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]


@pytest.mark.parametrize("every", [0.1, Duration(0.1), "0.1s", "1ms", "1ns"])
async def test_start_repeating_task(mocker, optimizer: Optimizer, every):
    connector = RepeatingConnector.construct()
    spy = mocker.spy(connector, "run_me")
    connector.start_repeating_task("report_progress", every, connector.run_me)
    await asyncio.sleep(0.0001)
    spy.assert_called()


async def test_start_repeating_lambda_task(mocker, optimizer: Optimizer):
    connector = RepeatingConnector.construct()
    spy = mocker.spy(connector, "run_me")
    connector.start_repeating_task("report_progress", 0.1, lambda: connector.run_me())
    await asyncio.sleep(0.001)
    spy.assert_called_once()


async def test_start_repeating_task_name_already_exists(optimizer: Optimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task(
        "report_progress", 0.1, lambda: connector.run_me()
    )
    assert task
    task.cancel()
    with pytest.raises(KeyError) as e:
        connector.start_repeating_task(
            "report_progress", 0.1, lambda: connector.run_me()
        )
    assert "repeating task already exists named 'report_progress'" in str(e.value)


async def test_cancel_repeating_task(optimizer: Optimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task(
        "report_progress", 0.1, lambda: connector.run_me()
    )
    connector.cancel_repeating_task("report_progress")
    await asyncio.sleep(0.001)
    assert task.cancelled()


async def test_cancel_repeating_task_name_doesnt_exist(optimizer: Optimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    result = connector.cancel_repeating_task("report_progress")
    assert result == None


async def test_cancel_repeating_task_already_cancelled(optimizer: Optimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task(
        "report_progress", 0.001, lambda: connector.run_me()
    )
    await asyncio.sleep(0.002)
    connector.cancel_repeating_task("report_progress")
    await asyncio.sleep(0.001)
    result = connector.cancel_repeating_task("report_progress")
    assert result == False


async def test_start_repeating_task_for_done_task(optimizer: Optimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task("report_progress", 0.0001, connector.run_me)
    await asyncio.sleep(0.0001)
    task.cancel()
    await asyncio.sleep(0.0001)
    assert task.done()
    task = connector.start_repeating_task("report_progress", 0.0001, connector.run_me)
    assert task


async def test_repeating_task_decorator():
    class RepeatedDecorator(Mixin):
        def __init__(self):
            self.called = False

        @repeating("1ms")
        async def repeat_this(self) -> None:
            self.called = True

    repeated = RepeatedDecorator()
    assert not repeated.called
    await asyncio.sleep(0.0001)
