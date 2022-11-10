import asyncio

import pytest
from pydantic import Extra

from servo import BaseConfiguration, BaseConnector, Duration, OpsaniOptimizer
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
async def test_start_repeating_task(mocker, every):
    connector = RepeatingConnector.construct()
    spy = mocker.spy(connector, "run_me")
    connector.start_repeating_task("report_progress", every, connector.run_me)
    await asyncio.sleep(0.0001)
    spy.assert_called()


async def test_start_repeating_lambda_task(mocker):
    connector = RepeatingConnector.construct()
    spy = mocker.spy(connector, "run_me")
    connector.start_repeating_task("report_progress", 0.1, lambda: connector.run_me())
    await asyncio.sleep(0.001)
    spy.assert_called_once()


async def test_start_repeating_task_name_already_exists(optimizer: OpsaniOptimizer):
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


async def test_cancel_repeating_task(optimizer: OpsaniOptimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task(
        "report_progress", 0.1, lambda: connector.run_me()
    )
    connector.cancel_repeating_task("report_progress")
    await asyncio.sleep(0.001)
    assert task.cancelled()


async def test_cancel_repeating_tasks(optimizer: OpsaniOptimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task1 = connector.start_repeating_task(
        "report_progress1", 0.1, lambda: connector.run_me()
    )
    task2 = connector.start_repeating_task(
        "report_progress2", 0.1, lambda: connector.run_me()
    )
    connector.cancel_repeating_tasks()
    await asyncio.sleep(0.001)
    assert task1.cancelled()
    assert task2.cancelled()
    assert connector.repeating_tasks == {
        "report_progress1": task1,
        "report_progress2": task2,
    }


async def test_cancel_repeating_task_name_doesnt_exist(optimizer: OpsaniOptimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    result = connector.cancel_repeating_task("report_progress")
    assert result == None


async def test_cancel_repeating_task_already_cancelled(optimizer: OpsaniOptimizer):
    connector = RepeatingConnector(config=BaseConfiguration(), optimizer=optimizer)
    task = connector.start_repeating_task(
        "report_progress", 0.001, lambda: connector.run_me()
    )
    await asyncio.sleep(0.002)
    connector.cancel_repeating_task("report_progress")
    await asyncio.sleep(0.001)
    result = connector.cancel_repeating_task("report_progress")
    assert result == task
    assert result.cancelled()


async def test_start_repeating_task_for_done_task(optimizer: OpsaniOptimizer):
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
        called: bool = False

        @repeating("1ms")
        async def repeat_this(self) -> None:
            self.called = True

    repeated = RepeatedDecorator()
    assert not repeated.called
    await asyncio.sleep(0.0001)
