from __future__ import annotations

from datetime import datetime

import asynctest
import loguru
import pytest
from freezegun import freeze_time

import servo
import servo.events
import servo.servo
from servo.logging import (
    DEFAULT_FILTER,
    DEFAULT_FORMATTER,
    ProgressHandler,
    log_execution,
    log_execution_time,
    logger,
    reset_to_defaults,
    set_level,
)

@pytest.fixture(autouse=True)
def reset_logging() -> None:
    servo.connector._current_context_var.set(None)
    servo.events._current_context_var.set(None)
    servo.assembly._current_context_var.set(None)
    # servo.servo._current_context_var.set(None)
    # Remove all handlers during logging tests
    logger.remove(None)
    yield
    reset_to_defaults()


class TestFilter:
    def test_logging_to_trace(self) -> None:
        messages = []
        logger.add(lambda m: messages.append(m), filter=DEFAULT_FILTER, level=0)

        set_level("TRACE")
        logger.critical("critical")
        logger.debug("debug")
        logger.trace("trace1")
        logger.trace("trace2")
        messages = list(map(lambda m: m.record["message"], messages))
        assert messages == ["critical", "debug", "trace1", "trace2"]

    def test_filtering_by_level(self) -> None:
        messages = []

        def raw_messages() -> List[str]:
            return list(map(lambda m: m.record["message"], messages))

        logger.remove(None)
        logger.add(lambda m: messages.append(m), filter=DEFAULT_FILTER, level=0)

        set_level("CRITICAL")
        logger.debug("Test")
        assert raw_messages() == []
        logger.critical("Another")
        assert raw_messages() == ["Another"]
        logger.info("Info")
        assert raw_messages() == ["Another"]

        set_level("INFO")
        logger.info("Info")
        assert raw_messages() == ["Another", "Info"]
        logger.trace("Trace")
        assert raw_messages() == ["Another", "Info"]

        set_level("TRACE")
        logger.trace("trace")
        logger.debug("debug")
        logger.error("error")
        assert raw_messages() == ["Another", "Info", "trace", "debug", "error"]


class TestFormatting:
    @property
    def name(self) -> str:
        return "TestFormatting"

    @pytest.fixture()
    def messages(self) -> List[str]:
        return []

    @pytest.fixture(autouse=True)
    def setup_handler(self, messages) -> loguru.Logger:
        logger.add(lambda m: messages.append(m), format=DEFAULT_FORMATTER, level=0)

    def test_explicit_component(self, messages):
        logger.info("Test", component="something_cool")
        message = messages[0]
        assert message.record["message"] == "Test"
        attributed_message = message.rsplit(" | ", 1)[1].strip()
        assert attributed_message == "something_cool - Test"

    def test_connector_name_included(self, messages):
        logger.info("Test", connector=self)
        message = messages[0]
        assert message.record["message"] == "Test"
        attributed_message = message.rsplit(" | ", 1)[1].strip()
        assert attributed_message == "TestFormatting - Test"

    def test_servo_name_default_included(self, messages):
        logger.info("Test")
        message = messages[0]
        assert message.record["message"] == "Test"
        attributed_message = message.rsplit(" | ", 1)[1].strip()
        assert attributed_message == "servo - Test"

    def test_event_context_included(self, messages):
        with servo.events.EventContext.from_str("before:adjust").current():
            logger.info("Test", connector=self)
            message = messages[0]
            assert message.record["message"] == "Test"
            attributed_message = message.rsplit(" | ", 1)[1].strip()
            assert attributed_message == "TestFormatting[before:adjust] - Test"

    def test_connector_context_var(self, messages):
        servo.connector._current_context_var.set(self)
        servo.events._current_context_var.set(servo.events.EventContext.from_str("before:adjust"))
        logger.info("Test")
        message = messages[0]
        assert message.record["message"] == "Test"
        attributed_message = message.rsplit(" | ", 1)[1].strip()
        assert attributed_message == "TestFormatting[before:adjust] - Test"

    def test_traceback(self, messages) -> None:
        logger.info("Test", with_traceback=True)
        message = messages[0]
        assert message.record["message"] == "Test"
        assert message.record["extra"]["traceback"]
        message_with_traceback = message.split("\n")
        assert len(message_with_traceback) > 5  # arbitrary check of depth


class TestProgressHandler:
    @pytest.fixture()
    def progress_reporter(self, mocker):
        return asynctest.Mock(name="progress reporter")

    @pytest.fixture()
    def error_reporter(self, mocker):
        return asynctest.Mock(name="error reporter")

    @pytest.fixture()
    async def handler(
        self, progress_reporter, error_reporter, event_loop
    ) -> ProgressHandler:
        handler = ProgressHandler(progress_reporter, error_reporter)
        yield handler
        await handler.shutdown()

    @pytest.fixture()
    def logger(self, handler: ProgressHandler) -> loguru.Logger:
        logger = loguru.logger.bind(connector="progress")
        logger.add(handler.sink)
        return logger

    async def test_ignored_without_progress_attribute(
        self, logger, progress_reporter, error_reporter
    ):
        logger.critical("Test...")
        await logger.complete()
        progress_reporter.assert_not_called()
        error_reporter.assert_not_called()

    async def test_error_no_connector(self, logger, progress_reporter, error_reporter):
        logger.critical("Test...", progress=50, connector=None)
        await logger.complete()
        progress_reporter.assert_not_called()
        assert (
            "declining request to report progress for record without a connector attribute"
            in error_reporter.call_args.args[0]
        )

    async def test_error_no_operation(self, logger, progress_reporter, error_reporter):
        logger.critical("Test...", progress=50, connector="foo")
        await logger.complete()
        progress_reporter.assert_not_called()
        assert (
            "declining request to report progress for record without an operation parameter or inferrable value from event context"
            in error_reporter.call_args.args[0]
        )

    async def test_error_no_started_at(self, logger, progress_reporter, error_reporter):
        logger.critical("Test...", progress=50, connector="foo", operation="hacking")
        await logger.complete()
        progress_reporter.assert_not_called()
        assert (
            "declining request to report progress for record without a started_at parameter or inferrable value from event context"
            in error_reporter.call_args.args[0]
        )

    async def test_success(self, logger, progress_reporter, error_reporter):
        logger.critical(
            "Test...",
            progress=50,
            connector="foo",
            operation="hacking",
            started_at=datetime.now(),
        )
        await logger.complete()
        progress_reporter.assert_called()
        error_reporter.assert_not_called()

    # NOTE: Inference tests dependent on context vars (operation and started_at)
    async def test_success_inference_from_context_var(
        self, logger, progress_reporter, error_reporter
    ):
        event_context = servo.events.EventContext.from_str("before:measure")
        with event_context.current():
            logger.critical("Test...", progress=50, connector="foo")
            await logger.complete()
            progress_reporter.assert_called()
            error_reporter.assert_not_called()


def test_log_execution() -> None:
    @log_execution
    def log_me():
        pass

    messages = []
    logger.add(lambda m: messages.append(m), level=0)
    log_me()
    assert messages[0].record["message"] == "Entering 'log_me' (args=(), kwargs={})"
    assert messages[1].record["message"] == "Exiting 'log_me' (result=None)"


@freeze_time("July 26th, 2020", auto_tick_seconds=15)
def test_log_execution_time() -> None:
    @log_execution_time(level="INFO")
    def log_me():
        pass

    messages = []
    logger.add(lambda m: messages.append(m), level=0)
    log_me()
    assert messages[0].record["message"] == "Function 'log_me' executed in 15s"
    assert messages[0].record["level"].name == "INFO"


@freeze_time("July 26th, 2020", auto_tick_seconds=15)
def test_log_execution_time_no_args() -> None:
    @log_execution_time
    def log_me():
        pass

    messages = []
    logger.add(lambda m: messages.append(m), level=0)
    log_me()
    assert messages[0].record["message"] == "Function 'log_me' executed in 15s"
