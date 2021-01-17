import asyncio
import datetime
import freezegun
import itertools
import operator
import pytest
import pytest_mock
import pydantic
import re
import servo
import servo.pubsub
import servo.utilities.pydantic

from typing import Callable, List, Optional


class TestMessage:
    def test_text_message(self) -> None:
        message = servo.pubsub.Message(text='A great and insightful message.')
        assert message.text == 'A great and insightful message.'
        assert message.content_type == 'text/plain'
        assert message.content == b'A great and insightful message.'

    def test_text_message_override_content_type(self) -> None:
        message = servo.pubsub.Message(text='A great and insightful message.', content_type="funky/text")
        assert message.text == 'A great and insightful message.'
        assert message.content_type == 'funky/text'
        assert message.content == b'A great and insightful message.'

    def test_text_message_raises_if_not_string(self) -> None:
        with pytest.raises(ValueError, match="Text Messages can only be created with `str` content: got 'int'"):
            servo.pubsub.Message(text=1234)

    def test_json_message(self) -> None:
        message = servo.pubsub.Message(json={"key": "value"})
        assert message.text == '{"key": "value"}'
        assert message.content_type == 'application/json'
        assert message.content == b'{"key": "value"}'

    @freezegun.freeze_time("2021-01-01 12:00:01")
    def test_json_message_via_protocol(self) -> None:
        # NOTE: Use Pydantic's json() method support
        channel = servo.pubsub.Channel.construct(name="whatever", created_at=datetime.datetime.now())
        message = servo.pubsub.Message(json=channel)
        assert message.text == '{"description": null, "created_at": "2021-01-01T12:00:01", "name": "whatever"}'
        assert message.content_type == 'application/json'
        assert message.content == b'{"description": null, "created_at": "2021-01-01T12:00:01", "name": "whatever"}'

    def test_yaml_message(self) -> None:
        message = servo.pubsub.Message(yaml={"key": "value"})
        assert message.text == 'key: value\n'
        assert message.content_type == 'application/x-yaml'
        assert message.content == b'key: value\n'

    def test_content_message(self) -> None:
        message = servo.pubsub.Message(content=b"This is the message", content_type="foo/bar")
        assert message.text == 'This is the message'
        assert message.content_type == 'foo/bar'
        assert message.content == b'This is the message'

    def test_created_at(self) -> None:
        message = servo.pubsub.Message(content=b"This is the message", content_type="foo/bar")
        assert message.created_at is not None

    class TestValidations:
        def test_content_required(self) -> None:
            with pytest.raises(pydantic.ValidationError) as excinfo:
                servo.pubsub.Message()

            assert {
                "loc": ("content",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in excinfo.value.errors()

        def test_content_type_required(self) -> None:
            with pytest.raises(pydantic.ValidationError) as excinfo:
                servo.pubsub.Message(content=b'foo')

            assert {
                "loc": ("content_type",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in excinfo.value.errors()

@pytest.fixture
async def exchange() -> servo.pubsub.Exchange:
    exchange = servo.pubsub.Exchange()
    yield exchange

    # Shutdown the exchange it is left running
    if exchange.running:
        await exchange.shutdown()
    else:
        exchange.clear()

@pytest.fixture
def channel(exchange: servo.pubsub.Exchange) -> servo.pubsub.Channel:
    # return servo.pubsub.Channel(name="metrics", exchange=exchange)
    return exchange.create_channel("metrics")

class TestChannel:
    class TestValidations:
        def test_name_required(self) -> None:
            with pytest.raises(pydantic.ValidationError) as excinfo:
                servo.pubsub.Channel()

            assert {
                "loc": ("name",),
                "msg": "field required",
                "type": "value_error.missing",
            } in excinfo.value.errors()

        def test_name_constraints(self) -> None:
            with pytest.raises(pydantic.ValidationError) as excinfo:
                servo.pubsub.Channel(name="THIS_IS_INVALID")

            assert {
                "loc": ("name",),
                "msg": 'string does not match regex "^[0-9a-zA-Z]([0-9a-zA-Z\\.-])*[0-9A-Za-z]$"',
                "type": "value_error.str.regex",
                'ctx': {
                    'pattern': '^[0-9a-zA-Z]([0-9a-zA-Z\\.-])*[0-9A-Za-z]$',
                }
            } in excinfo.value.errors()

        def test_exchange_required(self) -> None:
            with pytest.raises(pydantic.ValidationError) as excinfo:
                servo.pubsub.Channel()

            assert {
                "loc": ("exchange",),
                "msg": "field required",
                "type": "value_error.missing",
            } in excinfo.value.errors()

    def test_hashing(self, channel: servo.pubsub.Channel) -> None:
        channels = {channel,}
        copy_of_channel = channel.copy()
        assert copy_of_channel in channels
        copy_of_channel.name = "another_name"
        assert copy_of_channel not in channels
        channels.add(copy_of_channel)
        assert copy_of_channel in channels
        assert channel in channels

    def test_comparable_to_str(self, channel: servo.pubsub.Channel) -> None:
        assert channel != 'foo'
        assert channel == 'metrics'

    async def test_publish(self, channel: servo.pubsub.Channel, mocker: pytest_mock.MockFixture) -> None:
        message = servo.pubsub.Message(text="foo")
        with servo.utilities.pydantic.extra(channel.exchange):
            spy = mocker.spy(channel.exchange, "publish")
            await channel.publish(message)
            spy.assert_called_once_with(message, channel)

    async def test_publish_fails_if_channel_is_closed(self, channel: servo.pubsub.Channel, mocker: pytest_mock.MockFixture) -> None:
        await channel.close()
        assert channel.closed
        with pytest.raises(RuntimeError, match='Cannot publish messages to a closed Channel'):
            await channel.publish(servo.pubsub.Message(text="foo"))

    async def test_closing_channel_cancels_exclusive_subscribers(self, channel: servo.pubsub.Channel, mocker: pytest_mock.MockFixture) -> None:
        ...
        # TODO: attach one subscriber directly, one via glob, and one via regex and then close the channel

    async def test_iteration(self, channel: servo.pubsub.Channel, mocker: pytest_mock.MockFixture) -> None:
        messages = []

        async def _subscriber() -> None:
            async for message in channel:
                messages.append(message)

                if len(messages) == 3:
                    channel.stop()

        async def _publisher() -> None:
            for i in range(3):
                await channel.publish(servo.pubsub.Message(text=f"Message: {i}"))

        channel.exchange.start()
        await task_graph(
            _publisher(),
            _subscriber(),
            timeout=5.0
        )
        assert messages

class TestSubscription:
    def test_string_selector(self) -> None:
        subscription = servo.pubsub.Subscription(selector="metrics")
        assert subscription.selector == "metrics"
        assert isinstance(subscription.selector, str)

    def test_regex_selector(self) -> None:
        subscription = servo.pubsub.Subscription(selector=re.compile("metrics"))
        assert subscription.selector == re.compile("metrics")
        assert isinstance(subscription.selector, re.Pattern)

    def test_regex_selector_expansion(self) -> None:
        subscription = servo.pubsub.Subscription(selector="/metrics/")
        assert subscription.selector == re.compile("metrics")
        assert isinstance(subscription.selector, re.Pattern)

    def test_match_by_string(self, exchange: servo.pubsub.Exchange) -> None:
        metrics_channel = servo.pubsub.Channel(name="metrics", exchange=exchange)
        other_channel = servo.pubsub.Channel(name="other", exchange=exchange)
        message = servo.pubsub.Message(text="foo")
        subscription = servo.pubsub.Subscription(selector="metrics")
        assert subscription.matches(metrics_channel, message)
        assert not subscription.matches(other_channel, message)

    def test_match_no_message(self, exchange: servo.pubsub.Exchange) -> None:
        metrics_channel = servo.pubsub.Channel(name="metrics", exchange=exchange)
        other_channel = servo.pubsub.Channel(name="other", exchange=exchange)
        subscription = servo.pubsub.Subscription(selector="metrics")
        assert subscription.matches(metrics_channel)
        assert not subscription.matches(other_channel)

    @pytest.mark.parametrize(
        ("selector", "matches"),
        [
            ("metrics", False),
            ("metrics.prometheus.http", True),
            ("metrics.*", True),
            ("metrics.*.http", True),
            ("metrics.*.https", False),
            ("metrics.*.[abc]ttp", False),
            ("metrics.*.[hef]ttp", True),
        ]
    )
    def test_match_by_glob(self, exchange: servo.pubsub.Exchange, selector: str, matches: bool) -> None:
        metrics_channel = servo.pubsub.Channel(name="metrics.prometheus.http", exchange=exchange)
        message = servo.pubsub.Message(text="foo")
        subscription = servo.pubsub.Subscription(selector=selector)
        assert subscription.matches(metrics_channel, message) == matches

    @pytest.mark.parametrize(
        ("selector", "matches"),
        [
            ("/metrics/", False),
            ("/metrics.prometheus.http/", True),
            ("/metrics.*/", True),
            ("/metrics.*.http/", True),
            ("/metrics.*.https/", False),
            ("/metrics.*.[abc]ttp/", False),
            ("/metrics.*.[hef]ttp/", True),
            ("/metrics.(prometheus|datadog|newrelic).https?/", True),
        ]
    )
    def test_match_by_regex(self, exchange: servo.pubsub.Exchange, selector: str, matches: bool) -> None:
        channel = servo.pubsub.Channel(name="metrics.prometheus.http", exchange=exchange)
        message = servo.pubsub.Message(text="foo")
        subscription = servo.pubsub.Subscription(selector=selector)
        assert subscription.matches(channel, message) == matches, f"expected regex pattern '{selector}' match of '{channel.name}' to == {matches}"


@pytest.fixture
def subscriber(exchange: servo.pubsub.Exchange, subscription: servo.pubsub.Subscription) -> servo.pubsub.Subscriber:
    return servo.pubsub.Subscriber(exchange=exchange, subscription=subscription)

@pytest.fixture
def subscription(exchange: servo.pubsub.Exchange) -> servo.pubsub.Subscription:
    return servo.pubsub.Subscription(selector="metrics*")


class TestSubscriber:
    def test_not_cancelled_on_create(self, subscriber: servo.pubsub.Subscriber) -> None:
        assert not subscriber.cancelled

    async def test_sync_callback_is_invoked(self, subscriber: servo.pubsub.Subscriber, mocker: pytest_mock.MockerFixture) -> None:
        callback = mocker.Mock()
        subscriber.callback = callback

        message = servo.pubsub.Message(text="foo")
        channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)
        await subscriber(message, channel)
        callback.assert_called_once_with(message, channel)

    async def test_async_callback_is_invoked(self, subscriber: servo.pubsub.Subscriber, mocker: pytest_mock.MockerFixture) -> None:
        callback = mocker.AsyncMock()
        subscriber.callback = callback

        message = servo.pubsub.Message(text="foo")
        channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)
        await subscriber(message, channel)
        callback.assert_called_once_with(message, channel)

    async def test_async_iteration(self, subscriber: servo.pubsub.Subscriber) -> None:
        message = servo.pubsub.Message(text="foo")
        channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)

        messages = []
        event = asyncio.Event()
        async def _processor() -> None:
            event.set()
            async for message_, channel_ in subscriber:
                assert message_ == message
                assert channel_ == channel
                messages.append(message_)

                if len(messages) == 3:
                    subscriber.cancel()

        task = asyncio.create_task(_processor())
        await event.wait()
        for _ in range(3):
            await subscriber(message, channel)
        await task
        assert len(messages) == 3

    @pytest.fixture
    async def iterator_task_factory(self) -> Callable[[], asyncio.Task]:
        # TODO: This should accept a callback for customization
        async def _iterator_task_factory(subscriber: servo.pubsub.Subscriber) -> asyncio.Task:
            event = asyncio.Event()
            async def _iterate() -> None:
                messages = []
                event.set()
                async for message_, channel_ in subscriber:
                    messages.append(message_)

                    if len(messages) == 3:
                        subscriber.stop()

                return messages

            task = asyncio.create_task(_iterate())
            await event.wait()
            return task
        return _iterator_task_factory

    async def test_multiple_iterators(self, subscriber: servo.pubsub.Subscriber, iterator_task_factory: Callable[[], asyncio.Task]) -> None:
        message = servo.pubsub.Message(text="foo")
        channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)

        tasks = await asyncio.gather(
            iterator_task_factory(subscriber),
            iterator_task_factory(subscriber),
            iterator_task_factory(subscriber)
        )
        for _ in range(3):
            await subscriber(message, channel)
        results = await asyncio.gather(*tasks)
        messages = list(itertools.chain(*results))
        assert len(messages) == 9

    async def test_iterator_context(self, channel: servo.pubsub.Subscriber, subscriber: servo.pubsub.Subscriber) -> None:
        other_subscriber = servo.pubsub.Subscriber(exchange=subscriber.exchange, subscription=subscriber.subscription)

        async def _create_iterator(subscriber_, current):
            assert servo.pubsub._current_iterator() == current
            async for message_, channel_ in subscriber_:
                iterator = servo.pubsub._current_iterator()
                assert iterator
                assert iterator is not None, "Iterator context should not be None"
                assert iterator.subscriber == subscriber_
                subscriber_.stop()

        task = asyncio.gather(*[
            _create_iterator(subscriber, None),
            _create_iterator(other_subscriber, None),
        ])
        await asyncio.sleep(0.1)

        for subscriber in [subscriber, other_subscriber]:
            await subscriber(servo.pubsub.Message(text="foo"), channel)

        await task

    async def test_waiting(self, channel: servo.pubsub.Subscriber, subscriber: servo.pubsub.Subscriber) -> None:
        async def _iterator() -> None:
            async for message, channel in subscriber:
                subscriber.cancel()

        await asyncio.wait_for(
            task_graph(
                subscriber(servo.pubsub.Message(text="foo"), channel),
                _iterator(),
                subscriber.wait()
            ),
            timeout=1.0
        )
        assert subscriber.cancelled

    async def test_cannot_stop_inactive_iterator(self, channel, subscriber: servo.pubsub.Subscriber) -> None:
        other_subscriber = servo.pubsub.Subscriber(exchange=subscriber.exchange, subscription=subscriber.subscription)

        await asyncio.sleep(5)
        async def _test() -> self:
            with pytest.raises(RuntimeError, match="Attempted to stop an inactive iterator"):
                async for message_, channel_ in subscriber:
                    iterator = servo.pubsub._current_iterator()
                    assert iterator
                    assert iterator.subscriber == subscriber
                    other_subscriber.stop()

        await asyncio.gather(
            _test(),
            subscriber(servo.pubsub.Message(text="foo"), channel)
        )

    async def test_cannot_stop_without_an_iterator(self, subscriber: servo.pubsub.Subscriber) -> None:
        with pytest.raises(RuntimeError, match="Attempted to stop outside of an iterator"):
            subscriber.stop()

    async def test_cancellation_stops_all_iterators(self, channel, subscriber: servo.pubsub.Subscriber) -> None:
        async def _create_iterator():
            # will block waiting for messages
            async for message_, channel_ in subscriber:
                ...

        async def _cancel():
            subscriber.cancel()

        await asyncio.wait_for(
            task_graph(
                _cancel(),
                _create_iterator(), _create_iterator(), _create_iterator()
            ),
            timeout=1.0
        )

    async def _test_cannot_stop_nested_iterator(self, subscriber: servo.pubsub.Subscriber, iterator_task_factory: Callable[[], asyncio.Task]) -> None:
        channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)
        assert channel
        other_subscriber = subscriber.exchange.create_subscriber('metrics')
        assert other_subscriber

        async def _boom() -> None:
            debug("OUTER WAIT", subscriber, other_subscriber)
            async for message_, channel_ in subscriber:
                debug("INNER WAIT")
                # other_subscriber.stop()
                async for message_, channel_ in other_subscriber:
                    # Try to cancel the other subscriber to blow it up
                    subscriber.stop()

        task = asyncio.create_task(_boom())

        async def _emit_messages(*subscribers, channel) -> None:
            while True:
                for subscriber in subscribers:
                    debug("\n\nSENDIND TO", subscriber)
                    await subscriber(servo.pubsub.Message(text="foo"), channel)
                    debug("BACK", subscriber)
                debug("SENT")

        debug("GATHERING")
        await asyncio.gather(_emit_messages(subscriber, other_subscriber, channel=channel), task)
        # debug("emitting")
        # await _emit_messages(subscriber, other_subscriber, channel=channel)
        # debug("waiting on task")
        # await task
        # await asyncio.gather(*tasks)
        # await asyncio.gather()
        # with pytest.raises
        # other_subscriber
        # async def _iterate() -> None:
        #     async for message_, channel_ in subscriber:
        #         messages.append(message_)

        #         if len(messages) == 3:
        #             subscriber.stop()

        #     return messages

        # task = asyncio.create_task(_iterate())
        # other_subscriber = subscriber.exchange.create_subscriber('whatever', callback=lambda m, c: subscriber.stop())




@pytest.fixture
def publisher(exchange: servo.pubsub.Exchange, channel: servo.pubsub.Channel) -> servo.pubsub.Publisher:
    return servo.pubsub.Publisher(exchange=exchange, channel=channel)


class TestPublisher:
    async def test_calling_publishes_to_exchange(self, publisher: servo.pubsub.Publisher, mocker: pytest_mock.MockFixture) -> None:
        message = servo.pubsub.Message(text="foo")
        with servo.utilities.pydantic.extra(publisher.exchange):
            spy = mocker.spy(publisher.exchange, "publish")
            await publisher(message)
            spy.assert_called_once_with(message, publisher.channel)


class TestExchange:
    def test_starts_not_running(self, exchange: servo.pubsub.Exchange) -> None:
        assert not exchange.running

    async def test_start(self, exchange: servo.pubsub.Exchange) -> None:
        assert not exchange.running
        exchange.start()
        assert exchange.running
        await exchange.shutdown()

    def test_clear(self, exchange: servo.pubsub.Exchange) -> None:
        for i in range(3):
            name = f"channel-{i}"
            exchange.create_channel(name)
            exchange.create_publisher(name)
            exchange.create_subscriber(name)

        assert len(exchange.channels) == 3
        assert len(exchange._publishers) == 3
        assert len(exchange._subscribers) == 3
        exchange.clear()
        assert len(exchange.channels) == 0
        assert len(exchange._publishers) == 0
        assert len(exchange._subscribers) == 0

    async def test_shutdown(self, exchange: servo.pubsub.Exchange) -> None:
        for i in range(3):
            name = f"channel-{i}"
            exchange.create_channel(name)
            exchange.create_publisher(name)
            exchange.create_subscriber(name)

        exchange.start()
        assert exchange.running
        assert len(exchange.channels) == 3
        assert len(exchange._publishers) == 3
        assert len(exchange._subscribers) == 3

        await exchange.shutdown()

        assert not exchange.running
        assert len(exchange.channels) == 0
        assert len(exchange._publishers) == 0
        assert len(exchange._subscribers) == 0

    async def test_get_channel(self, exchange: servo.pubsub.Exchange) -> None:
        assert exchange.get_channel('whatever') is None
        channel = exchange.create_channel("whatever")
        assert channel is not None
        assert exchange.get_channel('whatever') == channel

    async def test_create_channel(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.create_channel("whatever")
        assert channel is not None
        assert channel.name == 'whatever'
        assert channel.exchange == exchange
        assert len(exchange.channels) == 1

    async def test_create_channel_names_must_be_unique(self, exchange: servo.pubsub.Exchange) -> None:
        exchange.create_channel("whatever")
        with pytest.raises(ValueError, match="A Channel named 'whatever' already exists"):
            exchange.create_channel("whatever")

    async def test_remove_publisher(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.create_channel("whatever")
        assert channel in exchange.channels
        channel = exchange.remove_channel(channel)
        assert channel not in exchange.channel

    async def test_publish(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        exchange.start()
        channel = exchange.create_channel("metrics")
        message = servo.pubsub.Message(text='Testing')

        event = asyncio.Event()
        callback = mocker.AsyncMock(side_effect=lambda m, c: event.set())
        subscriber = exchange.create_subscriber(channel.name)
        subscriber.callback = callback

        await exchange.publish(message, channel)
        await event.wait()
        callback.assert_awaited_once_with(message, channel)

    async def test_publish_to_channel_by_name(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        exchange.start()
        channel = exchange.create_channel("metrics")
        message = servo.pubsub.Message(text='Testing')

        event = asyncio.Event()
        callback = mocker.AsyncMock(side_effect=lambda m, c: event.set())
        subscriber = exchange.create_subscriber(channel.name)
        subscriber.callback = callback

        await exchange.publish(message, channel.name)
        await event.wait()
        callback.assert_awaited_once_with(message, channel)

    async def test_publish_to_unknown_channel_fails(self, exchange: servo.pubsub.Exchange) -> None:
        message = servo.pubsub.Message(text='Testing')
        with pytest.raises(ValueError, match="no such Channel: invalid"):
            await exchange.publish(message, "invalid")

    async def test_publish_when_not_running_enqueues(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.create_channel("metrics")
        message = servo.pubsub.Message(text='Testing')
        await exchange.publish(message, channel.name)
        assert exchange._queue.qsize() == 1

    async def test_create_publisher(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.create_channel("metrics")
        publisher = exchange.create_publisher(channel)
        assert publisher
        assert publisher.exchange == exchange
        assert publisher in exchange._publishers

    async def test_create_publisher_by_channel_name(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.create_channel("metrics")
        publisher = exchange.create_publisher(channel.name)
        assert publisher
        assert publisher in exchange._publishers

    async def test_create_publisher_creates_channels(self, exchange: servo.pubsub.Exchange) -> None:
        channel = exchange.get_channel("metrics")
        assert channel is None
        publisher = exchange.create_publisher('metrics')
        assert publisher
        assert publisher in exchange._publishers
        assert exchange.get_channel("metrics") is not None

    async def test_remove_publisher(self, exchange: servo.pubsub.Exchange) -> None:
        publisher = exchange.create_publisher('whatever')
        assert publisher
        assert publisher in exchange._publishers
        exchange.remove_publisher(publisher)
        assert publisher not in exchange._publishers

    async def test_create_subscriber(self, exchange: servo.pubsub.Exchange) -> None:
        subscriber = exchange.create_subscriber('whatever')
        assert subscriber
        assert subscriber.exchange == exchange
        assert subscriber in exchange._subscribers

    async def test_create_subscriber_with_dependency(self, exchange: servo.pubsub.Exchange) -> None:
        async def _dependency() -> None:
            ...

        exchange.start()
        subscriber = exchange.create_subscriber('whatever', until_done=_dependency())
        async for event in subscriber:
            # block forever unless the dependency intervenes
            ...

    async def test_create_subscriber_with_timeout(self, exchange: servo.pubsub.Exchange) -> None:
        exchange.start()
        subscriber = exchange.create_subscriber('whatever', timeout=0.01)
        async for event in subscriber:
            # block forever unless the timeout intervenes
            ...

    async def test_remove_subscriber(self, exchange: servo.pubsub.Exchange) -> None:
        subscriber = exchange.create_subscriber('whatever')
        assert subscriber
        assert subscriber in exchange._subscribers
        exchange.remove_subscriber(subscriber)
        assert subscriber not in exchange._subscribers

    async def test_publisher_to_subscriber(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        exchange.start()
        message = servo.pubsub.Message(text='Testing')

        event = asyncio.Event()
        callback = mocker.AsyncMock(side_effect=lambda m, c: event.set())
        subscriber = exchange.create_subscriber('metrics*')
        subscriber.callback = callback

        publisher = exchange.create_publisher("metrics.http.production")
        await publisher(message)
        await event.wait()
        callback.assert_awaited_once_with(message, publisher.channel)

    async def test_repr(self, exchange: servo.pubsub.Exchange) -> None:
        exchange.create_publisher('whatever')
        exchange.create_subscriber('whatever')
        assert repr(exchange) == "Exchange(running=False, channel_names=['whatever'], publisher_count=1, subscriber_count=1, queue_size=0)"

    async def test_subscribe_context_manager(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        # This is a little bit tricky. To ensure that the Subscriber is attached before the Publisher begins firing Messages
        # we use an Event to synchronize them and then gather them and compare the return values
        exchange.start()
        publisher = exchange.create_publisher("metrics.http.production")
        event = asyncio.Event()

        async def _publisher_func() -> List[servo.pubsub.Message]:
            # Wait for subscriber registration
            await event.wait()

            messages = []
            for i in range(10):
                message = servo.pubsub.Message(text=f'Test Message #{i}')
                await publisher(message)
                messages.append(message)

            return messages

        async def _subscriber_func() -> List[servo.pubsub.Message]:
            messages = []

            async with exchange.subscribe('metrics*') as subscription:
                # Trigger the Publisher to begin sending messages
                event.set()

                async for message, channel in subscription:
                    messages.append(message)

                    if len(messages) == 10:
                        subscription.cancel()

            return messages

        results = await asyncio.wait_for(
            asyncio.gather(_publisher_func(), _subscriber_func()),
            timeout=3.0
        )
        assert len(results) == 2
        assert len(results[0]) == 10
        assert len(results[1]) == 10
        assert list(map(operator.attrgetter("text"), results[0])) == [
            "Test Message #0",
            "Test Message #1",
            "Test Message #2",
            "Test Message #3",
            "Test Message #4",
            "Test Message #5",
            "Test Message #6",
            "Test Message #7",
            "Test Message #8",
            "Test Message #9",
        ]
        assert results[0] == results[1]

    async def test_current_context_in_callback(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        exchange.start()
        assert servo.pubsub.current_context() is None
        message = servo.pubsub.Message(text='Testing')

        event = asyncio.Event()
        current_context = None

        async def _callback(message: servo.pubsub.Message, channel: servo.pubsub.Channel) -> None:
            nonlocal current_context
            current_context = servo.pubsub.current_context()
            event.set()

        subscriber = exchange.create_subscriber('metrics*')
        subscriber.callback = _callback

        publisher = exchange.create_publisher("metrics.http.production")
        await publisher(message)
        await event.wait()
        assert current_context is not None
        assert current_context == (message, publisher.channel)
        assert current_context[1].exchange == exchange
        assert servo.pubsub.current_context() is None

    async def test_current_context_in_iterator(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockerFixture) -> None:
        exchange.start()
        publisher = exchange.create_publisher("metrics.http.production")
        message = servo.pubsub.Message(text='Some Message')
        event = asyncio.Event()
        current_context = None

        async def _publisher_func() -> None:
            # Wait for subscriber registration
            await event.wait()
            await publisher(message)

        async def _subscriber_func() -> None:
            nonlocal current_context

            async with exchange.subscribe('metrics*') as subscription:
                # Trigger the Publisher to begin sending messages
                event.set()

                async for message, channel in subscription:
                    current_context = servo.pubsub.current_context()
                    subscription.cancel()

        await asyncio.wait_for(
            asyncio.gather(_publisher_func(), _subscriber_func()),
            timeout=3.0
        )
        assert current_context is not None
        assert current_context == (message, publisher.channel)
        assert current_context[1].exchange == exchange
        assert servo.pubsub.current_context() is None

    async def test_iteration(self, exchange: servo.pubsub.Exchange, mocker: pytest_mock.MockFixture) -> None:
        channel = exchange.create_channel('whatever')
        messages = []

        async def _subscriber() -> None:
            async for message in exchange:
                messages.append(message)

                if len(messages) == 3:
                    exchange.stop()

        async def _publisher() -> None:
            for i in range(3):
                await exchange.publish(servo.pubsub.Message(text=f"Message: {i}"), channel)

        exchange.start()
        await task_graph(
            _publisher(),
            _subscriber(),
            timeout=5.0
        )
        assert messages

class HostObject(servo.pubsub.Mixin):
    async def _test_publisher_decorator(self, *, name: Optional[str] = None) -> None:
        @self.publish("metrics", name=name)
        async def _manual_publisher(publisher: servo.pubsub.Publisher) -> None:
            await publisher(servo.pubsub.Message(json={"throughput": "31337rps"}))
            await asyncio.sleep(30)

    async def _test_repeating_publisher_decorator(self) -> None:
        @self.publish("metrics", every="10ms")
        async def _repeating_publisher(publisher: servo.pubsub.Publisher) -> None:
            await publisher(servo.pubsub.Message(json={"throughput": "31337rps"}))

    async def _test_subscriber_decorator(self, callback, *, name: Optional[str] = None) -> None:
        @self.subscribe("metrics", name=name)
        async def _message_received(message: servo.pubsub.Message, channel: servo.pubsub.Channel) -> None:
            callback(message, channel)

class TestMixin:
    @pytest.fixture
    async def host_object(self) -> HostObject:
        host_object = HostObject()
        yield host_object
        if host_object.pubsub_exchange.running:
            await host_object.pubsub_exchange.shutdown()
        else:
            host_object.pubsub_exchange.clear()

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_init_with_pubsub_exchange(self) -> None:
        exchange = servo.pubsub.Exchange()
        obj = HostObject(pubsub_exchange=exchange)
        assert obj.pubsub_exchange == exchange

    async def test_exchange_property(self, host_object: HostObject) -> None:
        assert host_object.pubsub_exchange

    async def test_exchange_property_setter(self, host_object: HostObject, exchange: servo.pubsub.Exchange) -> None:
        assert host_object.pubsub_exchange
        assert host_object.pubsub_exchange != exchange
        host_object.pubsub_exchange = exchange
        assert host_object.pubsub_exchange == exchange

    async def test_publisher_decorator_repeating(self, host_object: HostObject) -> None:
        assert len(host_object.pubsub_exchange._publishers) == 0
        await host_object._test_repeating_publisher_decorator()
        assert len(host_object.pubsub_exchange._publishers) == 1
        assert host_object.pubsub_exchange._queue.qsize() == 0
        await asyncio.sleep(0.2)
        assert host_object.pubsub_exchange._queue.qsize() >= 10

    async def test_publisher_decorator_manual(self, host_object: HostObject) -> None:
        assert len(host_object.pubsub_exchange._publishers) == 0
        await host_object._test_publisher_decorator()
        assert len(host_object.pubsub_exchange._publishers) == 1
        assert host_object.pubsub_exchange._queue.qsize() == 0
        await asyncio.sleep(0.2)
        assert host_object.pubsub_exchange._queue.qsize() == 1

    async def test_publisher_context_manager(self, host_object: HostObject) -> None:
        assert len(host_object.pubsub_exchange._publishers) == 0
        async with host_object.publish('metrics') as publisher:
            assert publisher
            assert len(host_object.pubsub_exchange._publishers) == 1
            assert host_object.pubsub_exchange._queue.qsize() == 0
            await publisher(servo.pubsub.Message(text="context manager FTW!"))
            assert host_object.pubsub_exchange._queue.qsize() == 1

        assert len(host_object.pubsub_exchange._publishers) == 0
        assert host_object.pubsub_exchange._queue.qsize() == 1

    async def test_publisher_context_manager_rejects_every_arg(self, host_object: HostObject) -> None:
        with pytest.raises(TypeError, match='Cannot create repeating publisher when used as a context manager: `every` must be None'):
            async with host_object.publish('metrics', every="10s") as publisher:
                ...

    async def test_subscriber_decorator(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        event = asyncio.Event()
        stub = mocker.stub()
        stub.side_effect=lambda x,y: event.set()
        await host_object._test_subscriber_decorator(stub)
        host_object.pubsub_exchange.start()
        channel = host_object.pubsub_exchange.create_channel("metrics")
        message = servo.pubsub.Message(json={"throughput": "31337rps"})
        await host_object.pubsub_exchange.publish(message, channel)
        await event.wait()
        stub.assert_called_once_with(message, channel)

    async def test_subscriber_context_manager(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        host_object.pubsub_exchange.start()
        message = servo.pubsub.Message(json={"throughput": "31337rps"})
        channel = host_object.pubsub_exchange.create_channel("metrics")
        event = asyncio.Event()

        async def _publisher() -> None:
            await event.wait()
            await host_object.pubsub_exchange.publish(message, channel)

        async def _subscriber() -> None:
            async with host_object.subscribe('metrics') as subscriber:
                event.set()

                async for message, channel in subscriber:
                    stub(message, channel)
                    subscriber.cancel()

        await asyncio.wait_for(
            asyncio.gather(_publisher(), _subscriber()),
            timeout=3.0
        )
        stub.assert_called_once_with(message, channel)

    async def test_pubsub_between_decorators(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        notifications = []
        def _callback(message, channel) -> None:
            notification = f"Message #{len(notifications)} '{message.text}' (channel: '{channel.name}')"
            notifications.append(notification)

        await host_object._test_subscriber_decorator(_callback)
        await host_object._test_repeating_publisher_decorator()
        host_object.pubsub_exchange.start()

        await asyncio.sleep(0.2)
        assert len(notifications) > 10
        assert notifications[0:5] == [
            "Message #0 \'{\"throughput\": \"31337rps\"}\' (channel: 'metrics')",
            "Message #1 \'{\"throughput\": \"31337rps\"}\' (channel: 'metrics')",
            "Message #2 \'{\"throughput\": \"31337rps\"}\' (channel: 'metrics')",
            "Message #3 \'{\"throughput\": \"31337rps\"}\' (channel: 'metrics')",
            "Message #4 \'{\"throughput\": \"31337rps\"}\' (channel: 'metrics')",
        ]

    async def test_cancel_subscribers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        await host_object._test_subscriber_decorator(stub)
        await host_object._test_subscriber_decorator(stub, name="another_subscriber")
        with servo.utilities.pydantic.extra(host_object.pubsub_exchange):
            spy = mocker.spy(host_object.pubsub_exchange, "remove_subscriber")
            host_object.cancel_subscribers('_message_received')
            spy.assert_called_once()

            subscriber = spy.call_args.args[0]
            assert subscriber.cancelled
            assert subscriber not in host_object.pubsub_exchange._subscribers
            assert len(host_object.pubsub_exchange._subscribers) == 1
            assert len(host_object._subscribers_map) == 1
            assert host_object._subscribers_map['another_subscriber']

    async def test_cancel_all_subscribers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        await host_object._test_subscriber_decorator(stub, name="one_subscriber")
        await host_object._test_subscriber_decorator(stub, name="two_subscriber")
        await host_object._test_subscriber_decorator(stub, name="three_subscriber")
        assert len(host_object.pubsub_exchange._subscribers) == 3
        with servo.utilities.pydantic.extra(host_object.pubsub_exchange):
            spy = mocker.spy(host_object.pubsub_exchange, "remove_subscriber")
            host_object.cancel_subscribers()
            spy.assert_called()
            assert spy.call_count == 3

            for args in spy.call_args_list:
                subscriber, = args[0]
                assert subscriber.cancelled
                assert subscriber not in host_object.pubsub_exchange._subscribers

            assert len(host_object.pubsub_exchange._subscribers) == 0
            assert len(host_object._subscribers_map) == 0

    async def test_cancel_publishers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        await host_object._test_publisher_decorator()
        await host_object._test_publisher_decorator(name="another_publisher")
        assert len(host_object._publishers_map) == 2
        assert host_object._publishers_map['_manual_publisher']
        assert host_object._publishers_map['another_publisher']

        with servo.utilities.pydantic.extra(host_object.pubsub_exchange):
            spy = mocker.spy(host_object.pubsub_exchange, "remove_publisher")
            host_object.cancel_publishers('_manual_publisher')
            spy.assert_called_once()

            publisher = spy.call_args.args[0]
            assert subscriber not in host_object.pubsub_exchange._publishers
            assert len(host_object.pubsub_exchange._publishers) == 1
            assert len(host_object._publishers_map) == 1
            assert host_object._publishers_map['another_publisher']

    async def test_cancel_all_publishers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        await host_object._test_publisher_decorator(name="one_publisher")
        await host_object._test_publisher_decorator(name="two_publisher")
        await host_object._test_publisher_decorator(name="three_publisher")
        with servo.utilities.pydantic.extra(host_object.pubsub_exchange):
            spy = mocker.spy(host_object.pubsub_exchange, "remove_publisher")
            host_object.cancel_publishers()
            spy.assert_called()
            assert spy.call_count == 3

            for args in spy.call_args_list:
                publisher, = args[0]
                assert publisher not in host_object.pubsub_exchange._publishers

            assert len(host_object.pubsub_exchange._publishers) == 0
            assert len(host_object._publishers_map) == 0


class CountDownLatch:
    def __init__(self, count=1):
        self._count = count
        self._condition = asyncio.Condition()

    @property
    def count(self) -> int:
        return self._count

    async def decrement(self):
        async with self._condition:
            self._count -= 1
            if self._count <= 0:
                self._condition.notify_all()

    async def wait(self):
        async with self._condition:
            await self._condition.wait()


async def task_graph(task, *dependencies, timeout: Optional[servo.Duration] = None):
    async def _main_task():
        await latch.wait()
        await _run_task(task)

    async def _run_task(task):
        if asyncio.iscoroutinefunction(task):
            await task()
        elif asyncio.iscoroutine(task):
            await task
        else:
            task()

    async def _dependent_task(task):
        await latch.decrement()
        await _run_task(task)

    latch = CountDownLatch(len(dependencies))
    timeout_ = timeout and servo.Duration(timeout).total_seconds()
    await asyncio.wait_for(
        asyncio.gather(
            _main_task(),
            *list(map(_dependent_task, dependencies))
        ),
        timeout=timeout_
    )
