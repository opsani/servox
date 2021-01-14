import asyncio
import re
import pytest
import pytest_mock
import pydantic
import servo
import servo.pubsub
import servo.utilities.pydantic

from typing import List, Optional

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
    if exchange.is_running:
        await exchange.shutdown()
    else:
        exchange.clear()

@pytest.fixture
def channel(exchange: servo.pubsub.Exchange) -> servo.pubsub.Channel:
    return servo.pubsub.Channel(name="metrics", exchange=exchange)

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

    async def test_publish(self, channel: servo.pubsub.Channel, mocker: pytest_mock.MockFixture) -> None:
        message = servo.pubsub.Message(text="foo")
        with servo.utilities.pydantic.extra(channel.exchange):
            spy = mocker.spy(channel.exchange, "publish")
            await channel.publish(message)
            spy.assert_called_once_with(message, channel)

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
        assert subscription.matches(message, metrics_channel)
        assert not subscription.matches(message, other_channel)

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
        assert subscription.matches(message, metrics_channel) == matches

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
        assert subscription.matches(message, channel) == matches, f"expected regex pattern '{selector}' match of '{channel.name}' to == {matches}"


@pytest.fixture
def subscriber(exchange: servo.pubsub.Exchange, subscription: servo.pubsub.Subscription) -> servo.pubsub.Subscriber:
    return servo.pubsub.Subscriber(exchange=exchange, subscription=subscription)

@pytest.fixture
def subscription() -> servo.pubsub.Subscription:
    return servo.pubsub.Subscription(selector="metrics*")


class TestSubscriber:
    def test_is_running_on_create(self, subscriber: servo.pubsub.Subscriber) -> None:
        assert subscriber.is_running

    class TestCallback:
        async def test_cannot_use_as_iterator(self, subscriber: servo.pubsub.Subscriber, mocker: pytest_mock.MockerFixture) -> None:
            subscriber.callback = mocker.Mock()
            with pytest.raises(RuntimeError, match="Subscriber objects with a callback cannot be used as an async iterator"):
                async for message, channel in subscriber:
                    ...

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

    class TestAsyncIterator:
        async def test_matched_messages_are_enqueued(self, subscriber: servo.pubsub.Subscriber) -> None:
            message = servo.pubsub.Message(text="foo")
            channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)
            await subscriber(message, channel)
            assert subscriber._queue.qsize() == 1

        async def test_non_matched_messages_are_not_enqueued(self, subscriber: servo.pubsub.Subscriber) -> None:
            message = servo.pubsub.Message(text="foo")
            channel = servo.pubsub.Channel(name="not.gonna.match", exchange=subscriber.exchange)
            await subscriber(message, channel)
            assert subscriber._queue.qsize() == 0

        async def test_consuming_messages_via_async_iterator(self, subscriber: servo.pubsub.Subscriber) -> None:
            message = servo.pubsub.Message(text="foo")
            channel = servo.pubsub.Channel(name="metrics", exchange=subscriber.exchange)
            for _ in range(3):
                await subscriber(message, channel)
            assert subscriber._queue.qsize() == 3

            messages = []
            async for message_, channel_ in subscriber:
                assert message_ == message
                assert channel_ == channel
                messages.append(message_)

                if len(messages) == 3:
                    subscriber.stop()

            assert len(messages) == 3


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
        assert not exchange.is_running

    async def test_start(self, exchange: servo.pubsub.Exchange) -> None:
        assert not exchange.is_running
        exchange.start()
        assert exchange.is_running
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
        assert exchange.is_running
        assert len(exchange.channels) == 3
        assert len(exchange._publishers) == 3
        assert len(exchange._subscribers) == 3

        await exchange.shutdown()

        assert not exchange.is_running
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
        assert len(exchange.channels) == 1

    async def test_create_channel_names_must_be_unique(self, exchange: servo.pubsub.Exchange) -> None:
        exchange.create_channel("whatever")
        with pytest.raises(ValueError, match="A Channel named 'whatever' already exists"):
            exchange.create_channel("whatever")

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
        assert subscriber in exchange._subscribers

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
                        subscription.stop()

            return messages

        results = await asyncio.wait_for(
            asyncio.gather(_publisher_func(), _subscriber_func()),
            timeout=3.0
        )
        import operator
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

class HostObject(servo.pubsub.Mixin):
    async def _test_publisher_decorator(self, *, name: Optional[str] = None) -> None:
        @self.publisher("metrics", name=name)
        async def _manual_publisher(publisher: servo.pubsub.Publisher) -> None:
            await publisher(servo.pubsub.Message(json={"throughput": "31337rps"}))
            await asyncio.sleep(30)

    async def _test_repeating_publisher_decorator(self) -> None:
        @self.publisher("metrics", every="10ms")
        async def _repeating_publisher(publisher: servo.pubsub.Publisher) -> None:
            await publisher(servo.pubsub.Message(json={"throughput": "31337rps"}))

    async def _test_subscriber_decorator(self, callback, *, name: Optional[str] = None) -> None:
        @self.subscriber("metrics", name=name)
        async def _message_received(message: servo.pubsub.Message, channel: servo.pubsub.Channel) -> None:
            callback(message, channel)

class TestMixin:
    @pytest.fixture
    async def host_object(self) -> HostObject:
        host_object = HostObject()
        yield host_object
        if host_object.exchange.is_running:
            await host_object.exchange.shutdown()
        else:
            host_object.exchange.clear()

    async def test_exchange_property(self, host_object: HostObject) -> None:
        assert host_object.exchange

    async def test_exchange_property_setter(self, host_object: HostObject, exchange: servo.pubsub.Exchange) -> None:
        assert host_object.exchange
        assert host_object.exchange != exchange
        host_object.exchange = exchange
        assert host_object.exchange == exchange

    async def test_publisher_decorator_repeating(self, host_object: HostObject) -> None:
        assert len(host_object.exchange._publishers) == 0
        await host_object._test_repeating_publisher_decorator()
        assert len(host_object.exchange._publishers) == 1
        assert host_object.exchange._queue.qsize() == 0
        await asyncio.sleep(0.2)
        assert host_object.exchange._queue.qsize() >= 10

    async def test_publisher_decorator(self, host_object: HostObject) -> None:
        assert len(host_object.exchange._publishers) == 0
        await host_object._test_publisher_decorator()
        assert len(host_object.exchange._publishers) == 1
        assert host_object.exchange._queue.qsize() == 0
        await asyncio.sleep(0.2)
        assert host_object.exchange._queue.qsize() == 1

    async def test_subscriber_decorator(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        await host_object._test_subscriber_decorator(stub)
        host_object.exchange.start()
        channel = host_object.exchange.create_channel("metrics")
        message = servo.pubsub.Message(json={"throughput": "31337rps"})
        await host_object.exchange.publish(message, channel)
        await asyncio.sleep(0.2)
        stub.assert_called_once_with(message, channel)

    async def test_pubsub_between_decorators(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        notifications = []
        def _callback(message, channel) -> None:
            notification = f"Message #{len(notifications)} '{message.text}' (channel: '{channel.name}')"
            notifications.append(notification)

        await host_object._test_subscriber_decorator(_callback)
        await host_object._test_repeating_publisher_decorator()
        host_object.exchange.start()

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
        with servo.utilities.pydantic.extra(host_object.exchange):
            spy = mocker.spy(host_object.exchange, "remove_subscriber")
            host_object.cancel_subscribers('_message_received')
            spy.assert_called_once()

            subscriber = spy.call_args.args[0]
            assert not subscriber.is_running
            assert subscriber not in host_object.exchange._subscribers
            assert len(host_object.exchange._subscribers) == 1
            assert len(host_object._subscribers_map) == 1
            assert host_object._subscribers_map['another_subscriber']


    async def test_cancel_all_subscribers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        await host_object._test_subscriber_decorator(stub, name="one_subscriber")
        await host_object._test_subscriber_decorator(stub, name="two_subscriber")
        await host_object._test_subscriber_decorator(stub, name="three_subscriber")
        assert len(host_object.exchange._subscribers) == 3
        with servo.utilities.pydantic.extra(host_object.exchange):
            spy = mocker.spy(host_object.exchange, "remove_subscriber")
            host_object.cancel_subscribers()
            spy.assert_called()
            assert spy.call_count == 3

            for args in spy.call_args_list:
                subscriber, = args[0]
                assert not subscriber.is_running
                assert subscriber not in host_object.exchange._subscribers

            assert len(host_object.exchange._subscribers) == 0
            assert len(host_object._subscribers_map) == 0

    async def test_cancel_publishers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        stub = mocker.stub()
        await host_object._test_subscriber_decorator(stub)
        await host_object._test_subscriber_decorator(stub, name="another_subscriber")
        with servo.utilities.pydantic.extra(host_object.exchange):
            spy = mocker.spy(host_object.exchange, "remove_subscriber")
            host_object.cancel_subscribers('_message_received')
            spy.assert_called_once()

            subscriber = spy.call_args.args[0]
            assert not subscriber.is_running
            assert subscriber not in host_object.exchange._subscribers
            assert len(host_object.exchange._subscribers) == 1
            assert len(host_object._subscribers_map) == 1
            assert host_object._subscribers_map['another_subscriber']

    async def test_cancel_publishers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        await host_object._test_publisher_decorator()
        await host_object._test_publisher_decorator(name="another_publisher")
        assert len(host_object._publishers_map) == 2
        assert host_object._publishers_map['_manual_publisher']
        assert host_object._publishers_map['another_publisher']

        with servo.utilities.pydantic.extra(host_object.exchange):
            spy = mocker.spy(host_object.exchange, "remove_publisher")
            host_object.cancel_publishers('_manual_publisher')
            spy.assert_called_once()

            publisher = spy.call_args.args[0]
            assert subscriber not in host_object.exchange._publishers
            assert len(host_object.exchange._publishers) == 1
            assert len(host_object._publishers_map) == 1
            assert host_object._publishers_map['another_publisher']

    async def test_cancel_all_publishers(self, host_object: HostObject, mocker: pytest_mock.MockFixture) -> None:
        await host_object._test_publisher_decorator(name="one_publisher")
        await host_object._test_publisher_decorator(name="two_publisher")
        await host_object._test_publisher_decorator(name="three_publisher")
        with servo.utilities.pydantic.extra(host_object.exchange):
            spy = mocker.spy(host_object.exchange, "remove_publisher")
            host_object.cancel_publishers()
            spy.assert_called()
            assert spy.call_count == 3

            for args in spy.call_args_list:
                publisher, = args[0]
                assert publisher not in host_object.exchange._publishers

            assert len(host_object.exchange._publishers) == 0
            assert len(host_object._publishers_map) == 0
