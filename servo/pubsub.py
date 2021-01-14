"""Provides asynchronous publisher / subscriber capabilities."""
from __future__ import annotations

import abc
import asyncio
import contextlib
import codecs
import datetime
import fnmatch
import functools
import json as json_
import re
import yaml as yaml_

from typing import Any, AsyncIterable, AsyncContextManager, Awaitable, Callable, Dict, Iterable, List, Optional, Pattern, Set, Tuple, Union

import pydantic
import servo.types


__all__ = [
    'BaseSubscription',
    'Callback',
    'Channel',
    'Exchange',
    'Message',
    'Metadata',
    'Mixin',
    'Publisher',
    'Subscriber',
    'Subscription',
]

Metadata = Dict[str, str]
ByteStream = Union[Iterable[bytes], AsyncIterable[bytes]]
MessageContent = Union[str, bytes, ByteStream]


class Message(pydantic.BaseModel):
    """A Message is information published to a Channel within an Exchange.

    Attributes:
        content: The content of the message.
        content_type: A MIME Type describing the message content encoding.
        created_at: The date and time when the message was created.
        metadata: Arbitrary string key/value metadata about the message.

    Args:
        content: Byte array of raw message content.
        content_type: MIME Type describing the message content encoding.
        text: String message content. Defaults content_type to `text/plain` if omitted.
        json: A JSON serializable object to set as message content. Defaults `content_type`
            to `application/json` if omitted.
        yaml: A YAML serializable object to set as message content. Defaults `content_type`
            to `application/x-yaml` if omitted.
    """
    content: bytes
    content_type: str
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    metadata: Metadata = {}

    # Private cache attributes
    _text: Optional[str] = pydantic.PrivateAttr(None)

    def __init__(
        self,
        content: Optional[MessageContent] = None,
        content_type: Optional[str] = None,
        text: Optional[str] = None,
        json: Optional[Any] = None,
        yaml: Optional[Any] = None,
        metadata: Metadata = {},
        **kwargs
    ) -> Message:
        if len(list(filter(None, [content, text, json, yaml]))) > 1:
            raise ValueError(f"only one argument of content, text, json, or yaml can be given")

        if text is not None and not isinstance(text, str):
            raise ValueError(f"Text Messages can only be created with `str` content: got '{text.__class__.__name__}'")

        if content is None:
            if text is not None:
                content = text.encode()
            elif json is not None:
                content = json_.dumps(json)
            elif yaml is not None:
                content = yaml_.dump(yaml)

        if content_type is None:
            if text is not None:
                content_type = "text/plain"
            elif json is not None:
                content_type = "application/json"
            elif yaml is not None:
                content_type = "application/x-yaml"

        super().__init__(content=content, content_type=content_type, metadata=metadata)

    @property
    def text(self) -> str:
        """Return a representation of the message body decoded as UTF-8 text."""
        if self._text is None:
            content = self.content
            if not content:
                self._text = ""
            else:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
                text = decoder.decode(content)
                self._text = "".join([decoder.decode(self.content), decoder.decode(b"", True)])

        return self._text

    def json(self) -> Any:
        """Return a representation of the message content deserialized as JSON."""
        return json_.loads(self.content)

    def yaml(self) -> Any:
        """Return a representation of the message content deserialized as YAML."""
        return yaml_.load(self.content)


ChannelName = pydantic.constr(
    strip_whitespace=True,
    min_length=1,
    max_length=253,
    regex="^[0-9a-zA-Z]([0-9a-zA-Z\\.-])*[0-9A-Za-z]$",
)


class Channel(pydantic.BaseModel):
    """A Channel groups related Messages within an Exchange.

    Channel names conform to [RFC 1123](https://tools.ietf.org/html/rfc1123) and must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-' or '.'
        * start with an alphanumeric character
        * end with an alphanumeric character

    Attributes:
        name: The unique name of the Channel within the Exchange.
        description: An optional supplemental description of the Channel.
        created_at: The date and time that the Channel was created.
        exchange: The pub/sub Exchange that the Channel belongs to.
    """
    name: ChannelName
    description: Optional[str] = None
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    exchange: Exchange

    async def publish(self, message: Message) -> None:
        """Publish a Message into the Channel."""
        await self.exchange.publish(message, self)

    def __hash__(self): # noqa: D105
        return hash(
            (
                self.name,
            )
        )


class Exchange(pydantic.BaseModel):
    """An Exchange facilitates the publication and subscription of Messages in Channels."""
    _channels: Set[Channel] = pydantic.PrivateAttr(set())
    _publishers: List[Publisher] = pydantic.PrivateAttr([])
    _subscribers: List[Subscriber] = pydantic.PrivateAttr([])
    _queue: asyncio.Queue = pydantic.PrivateAttr(default_factory=asyncio.Queue)
    _queue_processor: Optional[asyncio.Task] = pydantic.PrivateAttr(None)

    def start(self) -> None:
        """Start exchanging Messages between Publishers and Subscribers."""
        if self.is_running:
            raise RuntimeError("the Exchange is already running")
        self._queue_processor = asyncio.create_task(self._process_queue())

    async def shutdown(self) -> None:
        """Shutdown the Exchange by processing all Messages and clearing all child objects."""
        if not self.is_running:
            raise RuntimeError("the Exchange is not running")
        await self._queue.join()
        self._queue_processor.cancel()
        await asyncio.gather(self._queue_processor, return_exceptions=True)

        # Purge all our children to break object cycles
        self.channels.clear()
        self._publishers.clear()
        self._subscribers.clear()

    async def _process_queue(self) -> None:
        while True:
            message, channel = await self._queue.get()
            if message is None:
                # Exit condition
                break

            # Notify subscribers in a new task to avoid blocking the queue
            asyncio.create_task(_deliver_message_to_subscribers(message, channel, self._subscribers))

            self._queue.task_done()

    @property
    def is_running(self) -> bool:
        """Return True if the Exchange is processing Messages."""
        return self._queue_processor is not None and not self._queue_processor.done()

    def __repr_args__(self) -> pydantic.ReprArgs:
        return [
            ('running', self.is_running),
            ('channel_names', list(map(lambda c: c.name, self.channels))),
            ('publisher_count', len(self._publishers)),
            ('subscriber_count', len(self._subscribers)),
            ('queue_size', self._queue.qsize()),
        ]

    @property
    def channels(self) -> Set[Channel]:
        """Return the set of Channels in the Exchange."""
        return self._channels.copy()

    def get_channel(self, name: str) -> Optional[Channel]:
        """Return a Channel by name or `None` if no such Channel exists."""
        return next(filter(lambda m: m.name == name, self._channels), None)

    def create_channel(self, name: str, description: Optional[str] = None) -> Channel:
        """Create a new Channel in the Exchange.

        Args:
            name: A unique name for the Channel.
            description: An optional textual description about the Channel.

        Raises:
            ValueError: Raised if a Channel already exists with the name given.

        Returns:
            A newly created Channel object.
        """
        if self.get_channel(name) is not None:
            raise ValueError(f"A Channel named '{name}' already exists")
        channel = Channel(name=name, description=description, exchange=self)
        self._channels.add(channel)
        return channel

    async def publish(self, message: Message, channel: Union[Channel, str]) -> None:
        """Publish a Message to a Channel, notifying all Subscribers asynchronously.

        Note that Messages are delivered only when the Exchange is running. When stopped,
        they will remain enqueued.

        Args:
            message: The Message to publish.
            channel: The Channel or name of the Channel to publish the Message to.

        Raises:
            ValueError: Raised if the Channel specified does not exist in the Exchange.
        """
        channel_ = (
            self.get_channel(channel) if isinstance(channel, str) else channel
        )
        if channel_ is None:
            raise ValueError(f"no such Channel: {channel}")

        await self._queue.put(
            (message, channel_)
        )

    def create_publisher(self, channel: Union[Channel, str]) -> Publisher:
        """Create a new Publisher bound to a Channel.

        If the `channel` does not already exist in the Exchange it is automatically created.

        Args:
            channel: The Channel or name of the Channel to bind the Publisher to.

        Raises:
            TypeError: Raised if the `channel` argument is of the wrong type.
            ValueError: Raised if the Channel is not valid.

        Returns:
            A new Publisher object bound to the desired Channel and the Exchange.
        """
        if not isinstance(channel, (Channel, str)):
            raise TypeError(f"channel argument must be a `str` or `Channel`, got: {channel.__class__.__name__}")

        channel_ = (
            self.get_channel(channel) if isinstance(channel, str) else channel
        )
        if channel_ is None:
            channel_ = self.create_channel(channel)

        publisher = Publisher(exchange=self, channel=channel_)
        self._publishers.append(publisher)
        return publisher

    def remove_publisher(self, publisher: Publisher) -> None:
        """Remove a Publisher from the Exchange.

        Args:
            publisher: The Publisher to remove.

        Raises:
            ValueError: Raised if the given publisher is not in the Exchange.
        """
        self._publishers.remove(publisher)

    @contextlib.asynccontextmanager
    async def subscribe(self, selector: Selector) -> AsyncContextManager[Subscriber]:
        """An async context manager for subscribing to Messages in the Exchange.

        A Subscriber is created, yielded to the caller, and deleted upon return.

        Usage:
            ```
            async with exchange.subscribe("metrics.*") as subscription:
                async for message, channel in subscription:
                    ...
            ```

        Yields:
            Subscriber: The block temporary subscriber.
        """
        subscriber = self.create_subscriber(selector)
        try:
            yield subscriber
        finally:
            self._subscribers.remove(subscriber)

    def create_subscriber(self, selector: Selector, *, callback: Optional[Callback] = None) -> Subscriber:
        """Create and return a new Subscriber with the given selector.

        Args:
            selector: A string or regular expression pattern matching Channels of interest.
            callback: An optional callback for processing Messages received.

        Returns:
            A new Subscriber object listening for Messages.
        """
        subscription = Subscription(selector=selector)
        subscriber = Subscriber(exchange=self, subscription=subscription, callback=callback)
        self._subscribers.append(subscriber)
        return subscriber

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        """Remove a Subscriber from the Exchange.

        Args:
            subscriber: The Subscriber to remove.

        Raises:
            ValueError: Raised if the given subscriber is not in the Exchange.
        """
        self._subscribers.remove(subscriber)


Channel.update_forward_refs()


class BaseSubscription(abc.ABC, pydantic.BaseModel):
    """Abstract base class for pub/sub subscriptions.

    Subscriptions are responsible for matching the Messages and Channels
    that a Subscriber is interested in.
    """

    @abc.abstractmethod
    def matches(self, message: Message, channel: Channel) -> bool:
        """Return True if the message and/or channel given match the subscription.

        Args:
            message: A Message to evaluate.
            channel: The Channel that the Message was published to.
        """


Selector = Union[pydantic.StrictStr, Pattern]


class Subscription(BaseSubscription):
    """A Subscription describes Channels and Messages interesting to a Subscriber.

    Subscriptions match against Channel names via a selector. Selectors can be
    literal string values, Unix shell glob patterns, or regular expressions.

    Unix shell glob string patterns may include wildcards characters:

        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq

    Selector string values in regex syntax of '/pattern/' are compiled into regex
    patterns.

    Example selectors:

        * `metrics.http.production` - Match only the channel named "metrics.http.production".
        * `metrics.*` - Match any channel prefixed with "metrics.".
        * `/metrics.(http|mqtt|dns).*/` - Match channels via regex prefixed with "metrics.http",
            "metrics.mqtt", or "metrics.dns".

    Attributes:
        selector: A string glob or regular expression pattern for matching Channels.
    """
    selector: Selector

    @pydantic.validator('selector', pre=True)
    def _expand_selector_regex(cls, v: str) -> Union[str, Pattern]:
        if isinstance(v, str) and v.startswith('/') and v.endswith('/'):
            return re.compile(v[1:-1])

        return v

    def matches(self, message: Message, channel: Channel) -> bool:
        """Return True if the message and/or channel given match the subscription.

        Args:
            message: A Message to evaluate.
            channel: The Channel that the Message was published to.
        """
        selector = self.selector
        if isinstance(selector, re.Pattern):
            return bool(selector.fullmatch(channel.name))
        elif isinstance(selector, str):
            return fnmatch.fnmatch(channel.name, selector)

        raise ValueError(f"unknown selector type: {selector.__class__.__name__}")


Callback = Callable[[Message, Channel], Union[None, Awaitable[None]]]


class Subscriber(pydantic.BaseModel):
    """A Subscriber consumes relevant Messages published to an Exchange.

    Subscribers can either invoke a callback when a new Message is available or
    can be used as an asynchronous iterator to process Messages. The `stop` method
    will halt message processing and stop async iteration.

    Subscribers are asynchronously callable for notification of the publication of new Messages.

    Attributes:
        exchange: The pub/sub exchange that the Subscriber belongs to.
        subscription: A descriptor of the types of Messages that the Subscriber is interested in.
        callback: An optional callable to be invoked whben the Subscriber is notified of new Messages.

     Usage:
            ```
            # Processing via a callback
            async def _my_callback(message: Message, channel: Channel) -> None:
                print(f"Notified of a new Message: {message}, {channel}")

            subscriber = Subscriber(exchange=exchange, subscription=subscription, callback=callback)

            # Processing via async iteration
            subscriber = Subscriber(exchange=exchange, subscription=subscription)

            async for message, channel in subscriber:
                print(f"Notified of a new Message: {message}, {channel}")

                # Break out of processing
                subscriber.stop()
            ```
    """
    exchange: Exchange
    subscription: Subscription
    callback: Optional[Callback]

    # supports usage as an async iterator
    _queue: asyncio.Queue = pydantic.PrivateAttr(default_factory=asyncio.Queue)
    _event: asyncio.Event = pydantic.PrivateAttr(default_factory=asyncio.Event)

    # TODO: should we put a limit on the queue size and warn on overflow?

    def stop(self) -> None:
        """Stop the subscriber from processing any further Messages."""
        self._event.set()

    @property
    def is_running(self) -> bool:
        """Return True if the subscriber is processing Messages."""
        return not self._event.is_set()

    async def __call__(self, message: Message, channel: Channel) -> None:
        if not self.is_running:
            servo.logger.warning(f"ignoring call to stopped Subscriber: {self}")
            return

        if self.subscription.matches(message, channel):
            if self.callback:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(message, channel)
                else:
                    self.callback(message, channel)
            else:
                # enqueue for processing via async iteration
                await self._queue.put(
                    (message, channel)
                )

    def __aiter__(self) -> Subscriber:
        if self.callback:
            raise RuntimeError("Subscriber objects with a callback cannot be used as an async iterator")
        return self

    async def __anext__(self) -> Tuple[Message, Channel]:
        if not self.is_running:
            raise StopAsyncIteration

        message, channel = await self._queue.get()
        if message is None:
            raise StopAsyncIteration

        return (message, channel)

    class Config:
        arbitrary_types_allowed = True



class Publisher(pydantic.BaseModel):
    """A Publisher broadcasts Messages to a Channel in an Exchange.

    Publishers are asynchronously callable to publish a Message.

    Attributes:
        exchange: The pub/sub Exchange that the Publisher belongs to.
        channel: The Channel that the Publisher publishes Messages to.
    """
    exchange: Exchange
    channel: Channel

    async def __call__(self, message: Message) -> None:
        await self.exchange.publish(message, self.channel)


class Mixin(pydantic.BaseModel):
    """Provides a simple pub/sub stack for subclasses."""
    __private_attributes__ = {
        '_exchange': pydantic.PrivateAttr(default_factory=Exchange),
        '_publisher_tasks': pydantic.PrivateAttr({})
    }

    @property
    def exchange(self) -> Exchange:
        """Return the pub/sub Exchange."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: Exchange):
        """Set the pub/sub Exchange."""
        self._exchange = exchange

    # TODO: Do we need/want cancellation for decorated subscribers?
    # TODO: Let you set the async task names?

    def subscriber(self, selector: Selector) -> None:
        """Transform a function into a pub/sub Subscriber.

        The decorated function may be synchronous or asynchronous but must accept
        two arguments: `message: Message, channel: Channel`.

        Usage:
            ```
            def some_method(self) -> None:
                @self.subscriber("metrics.*")
                def _message_received(message: Message, channel: Channel) -> None:
                    print(f"Notified of a new Message: {message}, {channel}")
            ```
        """
        def decorator(fn):
            self.exchange.create_subscriber(selector, callback=fn)

        return decorator

    def publisher(self, channel: Union[Channel, str], *, every: Optional[servo.types.DurationDescriptor] = None) -> None:
        """Transform a function into a pub/sub Publisher.

        When the `every` argument is not None, the publisher sleeps for the given duration to support
        publishing messages on a repeating time interval.

        The decorated function must be asynchronous and accept a single argument: `publisher: Publisher`.

        Args:
            channel: The Channel or the name of the Channel to bind the Publisher to.
            every: Optional Duration descriptor specifying how often the Publisher is to be awakened.

        Usage:
            ```
            # Using a repeating interval
            def repeating_example(self) -> None:
                @self.publisher("metrics", every="15s")
                async def _publish_metrics(publisher: Publisher) -> None:
                    await publisher(Message(json={"throughput": "31337rps"}))

            # Manually sleeping the publisher loop
            def manual_example(self) -> None:
                @self.publisher("metrics")
                async def _publish_metrics(publisher: Publisher) -> None:
                    await publisher(Message(json={"throughput": "31337rps"}))

                    seconds_to_sleep = random.randint(1, 15)
                    await asyncio.sleep(seconds_to_sleep)
            ```
        """
        def decorator(fn) -> None:
            if not asyncio.iscoroutinefunction(fn):
                raise ValueError("decorated function must be asynchronous")

            publisher = self.exchange.create_publisher(channel)
            if every is not None:
                duration = every if isinstance(every, servo.Duration) else servo.Duration(every)
            else:
                duration = None

            @functools.wraps(fn)
            async def _repeating_publisher() -> None:
                while True:
                    await fn(publisher)
                    if duration is not None:
                        await asyncio.sleep(duration.total_seconds())

            task = asyncio.create_task(_repeating_publisher())
            self._publisher_tasks.add(task)

        return decorator


async def _deliver_message_to_subscribers(message: Message, channel: Channel, subscribers: List[Subscriber]) -> None:
    results = await asyncio.gather(
        *list(
            map(
                lambda subscriber: subscriber(message, channel),
                subscribers
            )
        ),
        return_exceptions=True
    )

    # Log failures without aborting
    with servo.logger.catch(message="Subscriber raised exception"):
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                raise result
