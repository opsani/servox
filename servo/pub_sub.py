"""Provides asynchronous publisher / subscriber capabilities."""
from __future__ import annotations

import asyncio
import contextlib
import codecs
import datetime
import functools
import json
import re
import yaml

from typing import Any, AsyncIterable, AsyncContextManager, Awaitable, Callable, Dict, Iterable, Optional, Pattern, Set, Union

import pydantic
import servo.types

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
        json: A JSON string or serializable object to set as message content. Defaults `content_type`
            to `application/json` if omitted.
        yaml: A YAML string or serializable object to set as message content. Defaults `content_type`
            to `application/x-yaml` if omitted.
    """
    content: bytes
    content_type: str
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    metadata: Metadata = {}

    # Private cache attributes
    _text: str = pydantic.PrivateAttr()

    def __init__(
        self,
        content: MessageContent,
        content_type: Optional[str] = None,
        text: Optional[str] = None,
        json: Optional[str] = None,
        yaml: Optional[str] = None,
        metadata: Metadata = {},
        **kwargs
    ) -> Message:
        # TODO: validate that input makes sense
        # TODO: serialize JSON and YAML inputs
        # TODO: Handle content_type defaults
        if not isinstance(text, str):
            raise ValueError(f"Text Messages can only be created with `str` content: got '{text.__class__.__name__}'")
        super().__init__(content=content, content_type=content_type, metadata=metadata)

    @property
    def content(self) -> bytes:
        """Return the message content as a byte array."""
        return self._content

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
        """Return a representation of the message content deserialized from JSON."""
        return json.load(self.content)

    def yaml(self) -> Any:
        """Return a representation of the message content deserialized from YAML."""
        return yaml.load(self.content)


Selector = Union[Pattern, str]


class Subscription(pydantic.BaseModel):
    """A Subscription describes Channels and Messages interesting to a Subscriber."""
    selector: Selector

    def matches(self, message: Message, channel: Channel) -> bool:
         """Return True if the message and/or channel given match the subscription."""
         # TODO: evaluate as a string glob or regex


Callback = Callable[[Message, Channel], Union[None, Awaitable[None]]]


class Subscriber(pydantic.BaseModel):
    """A Subscriber consumes relevant Messages published to an Exchange.

    Subscribers are asynchronously callable to notify them when a new Message
    has been published.
    """
    exchange: Exchange
    subscription: Subscription
    callback: Optional[Callback]

    # TODO: Validate the callback function arity

    async def __call__(self, message: Message, channel: Channel) -> None:
        if self.subscription.matches(message, channel):
            if self.callback:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(message, channel)
                else:
                    self.callback(message, channel)

            # TODO: yield to the async iterator

    # TODO: Add the async iterator support here.

    class Config:
        arbitrary_types_allowed = True


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
        exchange: The pub/sub Exchange that the Channel belongs
    """
    name: ChannelName
    description: Optional[str] = None
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    exchange: Exchange

    async def publish(self, message: Message) -> None:
        """Publish a Message into the Channel."""
        await self.exchange.publish(message, self)


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


class Exchange(pydantic.BaseModel):
    """An Exchange facilitates the publication and subscription of Messages in Channels."""
    channels: Set[Channel] = {}
    _publishers: Set[Subscriber] = pydantic.PrivateAttr({})
    _subscribers: Set[Subscriber] = pydantic.PrivateAttr({})
    _queue: asyncio.Queue = pydantic.PrivateAttr(default_factory=asyncio.Queue)
    _queue_processor: asyncio.Task = pydantic.PrivateAttr()

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

            results = asyncio.gather(
                *list(
                    map(
                        lambda subscriber: subscriber(message, channel),
                        self._subscribers
                    )
                ),
                return_exceptions=True
            )

            # Log failures without aborting
            with servo.logger.catch(message="Subscriber raised exception"):
                for result in results:
                    if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                        raise result

            self._queue.task_done()

    @property
    def is_running(self) -> bool:
        """Return True if the Exchange is processing Messages."""
        return self._queue_processor is not None and not self._queue_processor.done()

    def get_channel(self, name: str) -> Optional[Channel]:
        """Return a Channel by name or `None` if no such Channel exists."""
        return next(filter(lambda m: m.name == name, self.channels), None)

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
        self.channels.add(channel)
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

        self._queue.put(
            tuple(message, channel_)
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
        self._publishers.add(publisher)
        return publisher

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
        self._subscribers.add(subscriber)
        return subscriber

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        """Remove a Subscriber from the Exchange.

        Args:
            subscriber: The Subscriber to remove.

        Raises:
            KeyError: Raised if the given subscriber is not in the Exchange.
        """
        self._subscribers.remove(subscriber)


class Mixin(pydantic.BaseModel):
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

    def publisher(self, channel: Union[Channel, str], every: servo.types.DurationDescriptor) -> None:
        """Transform a function into a repeating pub/sub Publisher.

        The decorated function must be asynchronous and accept a single argument: `publisher: Publisher`.

        Args:
            channel: The Channel or the name of the Channel to bind the Publisher to.
            every: A Duration descriptor specifying how often the Publisher is to be awakened.

        Usage:
            ```
            def some_method(self) -> None:
                @self.publisher("metrics", every="15s")
                async def _publish_metrics(publisher: Publisher) -> None:
                    await publisher(Message(json={"throughput": "31337rps"}))
            ```
        """
        def decorator(fn) -> None:
            if not asyncio.iscoroutinefunction(fn):
                raise ValueError("decorated function must be asynchronous")

            publisher = self.exchange.create_publisher(channel)
            duration = every if isinstance(every, servo.Duration) else servo.Duration(every)

            @functools.wraps(fn)
            async def _repeating_publisher() -> None:
                while True:
                    await fn(publisher)
                    await asyncio.sleep(every.total_seconds())

            task = asyncio.create_task(_repeating_publisher())
            self._publisher_tasks.add(task)
