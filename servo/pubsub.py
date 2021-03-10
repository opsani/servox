"""Provides asynchronous publisher / subscriber capabilities."""
from __future__ import annotations

import abc
import asyncio
import contextlib
import contextvars
import codecs
import datetime
import fnmatch
import functools
import inspect
import json as json_
import random
import string
import re
import yaml as yaml_
import weakref

from typing import Any, AsyncIterable, AsyncContextManager, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional, Pattern, Set, Tuple, Union

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

    The `json` and `yaml` arguments respect an informal protocol. If the input argument
    responds to `json()` or `yaml()` methods respectively they are called to perform
    serialization.

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
                content = (
                    json.json() if (hasattr(json, 'json') and callable(json.json))
                    else json_.dumps(json)
                )
            elif yaml is not None:
                content = (
                    yaml.yaml() if (hasattr(yaml, 'yaml') and callable(yaml.yaml))
                    else yaml_.dump(yaml)
                )

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
    regex="^[0-9a-zA-Z]([0-9a-zA-Z\\.\\-_])*[0-9A-Za-z]$",
)


class _ExchangeChildModel(pydantic.BaseModel):
    _exchange: Exchange = pydantic.PrivateAttr(None)
    __slots__ = ('__weakref__')  # NOTE: Pydantic and weakref both use __slots__

    def __init__(self, *args, exchange: Exchange, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._exchange = weakref.ref(exchange)

    @property
    def exchange(self) -> Exchange:
        """The pub/sub Exchange that the object belongs to."""
        return self._exchange()


class Channel(_ExchangeChildModel):
    """A Channel groups related Messages within an Exchange.

    Channel names must:
        * contain no more than 253 characters
        * contain only lowercase alphanumeric characters, '-', '_', or '.'
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
    _closed: bool = pydantic.PrivateAttr(False)

    async def publish(self, message: Message) -> None:
        """Publish a Message into the Channel."""
        if self.closed:
            raise RuntimeError(f"Cannot publish messages to a closed Channel")
        await self.exchange.publish(message, self)

    @property
    def closed(self) -> bool:
        """Return True if the channel has been closed and can no longer receive messages."""
        return self._closed

    async def close(self) -> None:
        """Close the channel.

        Closing a channel will cancel any exclusive Subscribers. Exclusive Subscribers are
        subscribed with the literal channel name and not through a pattern.
        """
        if self.closed:
            raise RuntimeError("Channel is already closed")

        self._closed = True

        for subscriber in self.exchange._subscribers_to_channel(self, exclusive=True):
            subscriber.cancel()

    def stop(self) -> None:
        """Stop the current async iterator.

        The iterator to be stopped is determined by the current iteration scope.
        Calling stop on a parent iterator scope will trigger a `RuntimeError`.

        Raises:
            RuntimeError: Raised if there is not an active iterator or the receiver
                is not being iterated in the local scope.
        """
        iterator = _current_iterator()
        if iterator is not None:
            # Find subscribers to this channel
            subscribers = self.exchange._subscribers_to_channel(self, exclusive=True)
            if iterator.subscriber not in subscribers:
                raise RuntimeError(f"Attempted to stop an inactive iterator")
            iterator.stop()
        else:
            raise RuntimeError("Attempted to stop outside of an iterator")

    def __aiter__(self):  # noqa: D105
        if self.closed:
            raise RuntimeError(f"Cannot iterate messages in a closed Channel")
        subscriber = self.exchange.create_subscriber(self.name)
        iterator: _Iterator = subscriber.__aiter__()
        iterator.yield_channel = False
        return iterator

    def __hash__(self): # noqa: D105
        return hash(
            (
                self.name,
            )
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name == other

        return super().__eq__(other)


_current_context_var = contextvars.ContextVar("servo.pubsub.current_message", default=None)


def current_message() -> Optional[Tuple[Message, Channel]]:
    """Return the Message and Channel for the current execution context, if any.

    The context is set upon entry into a Subscriber and restored to its previous
    state upon exit. If `current_message()` is not `None`, then the currently
    executing operation was triggered by a pub/sub Message.

    The value is managed by a contextvar and is concurrency safe.
    """
    return _current_context_var.get()


class Exchange(pydantic.BaseModel):
    """An Exchange facilitates the publication and subscription of Messages in Channels.

    Exchange objects are asynchronously iterable and will yield every Message published.
    """
    _channels: Set[Channel] = pydantic.PrivateAttr(set())
    _publishers: List[Publisher] = pydantic.PrivateAttr([])
    _subscribers: List[Subscriber] = pydantic.PrivateAttr([])
    _queue: asyncio.Queue = pydantic.PrivateAttr(default_factory=asyncio.Queue)
    _queue_processor: Optional[asyncio.Task] = pydantic.PrivateAttr(None)
    __slots__ = ('__weakref__')  # NOTE: Pydantic and weakref both use __slots__

    def start(self) -> None:
        """Start exchanging Messages between Publishers and Subscribers."""
        if self.running:
            raise RuntimeError("the Exchange is already running")
        self._queue_processor = asyncio.create_task(self._process_queue())

    def clear(self) -> None:
        """Clear the Exchange by discarding all channels, publishers, and subscribers."""
        self._channels.clear()
        self._publishers.clear()
        self._subscribers.clear()

    async def shutdown(self) -> None:
        """Shutdown the Exchange by processing all Messages and clearing all child objects."""
        if not self.running:
            raise RuntimeError("the Exchange is not running")
        await self._queue.join()
        self._queue_processor.cancel()
        await asyncio.gather(self._queue_processor, return_exceptions=True)
        self.clear()

    def stop(self) -> None:
        """Stop the current async iterator.

        The iterator to be stopped is determined by the current iteration scope.
        Calling stop on a parent iterator scope will trigger a `RuntimeError`.

        Raises:
            RuntimeError: Raised if there is not an active iterator or the receiver
                is not being iterated in the local scope.
        """
        iterator = _current_iterator()
        if iterator is not None:
            subscribers = list(filter(lambda s: s.subscription.selector == '*', self._subscribers))
            if iterator.subscriber not in subscribers:
                raise RuntimeError(f"Attempted to stop an inactive iterator")
            iterator.stop()
        else:
            raise RuntimeError("Attempted to stop outside of an iterator")

    def __aiter__(self):  # noqa: D105
        if not self.running:
            raise RuntimeError(f"Cannot iterate messages in an Exchange that is not running")
        subscriber = self.create_subscriber('*')
        iterator: _Iterator = subscriber.__aiter__()
        return iterator

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
    def running(self) -> bool:
        """Return True if the Exchange is processing Messages."""
        return self._queue_processor is not None and not self._queue_processor.done()

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

    def remove_channel(self, channel: Channel) -> None:
        """Remove a Channel from the Exchange.

        Args:
            channel: The Channel to remove.

        Raises:
            ValueError: Raised if the given Channel is not in the Exchange.
        """
        self._channels.remove(channel)

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

    def create_publisher(self, *channels: List[Union[Channel, str]]) -> Publisher:
        """Create a new Publisher bound to one or more Channels.

        If a Channel referenced does not already exist in the Exchange it is automatically created.

        Args:
            channels: The list of Channel objects or names to bind the Publisher to.

        Raises:
            TypeError: Raised if the `channels` argument contains an object of the wrong type.
            ValueError: Raised if the Channel is not valid.

        Returns:
            A new Publisher object bound to the desired Channels and the Exchange.
        """
        for channel in channels:
            if not isinstance(channel, (Channel, str)):
                raise TypeError(f"channel argument must be a `str` or `Channel`, got: {channel.__class__.__name__}")

        channels_ = []
        for channel in channels:
            channel_ = (
                self.get_channel(channel) if isinstance(channel, str) else channel
            )
            if channel_ is None:
                channel_ = self.create_channel(channel)

            channels_.append(channel_)

        publisher = Publisher(exchange=self, channels=channels_)
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
    async def subscribe(
        self,
        selector: Selector,
        *,
        timeout: Optional[servo.types.DurationDescriptor] = None,
        until_done: Optional[servo.types.Futuristic] = None
    ) -> AsyncContextManager[Subscriber]:
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
        subscriber = self.create_subscriber(selector, timeout=timeout, until_done=until_done)
        try:
            yield subscriber
        finally:
            self._subscribers.remove(subscriber)

    def create_subscriber(
        self,
        selector: Selector,
        *,
        callback: Optional[Callback] = None,
        timeout: Optional[servo.types.DurationDescriptor] = None,
        until_done: Optional[servo.types.Futuristic] = None
    ) -> Subscriber:
        """Create and return a new Subscriber with the given selector.

        Args:
            selector: A string or regular expression pattern matching Channels of interest.
            callback: An optional callback for processing Messages received.
            timeout: An optional duration description for specifying when to cancel the request.
            until_done: An optional future to to tie the subscription lifetime to.

        Returns:
            A new Subscriber object listening for Messages.
        """
        subscription = Subscription(selector=selector)
        subscriber = Subscriber(exchange=self, subscription=subscription, callback=callback)
        self._subscribers.append(subscriber)

        # Handle async affordances
        def _cancelizer(*args, **kwargs) -> None:
            if not subscriber.cancelled:
                subscriber.cancel()

        if until_done:
            future = asyncio.ensure_future(until_done)
            future.add_done_callback(_cancelizer)

        if timeout is not None:
            asyncio.get_event_loop().call_later(
                servo.Duration(timeout).total_seconds(),
                _cancelizer
            )

        return subscriber

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        """Remove a Subscriber from the Exchange.

        Args:
            subscriber: The Subscriber to remove.

        Raises:
            ValueError: Raised if the given subscriber is not in the Exchange.
        """
        self._subscribers.remove(subscriber)

    def _subscribers_to_channel(self, channel: Channel, *, exclusive: bool = False) -> List[Subscriber]:
        if exclusive:
            return list(filter(lambda s: s.subscription.selector == channel.name, self._subscribers))
        else:
            return list(filter(lambda s: s.subscription.matches(channel), self._subscribers))

    def __repr_args__(self) -> pydantic.ReprArgs:
        return [
            ('running', self.running),
            ('channel_names', list(map(lambda c: c.name, self._channels))),
            ('publisher_count', len(self._publishers)),
            ('subscriber_count', len(self._subscribers)),
            ('queue_size', self._queue.qsize()),
        ]

    def __eq__(self, other) -> bool:
        # compare exchanges by object identity rather than fields
        if isinstance(other, Exchange):
            return id(self) == id(other)

        return False


class BaseSubscription(abc.ABC, pydantic.BaseModel):
    """Abstract base class for pub/sub subscriptions.

    Subscriptions are responsible for matching the Channels and Messages
    that a Subscriber is interested in. Subclass implementations must provide
    a `matches` method that evaluates the given `Channel` and optional `Message`
    objects. Subscriptions must always be matchable against channels. The message
    is provided for implementing attribute or content based matching when evaluating
    a published message.
    """

    @abc.abstractmethod
    def matches(self, channel: Channel, message: Optional[Message] = None) -> bool:
        """Return True if the channel and message match the subscription.

        Args:
            channel: The Channel to match against.
            message: The optional Message to match against.
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

    @property
    def is_pattern(self) -> bool:
        return isinstance(self.selector, re.Pattern) or not re.match(ChannelName.regex, self.selector)

    def matches(self, channel: Channel, message: Optional[Message] = None) -> bool:
        """Return True if the channel and message matches the subscription.

        Args:
            channel: The Channel to match against.
            message: The optional Message to match against.
        """
        if channel is None:
            raise ValueError("`channel` cannot be `None`")

        selector = self.selector
        if isinstance(selector, re.Pattern):
            return bool(selector.fullmatch(channel.name))
        elif isinstance(selector, str):
            return fnmatch.fnmatch(channel.name, selector)

        raise ValueError(f"unknown selector type: {selector.__class__.__name__}")


_current_iterator_var = contextvars.ContextVar("servo.pubsub._Iterator.current", default=None)


class _Iterator(pydantic.BaseModel):
    subscriber: Subscriber
    yield_channel: bool = True
    _queue: asyncio.Queue = pydantic.PrivateAttr(default_factory=asyncio.Queue)
    _stopped: asyncio.Event = pydantic.PrivateAttr(False)
    _message_reset_token: Optional[contextvars.Token] = pydantic.PrivateAttr(None)
    _iterator_reset_token: Optional[contextvars.Token] = pydantic.PrivateAttr(None)

    def __init__(self, subscriber: Subscriber, **kwargs) -> None:
        super().__init__(**kwargs, subscriber=subscriber)
        self.subscriber = subscriber  # Pydantic copying
        self._message_reset_token = _current_context_var.set(None)
        self._iterator_reset_token = _current_iterator_var.set(self)

    def stop(self) -> None:
        self._stopped = True
        self._queue.put_nowait(
            None
        )

    @property
    def stopped(self) -> bool:
        return self._stopped

    async def __call__(self, message: Message, channel: Channel) -> None:
        await self._queue.put(
            (message, channel)
        )

    def _stop_iteration(self) -> None:
        _current_context_var.reset(self._message_reset_token)
        _current_iterator_var.reset(self._iterator_reset_token)
        raise StopAsyncIteration

    async def __anext__(self) -> Tuple[Message, Channel]:
        if self.stopped:
            self._stop_iteration()

        message_context = await self._queue.get()
        if message_context is None:
            self._stop_iteration()

        _current_context_var.set(message_context)
        if self.yield_channel:
            return message_context
        else:
            return message_context[0]

    def __hash__(self): # noqa: D105
        return hash(
            (
                id(self),
            )
        )


Callback = Callable[[Message, Channel], Union[None, Awaitable[None]]]


class Subscriber(_ExchangeChildModel):
    """A Subscriber consumes relevant Messages published to an Exchange.

    Subscribers can invoke a callback when a new Message is published and can be used
    as an asynchronous iterator to process Messages. The `cancel` method
    will halt message processing and stop any attached async iterators.

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
                subscriber.cancel()
            ```
    """
    subscription: Subscription
    callback: Optional[Callback]
    _event: asyncio.Event = pydantic.PrivateAttr(default_factory=asyncio.Event)
    _iterators: List[_Iterator] = pydantic.PrivateAttr([])

    def stop(self) -> None:
        """Stop the current async iterator.

        The iterator to be stopped is determined by the current iteration scope.
        Calling stop on a parent iterator scope will trigger a `RuntimeError`.

        Raises:
            RuntimeError: Raised if there is not an active iterator or the receiver
                is not being iterated in the local scope.
        """
        iterator = _current_iterator()
        if iterator is not None:
            if iterator.subscriber != self:
                raise RuntimeError(f"Attempted to stop an inactive iterator")
            iterator.stop()
        else:
            raise RuntimeError("Attempted to stop outside of an iterator")

    def cancel(self) -> None:
        """Cancel the subscriber from receiving any further Messages.

        Any objects waiting on the Subscriber and any async iterators are released.

        Raises:
            RuntimeError: Raised if the Subscriber has alreayd been cancelled.
        """
        if self.cancelled:
            raise RuntimeError(f"Subscriber is already cancelled")
        self._event.set()

        # Stop any attached iterators
        for iterator in self._iterators:
            iterator.stop()

        self._iterators.clear()

    @property
    def cancelled(self) -> bool:
        """Return True if the subscriber has been cancelled."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait for the subscriber to be cancelled.

        The caller will block until the Subscriber is cancelled.
        """
        await self._event.wait()

    async def __call__(self, message: Message, channel: Channel) -> None:
        if self.cancelled:
            servo.logger.warning(f"ignoring call to cancelled Subscriber: {self}")
            return

        if self.subscription.matches(channel, message):
            if self.callback:
                # NOTE: Yield message or message, channel based on callable arity
                signature = inspect.Signature.from_callable(self.callback)
                if len(signature.parameters) == 1:
                    if asyncio.iscoroutinefunction(self.callback):
                        await self.callback(message)
                    else:
                        self.callback(message)
                elif len(signature.parameters) == 2:
                    if asyncio.iscoroutinefunction(self.callback):
                        await self.callback(message, channel)
                    else:
                        self.callback(message, channel)
                else:
                    raise TypeError(f"Incorrect callback")


            for _, iterator in enumerate(self._iterators):
                if iterator.stopped:
                    self._iterators.remove(iterator)
                else:
                    await iterator(message, channel)

    def __aiter__(self):  # noqa: D105
        iterator = _Iterator(self)
        self._iterators.append(iterator)
        return iterator

    def __eq__(self, other) -> bool:
        # compare exchanges by object identity rather than fields
        if isinstance(other, Subscriber):
            return id(self) == id(other)

        return False
    class Config:
        arbitrary_types_allowed = True


class Publisher(_ExchangeChildModel):
    """A Publisher broadcasts Messages to Channels in an Exchange.

    Publishers are asynchronously callable to publish a Message.

    Attributes:
        exchange: The pub/sub Exchange that the Publisher belongs to.
        channels: The Channels that the Publisher publishes Messages to.
    """
    channels: pydantic.conlist(Channel, min_items=1)

    async def __call__(self, message: Message, *channels: List[Union[Channel, str]]) -> None:
        for channel in (channels or self.channels):
            if channel_ := self._find_channel(channel):
                await self.exchange.publish(message, channel_)
            else:
                raise ValueError(f"Publisher is not bound to Channel: '{channel}'")

    def _find_channel(self, channel: Union[Channel, str]) -> Optional[Channel]:
        return next(filter(lambda c: c == channel, self.channels), None)


class _PublisherMethod:
    def __init__(
        self,
        parent: Mixin,
        channels: List[Union[Channel, str]],
        *,
        every: Optional[servo.types.DurationDescriptor] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__()
        self.pubsub_exchange = parent.pubsub_exchange
        self._publishers_map = parent._publishers_map
        self.channels = channels
        self.every = every
        self.name = name

    def __call__(self, fn) -> None:
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError("decorated function must be asynchronous")

        name_ = self.name or fn.__name__
        if name_ in self._publishers_map:
            raise KeyError(f"a Publisher named '{name_}' already exists")

        publisher = self.pubsub_exchange.create_publisher(*self.channels)
        if self.every is not None:
            duration = self.every if isinstance(self.every, servo.Duration) else servo.Duration(self.every)
        else:
            duration = None

        @functools.wraps(fn)
        async def _repeating_publisher() -> None:
            while True:
                await fn(publisher)
                if duration is not None:
                    await asyncio.sleep(duration.total_seconds())

        task = asyncio.create_task(_repeating_publisher())
        task.add_done_callback(_error_watcher)
        task.add_done_callback(lambda _: self._publishers_map.pop(name_))
        self._publishers_map[name_] = (publisher, task)

    async def __aenter__(self) -> None:
        if self.every is not None:
            raise TypeError(f"Cannot create repeating publisher when used as a context manager: `every` must be None")

        self.publisher = self.pubsub_exchange.create_publisher(*self.channels)
        return self.publisher

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.pubsub_exchange.remove_publisher(self.publisher)

    def __await__(self):
        # NOTE: If we are awaited, make the caller wait on publish() instead
        channel = (
            self.pubsub_exchange.get_channel(self.channels[1])
            or self.pubsub_exchange.create_channel(self.channels[1])
        )
        return channel.publish(self.channels[0]).__await__()


def _random_string() -> str:
    characters = string.ascii_lowercase + string.digits + '-'
    return "".join(random.choice(characters) for i in range(32))


class _ChannelMethod:
    def __init__(
        self,
        parent: Mixin,
        name: Optional[str],
        description: Optional[str],
    ) -> None:
        super().__init__()
        self.pubsub_exchange = parent.pubsub_exchange
        self.name = name or self._random_unique_channel_name()
        self.description = description

    def _random_unique_channel_name(self) -> str:
        while True:
            name = _random_string()
            if (self.pubsub_exchange.get_channel(name) is None
                and re.match(ChannelName.regex, name)):
                return name

    async def __aenter__(self) -> None:
        channel = self.pubsub_exchange.get_channel(self.name)
        if channel:
            self.temporary = False
        else:
            self.temporary = True
            channel = self.pubsub_exchange.create_channel(self.name, self.description)
        self.channel = channel
        return channel

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.temporary:
            await self.channel.close()
            self.pubsub_exchange.remove_channel(self.channel)

class _SubscriberMethod:
    def __init__(
        self,
        parent: Mixin,
        selector: Selector,
        name: Optional[str] = None,
        timeout: Optional[servo.types.DurationDescriptor] = None,
        until_done: Optional[servo.types.Futuristic] = None
    ) -> None:
        super().__init__()
        self.pubsub_exchange = parent.pubsub_exchange
        self._subscribers_map = parent._subscribers_map
        self.selector = selector
        self.name = name
        self.timeout = timeout
        self.until_done = until_done

    def __call__(self, fn) -> None:
        name_ = self.name or fn.__name__
        if name_ in self._subscribers_map:
            raise KeyError(f"a Subscriber named '{name_}' already exists")

        self._subscribers_map[name_] = self.pubsub_exchange.create_subscriber(
            self.selector, callback=fn, timeout=self.timeout, until_done=self.until_done
        )

    async def __aenter__(self) -> None:
        self.subscriber = self.pubsub_exchange.create_subscriber(self.selector)
        return self.subscriber

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.pubsub_exchange.remove_subscriber(self.subscriber)


class Mixin(pydantic.BaseModel):
    """Provides a simple pub/sub stack for subclasses.

    The `Mixin` class provides a very high-level API for interacting with the
    pub/sub system. Most of the details of the module are abstracted away and
    the interface exposed focuses three concepts: subscribing, publishing,
    and cancellation. The API is opinionated and provides methods that are
    usable as callable methods, decorators, or context managers. This keeps
    down the cognitive load for downstream developers and narrows the surface
    area in terms of attributes and methods introduced into subclasses. All
    functionality provided by the mixin is implemented on the lower level
    APIs of the module. It provides expressive, flexible abstractions but
    if you want them but inheriting from `Mixin` is not required to utilize
    the core functionality.

    The exchange is initialized into a stopped state and must be started to
    begin message exchange.

    Attributes:
        pubsub_exchange: The pub/sub Exchange that the object belongs to.
    """
    __private_attributes__ = {
        '_publishers_map': pydantic.PrivateAttr({}),
        '_subscribers_map': pydantic.PrivateAttr({}),
    }
    pubsub_exchange: Exchange = pydantic.Field(default_factory=Exchange)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # NOTE: Assign the exchange directly as Pydantic will copy it
        if exchange := kwargs.get('pubsub_exchange'):
            self.pubsub_exchange = exchange

    def channel(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """A context manager for retrieving pub/sub Channels.

        Retrieves a Channel with find-or-create semantics from the Exchange. If
        an existing Channel is found, it is yieled to the block. If no Channel
        with the given name exists or the name is omitted, a temporary Channel
        is created, yielded to the context block, and closed and removed from
        the Exchange upon return.

        The `name` may be omitted in which case a unique name is generated. If
        an existing Channel is referenced, then the Channel is not closed or
        removed upon return.

        Args:
            name: A name for the temporary Channel. When omitted, a random
                unique name is generated.
            description: An optional textual description of the Channel.
        """
        return _ChannelMethod(
            self,
            name=name,
            description=description
        )

    def subscribe(
        self,
        selector: Selector,
        *,
        name: Optional[str] = None,
        timeout: Optional[servo.types.DurationDescriptor] = None,
        until_done: Optional[asyncio.Future] = None
    ):
        """Create a Subscriber in the pub/sub Exchange.

        This method can be used as a decorator or as a context manager.

        When used as a decorator, the decorated function may be synchronous or
        asynchronous but must accept two arguments: `message: Message, channel:
        Channel`. The Subscriber created is persistent and will continue
        processing Messages until explicitly cancelled.

        When used as a context manager, a temporary Subscriber is created,
        attached to the Exchange, and automatically cancelled and removed upon
        exit from the managed context block scope. Messages are consumed via
        use of the Subscriber as an async iterator (`async for X in Y` syntax).
        Iteration can be terminated via the `stop` method of the Subscriber.

        Args:
            selector: A string or regular expression pattern matching Channels of interest.
            name: A name for the subscriber. When omitted, defaults to the name of
                the decorated function.

        Usage:
            ```
            # As a decorator
            def _decorator_example(self) -> None:
                @self.subscribe("metrics.*")
                def _message_received(message: Message, channel: Channel) -> None:
                    print(f"Notified of a new Message: {message}, {channel}")

            # As a context manager
            def _context_manager_example(self) -> None:
                async with self.subscribe("metrics.*") as subscriber:
                    async for message, channel in subscriber:
                        print(f"Notified of a new Message: {message}, {channel}")

                def _message_received(message: Message, channel: Channel) -> None:
                    print(f"Notified of a new Message: {message}, {channel}")
            ```
        """
        return _SubscriberMethod(
            self,
            selector=selector,
            name=name,
            timeout=timeout,
            until_done=until_done
        )

    def cancel_subscribers(self, *names: List[str]) -> None:
        """Cancel active pub/sub subscribers.

        When called without any names all subscribers are cancelled.

        Args:
            names: The names of the subscribers to cancel. When no names are given,
                all subscribers are cancelled.

        Raises:
            KeyError: Raised if there is no subscriber with the given name.
        """
        subscribers = (
            list(map(self._subscribers_map.get, names)) if names
            else self._subscribers_map.values()
        )

        for subscriber in subscribers:
            if not subscriber.cancelled:
                subscriber.cancel()
            self.pubsub_exchange.remove_subscriber(subscriber)

        self._subscribers_map = dict(
            filter(lambda i: i[1] not in subscribers, self._subscribers_map.items())
        )

    def publish(
        self,
        *channels: List[Union[Channel, str]],
        every: Optional[servo.types.DurationDescriptor] = None,
        name: Optional[str] = None
    ) -> None:
        """Create a Publisher in the pub/sub Exchange.

        This method can be used as an asynchronous callable, decorator, or
        context manager.

        When awaited, the method will publish a message to a target channel and
        functions as a convenience alias for `self.pubsub_exchange.publish()`.

        When used as a decorator, the Publisher created is persistent and will
        continue executing until cancelled. The `every` argument configures the
        publication of messages on a repeating time interval. When `every` is
        None, the caller is responsible for managing the sleep schedule of the
        Publisher. The decorated function must be asynchronous and accept a
        single argument: `publisher: Publisher`.

        When used as a context manager, a temporary Publisher is created,
        attached to the Exchange, and automatically cancelled and removed upon
        exit from the managed context block scope.

        Args:
            channel: The Channel or name of the Channel to bind the Publisher to.
            every: Optional Duration descriptor specifying how often the Publisher
                is to be awakened.
            name: An optional name to assign to the Publisher. When omitted, the
                name of the decorated function is used.

        Usage:
            ```
            # As a decorator

            ## Using a repeating interval...
            def repeating_example(self) -> None:
                @self.publish("metrics", every="15s")
                async def _publish_metrics(publisher: Publisher) -> None:
                    await publisher(Message(json={"throughput": "31337rps"}))

            ## Manually sleeping the publisher loop...
            def manual_example(self) -> None:
                @self.publish("metrics")
                async def _publish_metrics(publisher: Publisher) -> None:
                    await publisher(Message(json={"throughput": "31337rps"}))

                    seconds_to_sleep = random.randint(1, 15)
                    await asyncio.sleep(seconds_to_sleep)

            # As a context manager

            def context_manager_example(self) -> None:
                # Send 5 messages to the `metrics` channel and exit
                async with self.publish("metrics") as publisher:
                    for _ in range(5):
                        await publisher(Message(json={"throughput": "31337rps"}))
            ```
        """
        return _PublisherMethod(self, channels=channels, every=every, name=name)

    def cancel_publishers(self, *names: List[str]) -> None:
        """Cancel active pub/sub publishers.

        When called without any names all publishers are cancelled.

        Args:
            names: The names of the publishers to cancel. When no names are given,
                all publishers are cancelled.

        Raises:
            KeyError: Raised if there is no publishers with the given name.
        """
        publisher_tuples = (
            list(map(self._publishers_map.get, names)) if names
            else self._publishers_map.values()
        )

        for publisher, task in publisher_tuples:
            self.pubsub_exchange.remove_publisher(publisher)
            if not task.done():
                task.cancel()

        self._publishers_map = dict(
            filter(lambda i: i[1] not in publisher_tuples, self._publishers_map.items())
        )


async def _deliver_message_to_subscribers(message: Message, channel: Channel, subscribers: List[Subscriber]) -> None:
    reset_token = _current_context_var.set((message, channel))
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
            if isinstance(result, Exception):
                raise result

    _current_context_var.reset(reset_token)


def _current_iterator() -> Optional[AsyncIterator]:
    return servo.pubsub._current_iterator_var.get()

Channel.update_forward_refs()
_Iterator.update_forward_refs()


def _error_watcher(task: asyncio.Task) -> None:
    # Ensure that any exceptions from publishers are surfaced
    if task.done() and not task.cancelled():
        exception = task.exception()
        if exception and not isinstance(exception, asyncio.TimeoutError):
            loop = asyncio.get_event_loop()
            servo.logger.exception(f"Publisher task failed with exception: {exception}")
            context = {
                'future': task,
                'exception': exception,
                'message': f"Publisher Task failed with exception: {exception}",
            }
            loop.call_exception_handler(context)
