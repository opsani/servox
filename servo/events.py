from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import enum
import functools
import inspect
import sys
import weakref
from typing import Any, AsyncContextManager, Awaitable, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union

import pydantic
import pydantic.main

import servo.utilities.inspect
import servo.utilities.strings

__all__ = [
    "Event",
    "EventHandler",
    "EventResult",
    "Preposition",
    "create_event",
    "event",
    "before_event",
    "on_event",
    "after_event",
    "event_handler",
]

_signature_cache: Dict[str, inspect.Signature] = {}


class Event(pydantic.BaseModel):
    """
    The Event class defines a named event that can be dispatched and
    processed with before, on, and after handlers.
    """

    name: str
    """Unique name of the event.
    """

    module: Optional[str] = None
    """Module that defined the event.
    """

    on_handler_context_manager: Callable[[None], AsyncContextManager]
    """Context manager callable providing a default on event handler for the event.
    """

    def __init__(
        self, name: str, signature: inspect.Signature, *args, **kwargs
    ) -> None:
        _signature_cache[name] = signature
        super().__init__(name=name, *args, **kwargs)

    @property
    def signature(self) -> inspect.Signature:
        # Hide signature from Pydantic
        return _signature_cache[self.name]

    def __hash__(self):
        return hash(
            (
                self.name,
                self.signature,
            )
        )

    def __str__(self):
        return self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.__str__() == other
        elif isinstance(other, Event):
            return self.name == other.name and self.signature == other.signature
        return super().__eq__(other)

    def dict(
        self,
        *,
        include: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        exclude: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> pydantic.DictStrAny:
        if exclude is None:
            exclude = set()
        exclude.add("on_handler_context_manager")
        return super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    class Config:
        arbitrary_types_allowed = True


EventCallable = TypeVar("EventCallable", bound=Callable[..., Any])


class Preposition(enum.Flag):
    BEFORE = enum.auto()
    ON = enum.auto()
    AFTER = enum.auto()
    ALL = BEFORE | ON | AFTER

    @classmethod
    def from_str(cls, prep: str) -> "Preposition":
        if not isinstance(prep, str):
            return prep

        if prep == "before":
            return Preposition.BEFORE
        elif prep == "on":
            return Preposition.ON
        elif prep == "after":
            return Preposition.AFTER
        else:
            raise ValueError(f"unsupported value for Preposition '{prep}'")

    def __str__(self):
        if self == Preposition.BEFORE:
            return "before"
        elif self == Preposition.ON:
            return "on"
        elif self == Preposition.AFTER:
            return "after"


class EventContext(pydantic.BaseModel):
    event: Event
    preposition: Preposition
    created_at: datetime.datetime = None

    @classmethod  # Usable as a validator
    def from_str(cls, event_str) -> Optional["EventContext"]:
        if event := get_event(event_str, None):
            return EventContext(preposition=Preposition.ON, event=event)

        components = event_str.split(":", 1)
        if len(components) < 2:
            return None
        preposition, event_name = components
        if not (preposition or event_name):
            return None

        if preposition not in ("before", "on", "after"):
            return None

        event = get_event(event_name, None)
        if not event:
            return None

        return EventContext(preposition=Preposition.from_str(preposition), event=event)

    @pydantic.validator("created_at", pre=True, always=True)
    @classmethod
    def set_created_at_now(cls, v):
        return v or datetime.datetime.now()

    def is_before(self) -> bool:
        return self.preposition == Preposition.BEFORE

    def is_on(self) -> bool:
        return self.preposition == Preposition.ON

    def is_after(self) -> bool:
        return self.preposition == Preposition.AFTER

    def __str__(self):
        if self.preposition == Preposition.ON:
            return self.event.name
        else:
            return f"{self.preposition}:{self.event.name}"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return other in (self.__str__(), f"on:{self.event.name}")
        return super().__eq__(other)

    # FIXME: This should be aligned with `servo.api.Command.response_event` somehow
    def operation(self) -> Optional[str]:
        event_name = self.event.name.upper()
        if event_name == "DESCRIBE":
            return "DESCRIPTION"
        elif event_name == "MEASURE":
            return "MEASUREMENT"
        elif event_name == "ADJUST":
            return "ADJUSTMENT"
        else:
            return None


def validate_event_contexts(
    cls,
    value: Union[str, EventContext, Sequence[Union[str, EventContext]]],
    field: pydantic.ModelField,
) -> Union[str, List[str]]:
    """
    A Pydantic validator function that ensures that the input value or values are
    valid event context identifiers (e.g. "measure", "before:adjust", etc)
    """
    if isinstance(value, str):
        if not EventContext.from_str(value):
            raise ValueError(f"Invalid event {value}")
        return value
    elif isinstance(value, EventContext):
        return str(value)
    elif isinstance(value, List):
        for e in value:
            validate_event_contexts(cls, e, field)
        return value
    else:
        raise ValueError(f"Invalid value for {field.name}")


class EventHandler(pydantic.BaseModel):
    event: Event
    preposition: Preposition
    kwargs: Dict[str, Any]
    connector_type: Optional[Type["servo.connector.BaseConnector"]]  # NOTE: Optional due to decorator
    handler: EventCallable

    def __str__(self):
        return f"{self.connector_type}({self.preposition}:{self.event}->{self.handler})"


class EventResult(pydantic.BaseModel):
    """
    Encapsulates the result of a dispatched Connector event
    """

    event: Event
    preposition: Preposition
    handler: EventHandler
    connector: "servo.connector.BaseConnector"
    created_at: datetime.datetime = None
    value: Any

    @pydantic.validator("created_at", pre=True, always=True)
    @classmethod
    def set_created_at_now(cls, v):
        return v or datetime.datetime.now()


class EventError(RuntimeError):
    pass


class CancelEventError(EventError):
    result: EventResult


##
# Event registry

_events: Dict[str, Event] = {}


def get_events() -> List[Event]:
    """
    Return all registered events.
    """
    return list(_events.values())


def get_event(name: str, default=...) -> Optional[Event]:
    """
    Retrieve an event by name.
    """
    if default is Ellipsis:
        return _events[name]
    else:
        return _events.get(name, default)


def create_event(
    name: str,
    signature: Union[Callable[[Any], Awaitable], inspect.Signature],
    *,
    module: Optional[str] = None,
) -> Event:
    """
    Create an event programmatically from a name and function signature.

    Args:
        name: The name of the event to be created.
        signature: The method signature of on event handlers of the event.
        module: The module that defined the event. When `None`, inferred via the `inspect` module.
    """
    if _events.get(name, None):
        raise ValueError(f"Event '{name}' has already been created")

    def _default_context_manager() -> AsyncContextManager:
        # Simply yield to the on event handler
        async def fn(self) -> None:
            yield

        return contextlib.asynccontextmanager(fn)

    if callable(signature):
        if inspect.isasyncgenfunction(signature):
            # We have an async generator function defining setup/teardown activities, wrap into a context manager
            # This is useful for shared behaviors like startup delays, settlement times, etc.
            on_handler_context_manager = contextlib.asynccontextmanager(signature)

        elif not inspect.iscoroutinefunction(signature):
            raise ValueError(
                f"events must be async: add `async` prefix to your function declaration and await as necessary ({signature})"
            )

        else:
            # Sanity check callables that don't yield are stubs
            # We expect the last line to be 'pass' or '...' for stub code
            try:
                lines = inspect.getsourcelines(signature)
                last = lines[0][-1]
                if not last.strip() in ("pass", "..."):
                    raise ValueError(
                        "function body of event declaration must be an async generator or a stub using `...` or `pass` keywords"
                    )

                # use the default since our input doesn't yield
                on_handler_context_manager = _default_context_manager()

            except OSError:
                from servo.logging import logger

                logger.warning(
                    f"unable to inspect event declaration for '{name}': dropping event body and proceeding"
                )
                on_handler_context_manager = _default_context_manager()

    else:
        # Signatures are opaque from introspection
        on_handler_context_manager = _default_context_manager()

    signature = (
        signature
        if isinstance(signature, inspect.Signature)
        else inspect.Signature.from_callable(signature)
    )
    if list(
        filter(
            lambda param: param.kind == inspect.Parameter.VAR_POSITIONAL,
            signature.parameters.values(),
        )
    ):
        raise TypeError(
            f"Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
        )

    # Get the module from the calling stack frame
    if module is None:
        localns = inspect.currentframe().f_back.f_locals
        module = localns.get("__module__", None)

    event = Event(
        name=name,
        signature=signature,
        module=module,
        on_handler_context_manager=on_handler_context_manager,
    )
    _events[name] = event
    return event


def event(
    name: Optional[str] = None, *, handler: bool = False
) -> Callable[[EventCallable], EventCallable]:
    """
    Creates a new event using the signature of the decorated function.

    Events must be defined before handlers can be registered using before_event, on_event, after_event, or
    event_handler.

    :param handler: When True, the decorated function implementation is registered as an on event handler.
    """

    def decorator(fn: EventCallable) -> EventCallable:
        event_name = name if name else fn.__name__
        module = inspect.currentframe().f_back.f_locals.get("__module__", None)
        if handler:
            # If the method body is a handler, pass the signature directly into `create_event`
            # as we are going to pass the method body into `on_event`
            signature = inspect.Signature.from_callable(fn)
            create_event(event_name, signature, module=module)
        else:
            create_event(event_name, fn, module=module)

        if handler:
            decorator = on_event(event_name)
            return decorator(fn)
        else:
            return fn

    return decorator


def before_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run before the specified event.

    Before event handlers require no arguments positional or keyword arguments and return `None`. Any arguments
    provided via the `kwargs` parameter are passed through at invocation time. Before event handlers
    can cancel event propagation by raising `CancelEventError`. Canceled events are reported to the
    event originator by attaching the `CancelEventError` instance to the `EventResult`.

    :param event: The event or name of the event to run the handler before.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.BEFORE, **kwargs)


def on_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run on the specified event.

    :param event: The event or name of the event to run the handler on.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.ON, **kwargs)


def after_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run after the specified event.

    After event handlers are invoked with the event results as their first argument (type `List[EventResult]`)
    and return `None`.

    :param event: The event or name of the event to run the handler after.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.AFTER, **kwargs)


def event_handler(
    event_name: Optional[str] = None,
    preposition: Preposition = Preposition.ON,
    **kwargs,
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler.

    Event handlers are the foundational mechanism for connectors to provide functionality
    to the servo assembly. As events occur during operation, handlers are invoked and the
    results are aggregated and evalauted by the servo. Event handlers are registered for
    a specific event preposition, enabling them to execute before, after, or during an event.

    :param event: Specifies the event name. If not given, inferred from the name of the decorated handler function.
    :param preposition: Specifies the sequencing of a handler in relation to the event.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """

    def decorator(fn: EventCallable) -> EventCallable:
        name = event_name if event_name else fn.__name__
        event = _events.get(name, None)
        if event is None:
            raise ValueError(f"Unknown event '{name}'")
        if preposition != Preposition.ON:
            name = f"{preposition}:{name}"

        # Build namespaces that can resolve names for the event definition and handler
        event_globalns = (
            sys.modules[event.module].__dict__.copy() if event.module else {}
        )
        event_globalns.update(globals())
        handler_signature = inspect.Signature.from_callable(fn)
        handler_globalns = inspect.currentframe().f_back.f_globals
        handler_localns = inspect.currentframe().f_back.f_locals

        handler_mod_name = handler_localns.get("__module__", None)
        handler_module = sys.modules[handler_mod_name] if handler_mod_name else None

        if preposition == Preposition.BEFORE:
            before_handler_signature = inspect.Signature.from_callable(__before_handler)
            servo.utilities.inspect.assert_equal_callable_descriptors(
                servo.utilities.inspect.CallableDescriptor(
                    signature=before_handler_signature,
                    module=event.module,
                    globalns=event_globalns,
                    localns=locals(),
                ),
                servo.utilities.inspect.CallableDescriptor(
                    signature=handler_signature,
                    module=handler_module,
                    globalns=handler_globalns,
                    localns=handler_localns,
                ),
                name=name,
                method=True,
            )
        elif preposition == Preposition.ON:
            servo.utilities.inspect.assert_equal_callable_descriptors(
                servo.utilities.inspect.CallableDescriptor(
                    signature=event.signature,
                    module=event.module,
                    globalns=event_globalns,
                    localns=locals(),
                ),
                servo.utilities.inspect.CallableDescriptor(
                    signature=handler_signature,
                    module=handler_module,
                    globalns=handler_globalns,
                    localns=handler_localns,
                ),
                name=name,
                method=True,
            )
        elif preposition == Preposition.AFTER:
            after_handler_signature = inspect.Signature.from_callable(__after_handler)
            servo.utilities.inspect.assert_equal_callable_descriptors(
                servo.utilities.inspect.CallableDescriptor(
                    signature=after_handler_signature,
                    module=event.module,
                    globalns=event_globalns,
                    localns=locals(),
                ),
                servo.utilities.inspect.CallableDescriptor(
                    signature=handler_signature,
                    module=handler_module,
                    globalns=handler_globalns,
                    localns=handler_localns,
                ),
                name=name,
                method=True,
            )
        else:
            assert "Undefined preposition value"

        # Annotate the function for processing later, see Connector.__init_subclass__
        fn.__event_handler__ = EventHandler(
            event=event, preposition=preposition, handler=fn, kwargs=kwargs
        )
        return fn

    return decorator


def __before_handler(self) -> None:
    pass


def __after_handler(self, results: List[EventResult]) -> None:
    pass


# Context vars for asyncio tasks managed by run_event_handlers
_event_context_var = contextvars.ContextVar("servo.event", default=None)
_connector_context_var = contextvars.ContextVar("servo.connector", default=None)
_connector_event_bus = weakref.WeakKeyDictionary()


# NOTE: Boolean flag to know if we can safely reference base class from the metaclass
_is_base_class_defined = False


class Metaclass(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Decorate the class with an event registry, inheriting from our parent connectors
        event_handlers: List[EventHandler] = []

        for base in reversed(bases):
            if _is_base_class_defined and issubclass(base, Mixin) and base is not Mixin:
                event_handlers.extend(base.__event_handlers__)

        new_namespace = {
            "__event_handlers__": event_handlers,
            **{n: v for n, v in namespace.items()},
        }

        cls = super().__new__(mcs, name, bases, new_namespace, **kwargs)
        return cls


class Mixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register events handlers for all annotated methods (see `event_handler` decorator)
        for key, value in cls.__dict__.items():
            if handler := getattr(value, "__event_handler__", None):
                if not isinstance(handler, EventHandler):
                    raise TypeError(
                        f"Unexpected event descriptor of type '{handler.__class__}'"
                    )

                handler.connector_type = cls
                cls.__event_handlers__.append(handler)

    def __init__(
        self,
        *args,
        __connectors__: List["servo.connector.BaseConnector"] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # NOTE: Connector references are held off the model so
        # that Pydantic doesn't see additional attributes
        __connectors__ = __connectors__ if __connectors__ is not None else [self]
        _connector_event_bus[self] = __connectors__

    @classmethod
    def responds_to_event(cls, event: Union[Event, str]) -> bool:
        """
        Returns True if the Connector processes the specified event (before, on, or after).
        """
        if isinstance(event, str):
            event = get_event(event)

        handlers = list(
            filter(lambda handler: handler.event == event, cls.__event_handlers__)
        )
        return len(handlers) > 0

    @classmethod
    def get_event_handlers(
        cls, event: Union[Event, str], preposition: Preposition = Preposition.ALL
    ) -> List[EventHandler]:
        """
        Retrieves the event handlers for the given event and preposition.
        """
        if isinstance(event, str):
            event = get_event(event, None)

        return list(
            filter(
                lambda handler: handler.event == event
                and handler.preposition & preposition,
                cls.__event_handlers__,
            )
        )

    @classmethod
    def add_event_handler(
        cls, event: Event, preposition: Preposition, callable: EventCallable, **kwargs
    ) -> EventHandler:
        """
        Programmatically creates and adds an event handler to the receiving class.

        This method is functionally equivalent to using the decorators to add
        event handlers at import time.
        """
        # Reuse the existing decorator and as would be done in __init__
        d_callable = event_handler(event.name, preposition, **kwargs)(callable)
        handler = d_callable.__event_handler__
        handler.connector_type = cls
        cls.__event_handlers__.append(handler)
        return handler

    @property
    def __connectors__(self) -> List["servo.connector.BaseConnector"]:
        return _connector_event_bus[self]

    def broadcast_event(
        self,
        event: Union[Event, str],
        *args,
        first: bool = False,
        include: Optional[List["servo.connector.BaseConnector"]] = None,
        exclude: Optional[List["servo.connector.BaseConnector"]] = None,
        prepositions: Preposition = (
            Preposition.BEFORE | Preposition.ON | Preposition.AFTER
        ),
        return_exceptions: bool = False,
        **kwargs,
    ) -> asyncio.Task:
        """
        Broadcast an event asynchronously in a fire and forget manner.

        Useful for dispatching notification events where you do not need
        or care about the result.
        """
        return asyncio.create_task(
            self.dispatch_event(
                event,
                *args,
                first=first,
                include=include,
                exclude=exclude,
                prepositions=prepositions,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        )

    async def dispatch_event(
        self,
        event: Union[Event, str],
        *args,
        first: bool = False,
        include: Optional[List["servo.connector.BaseConnector"]] = None,
        exclude: Optional[List["servo.connector.BaseConnector"]] = None,
        prepositions: Preposition = (
            Preposition.BEFORE | Preposition.ON | Preposition.AFTER
        ),
        return_exceptions: bool = False,
        **kwargs,
    ) -> Union[Optional[EventResult], List[EventResult]]:
        """
        Dispatches an event to active connectors for processing and returns the results.

        Eventing can be used to notify other connectors of activities and state changes
        driven by one connector or to facilitate loosely coupled cross-connector RPC
        communication.

        :param first: When True, halt dispatch and return the result from the first connector that responds.
        :param include: A list of specific connectors to dispatch the event to.
        :param exclude: A list of specific connectors to exclude from event dispatch.
        :param return_exceptions: When True, exceptions returned by on event handlers are returned as results.
        """
        results: List[EventResult] = []
        connectors = include if include is not None else self.__connectors__
        event = get_event(event) if isinstance(event, str) else event

        if exclude:
            # NOTE: We filter by name to avoid recursive hell in Pydantic
            excluded_names = list(map(lambda c: c.name, exclude))
            connectors = list(
                filter(lambda c: c.name not in excluded_names, connectors)
            )

        # Invoke the before event handlers
        if prepositions & Preposition.BEFORE:
            try:
                for connector in connectors:
                    await connector.run_event_handlers(
                        event, Preposition.BEFORE
                    )
            except CancelEventError as error:
                # Cancelled by a before event handler. Unpack the result and return it
                return [error.result]

        # Invoke the on event handlers and gather results
        if prepositions & Preposition.ON:
            if first:
                # A single responder has been requested
                for connector in connectors:
                    results = await connector.run_event_handlers(
                        event, Preposition.ON, *args, **kwargs
                    )
                    if results:
                        break
            else:
                group = asyncio.gather(
                    *list(
                        map(
                            lambda c: c.run_event_handlers(
                                event, Preposition.ON, *args, **kwargs
                            ),
                            connectors,
                        )
                    ),
                    return_exceptions=return_exceptions,
                )
                results = await group
                results = list(filter(lambda r: r is not None, results))
                if results:
                    results = functools.reduce(lambda x, y: x + y, results)

        # Invoke the after event handlers
        if prepositions & Preposition.AFTER:
            await asyncio.gather(
                *list(
                    map(
                        lambda c: c.run_event_handlers(
                            event, Preposition.AFTER, results
                        ),
                        connectors,
                    )
                )
            )

        if first:
            return results[0] if results else None

        return results

    async def run_event_handlers(
        self,
        event: Event,
        preposition: Preposition,
        *args,
        return_exceptions: bool = False,
        **kwargs,
    ) -> Optional[List[EventResult]]:
        """
        Run handlers for the given event and preposition and return the results or None if there are no handlers.
        """
        if not isinstance(event, Event):
            raise ValueError(
                f"event must be an Event object, got {event.__class__.__name__}"
            )

        event_handlers = self.get_event_handlers(event, preposition)
        if not event_handlers:
            return None

        results: List[EventResult] = []
        try:
            prev_connector_token = _connector_context_var.set(self)
            prev_event_token = _event_context_var.set(
                EventContext(event=event, preposition=preposition)
            )
            for event_handler in event_handlers:
                # NOTE: Explicit kwargs take precendence over those defined during handler declaration
                handler_kwargs = event_handler.kwargs.copy()
                handler_kwargs.update(kwargs)
                try:
                    async with event.on_handler_context_manager(self):
                        if asyncio.iscoroutinefunction(event_handler.handler):
                            value = await asyncio.create_task(
                                event_handler.handler(self, *args, **kwargs),
                                name=f"{preposition}:{event}",
                            )
                        else:
                            value = event_handler.handler(self, *args, **kwargs)

                except CancelEventError as error:
                    if preposition != Preposition.BEFORE:
                        raise TypeError(
                            f"Cannot cancel an event from an {preposition} handler"
                        ) from error

                    # Annotate the exception and reraise to halt execution
                    error.result = EventResult(
                        connector=self,
                        event=event,
                        preposition=preposition,
                        handler=event_handler,
                        value=error,
                    )
                    raise error

                except EventError as error:
                    value = error

                except Exception as error:
                    if return_exceptions:
                        value = error
                    else:
                        raise error

                # TODO: Annotate the responses to verify that they are of the correct types
                # TODO: Should have warning logs and options/way to retrieve bad results for debugging
                result = EventResult(
                    connector=self,
                    event=event,
                    preposition=preposition,
                    handler=event_handler,
                    value=value,
                )
                results.append(result)
        finally:
            _connector_context_var.reset(prev_connector_token)
            _event_context_var.reset(prev_event_token)

        return results

    @property
    def current_event(self) -> Optional[EventContext]:
        """
        Returns an object that describes the actively executing event context, if any.

        The event context is helpful in introspecting concurrent runtime state without having to pass
        around info across methods. The `EventContext` object can be compared to strings for convenience
        and supports string comparison to both `event_name` and `preposition:event_name` constructs for
        easily checking current state.
        """
        return _event_context_var.get()


_is_base_class_defined = True
