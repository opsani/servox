# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import enum
import functools
import inspect
import sys
import types
import weakref
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import pydantic
import pydantic.typing

import servo.errors
import servo.pubsub
import servo.utilities.inspect
import servo.utilities.strings

__all__ = [
    "Event",
    "EventHandler",
    "EventResult",
    "Preposition",
    "create_event",
    "current_event",
    "event",
    "before_event",
    "on_event",
    "after_event",
    "event_handler",
]


# Context vars for asyncio tasks managed by run_event_handlers
_current_context_var = contextvars.ContextVar("servox.current_event", default=None)


def current_event() -> Optional[EventContext]:
    """
    Returns an object that describes the actively executing event context, if any.

    The event context is helpful in introspecting concurrent runtime state without having to pass
    around info across methods. The `EventContext` object can be compared to strings for convenience
    and supports string comparison to both `event_name` and `preposition:event_name` constructs for
    easily checking current state.
    """
    return _current_context_var.get()


_connector_event_bus = weakref.WeakKeyDictionary()

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
    ) -> None:  # noqa: D107
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
    before = enum.auto()
    on = enum.auto()
    after = enum.auto()
    all = before | on | after

    @classmethod
    def from_str(cls, prep: str) -> "Preposition":
        if not isinstance(prep, str):
            return prep

        if prep == "before":
            return Preposition.before
        elif prep == "on":
            return Preposition.on
        elif prep == "after":
            return Preposition.after
        else:
            raise ValueError(f"unsupported value for Preposition '{prep}'")

    @property
    def flag(self) -> bool:
        """
        Return a boolean value that indicates if the requirements are an individual flag value.
        The implementation relies on the Python `enum.Flag` modeling of individual members of
        the flag enumeration as values that are powers of two (1, 2, 4, 8, …), while combinations
        of flags are not.
        """
        value = self.value
        return bool((value & (value - 1) == 0) and value != 0)

    def __str__(self):
        if self == Preposition.before:
            return "before"
        elif self == Preposition.on:
            return "on"
        elif self == Preposition.after:
            return "after"


class EventContext(pydantic.BaseModel):
    event: Event
    preposition: Preposition
    created_at: datetime.datetime = None

    @classmethod  # Usable as a validator
    def from_str(cls, event_str) -> Optional["EventContext"]:
        if event := get_event(event_str, None):
            return EventContext(preposition=Preposition.on, event=event)

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
        return self.preposition == Preposition.before

    def is_on(self) -> bool:
        return self.preposition == Preposition.on

    def is_after(self) -> bool:
        return self.preposition == Preposition.after

    @contextlib.contextmanager
    def current(self):
        """A context manager that sets the current connector context."""
        try:
            token = _current_context_var.set(self)
            yield self

        finally:
            _current_context_var.reset(token)

    def __str__(self):
        if self.preposition == Preposition.on:
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
    connector_type: Optional[Type["Mixin"]] = None  # NOTE: Optional due to decorator
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
    connector: "Mixin"
    created_at: datetime.datetime = None
    value: Any

    @pydantic.validator("created_at", pre=True, always=True)
    @classmethod
    def set_created_at_now(cls, v):
        return v or datetime.datetime.now()


##
# Event registry

_events: Dict[str, Event] = {}


def get_events() -> List[Event]:
    """Return all registered events."""
    return list(_events.values())


def get_event(name: str, default=...) -> Optional[Event]:
    """Retrieve an event by name."""
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
    """Create a new event using the signature of a decorated function.

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
    """Register a decorated function as an event handler to be run before the specified event.

    Before event handlers require no arguments positional or keyword arguments and return `None`. Any arguments
    provided via the `kwargs` parameter are passed through at invocation time. Before event handlers
    can cancel event propagation by raising `servo.errors.EventCancelledError`. Cancelled events are reported to the
    event originator by attaching the `servo.errors.EventCancelledError` instance to the `EventResult`.

    :param event: The event or name of the event to run the handler before.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.before, **kwargs)


def on_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """Register a decorated function as an event handler to be run on the specified event.

    :param event: The event or name of the event to run the handler on.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.on, **kwargs)


def after_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """Register a decorated function as an event handler to be run after the specified event.

    After event handlers are invoked with the event results as their first argument (type `List[EventResult]`)
    and return `None`.

    :param event: The event or name of the event to run the handler after.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.after, **kwargs)


def event_handler(
    event_name: Optional[str] = None,
    preposition: Preposition = Preposition.on,
    **kwargs,
) -> Callable[[EventCallable], EventCallable]:
    """Register a decorated function as an event handler.

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
        if preposition != Preposition.on:
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

        if preposition in (Preposition.before, Preposition.on):
            ref_signature = event.signature
            if preposition == Preposition.before:
                # 'before' event takes same args as 'on' event, but returns None
                ref_signature = ref_signature.replace(return_annotation="None")
            servo.utilities.inspect.assert_equal_callable_descriptors(
                servo.utilities.inspect.CallableDescriptor(
                    signature=ref_signature,
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
                callable_description="event handler"
                if preposition == Preposition.on
                else "before event handler",
            )
        elif preposition == Preposition.after:
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
                callable_description="after event handler",
            )
        else:
            assert "Undefined preposition value"

        # Annotate the function for processing later, see Connector.__init_subclass__
        fn.__event_handler__ = EventHandler(
            event=event, preposition=preposition, handler=fn, kwargs=kwargs
        )
        return fn

    return decorator


def __after_handler(self, results: List[EventResult]) -> None:
    pass


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
        __connectors__: List[Mixin] = None,
        **kwargs,
    ) -> None:  # noqa: D107
        super().__init__(
            *args,
            **kwargs,
        )

        # NOTE: Connector references are held off the model so
        # that Pydantic doesn't see additional attributes
        __connectors__ = __connectors__ if __connectors__ is not None else [self]
        _connector_event_bus[self] = __connectors__

    @classmethod
    def __get_validators__(cls: Mixin) -> pydantic.typing.CallableGenerator:
        yield cls.validate

    @classmethod
    def validate(cls: Mixin, value: Any) -> Mixin:
        if not isinstance(value, Mixin):
            raise TypeError(
                f"field (type {type(value)}) must be instance of events.Mixin"
            )

        # Ideally, the name property would be part of an abstract base but pydantic doesn't play nice with abc
        # https://github.com/samuelcolvin/pydantic/discussions/2410
        if (not hasattr(value, "name")) or not isinstance(value.name, str):
            raise TypeError(
                f"events.Mixin inheritors must define a name property of type str (found {type(getattr(value, 'name', None))})"
            )

        return value

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
        cls, event: Union[Event, str], preposition: Preposition = Preposition.all
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
    def __connectors__(self) -> List[Mixin]:
        return _connector_event_bus[self]

    def dispatch_event(
        self,
        event: Union[Event, str],
        *args,
        first: bool = False,
        include: Optional[List[Union[str, Mixin]]] = None,
        exclude: Optional[List[Union[str, Mixin]]] = None,
        return_exceptions: bool = False,
        _prepositions: Preposition = (
            Preposition.before | Preposition.on | Preposition.after
        ),
        **kwargs,
    ) -> Union[Optional[EventResult], List[EventResult]]:
        """
        Dispatch an event to active connectors for processing and returns the results.

        Eventing is used to notify other connectors of activities and state changes
        driven by one connector or to facilitate loosely coupled cross-connector RPC
        communication.

        Events are dispatched by invoking the before, on, and after event
        handlers of the active connectors for the given event. Before handlers
        may cancel the event by raising an `EventCancelledError`. When an event
        is cancelled by a before handler, an empty result list is returned and a
        warning is logged. Other exceptions are raised and interrupt the
        event dispatch operation exceptionally.

        When the `return_exceptions` argument is True, exceptions raised by on
        event handlers are returned as the `value` attribute of `EventResult`
        objects in the list returned. After event handlers are invoked with the
        complete list of results before the method returns.

        When the `return_exceptions` argument is False, the first exception
        encountered will interrupt the event dispatch operation exceptionally.

        Whenever possible, event handlers should raise an exception derived from
        `servo.events.EventError` when a runtime error is encountered and
        utilize exception chaining as described in [PEP 3134](https://www.python.org/dev/peps/pep-3134/).

        Args:
            event: The name or event to dispatch.
            first: When True, halt dispatch and return the result from the first
                connector that responds.
            include: A list of specific connectors to dispatch the event to.
            exclude: A list of specific connectors to exclude from event
                dispatch.
            return_exceptions: When True, exceptions raised by on event handlers
                are returned as event results.

        Returns:
            A list of event result objects detailing the results returned.
        """
        connectors: List[Mixin] = self.__connectors__
        event = get_event(event) if isinstance(event, str) else event

        if include is not None:
            included_names = list(
                map(lambda c: c if isinstance(c, str) else c.name, include)
            )
            connectors = list(filter(lambda c: c.name in included_names, connectors))

        if exclude is not None:
            excluded_names = list(
                map(lambda c: c if isinstance(c, str) else c.name, exclude)
            )
            connectors = list(
                filter(lambda c: c.name not in excluded_names, connectors)
            )

        # Validate that we are dispatching to connectors that are in our graph
        if not set(connectors).issubset(self.__connectors__):
            raise ValueError(
                f"invalid target connectors: cannot dispatch events to connectors that are in the active servo"
            )

        return _DispatchEvent(
            connectors=connectors,
            event=event,
            parent=self,
            args=args,
            first=first,
            include=include,
            exclude=exclude,
            return_exceptions=return_exceptions,
            _prepositions=_prepositions,
            kwargs=kwargs,
        )

    async def run_event_handlers(
        self,
        event: Event,
        preposition: Preposition,
        *args,
        return_exceptions: bool = False,
        **kwargs,
    ) -> Optional[List[EventResult]]:
        """
        Run handlers for the given event and preposition and return the results
        or None if there are no handlers.

        Exceptions are rescued and returned as event result objects in which the
        `value` attribute is the exception.
        """
        if not isinstance(event, Event):
            raise ValueError(
                f"event must be an Event object, got {event.__class__.__name__}"
            )

        event_handlers = self.get_event_handlers(event, preposition)
        if not event_handlers:
            return None

        with self.current():
            with EventContext(event=event, preposition=preposition).current():
                results: List[EventResult] = []
                for event_handler in event_handlers:
                    # NOTE: Explicit kwargs take precendence over those defined during handler declaration
                    merged_kwargs = event_handler.kwargs.copy()
                    merged_kwargs.update(kwargs)
                    try:
                        method = types.MethodType(event_handler.handler, self)
                        async with event.on_handler_context_manager(self):
                            if asyncio.iscoroutinefunction(method):
                                value = await asyncio.create_task(
                                    method(*args, **merged_kwargs),
                                    name=f"{preposition}:{event}",
                                )
                            else:
                                value = method(*args, **merged_kwargs)

                        result = EventResult(
                            connector=self,
                            event=event,
                            preposition=preposition,
                            handler=event_handler,
                            value=value,
                        )
                        results.append(result)

                    except Exception as error:
                        if (
                            isinstance(error, servo.errors.EventCancelledError)
                            and preposition != Preposition.before
                        ):
                            if return_exceptions:
                                self.logger.warning(
                                    f"Cannot cancel an event from an {preposition} handler: event dispatched"
                                )
                            else:
                                cause = error
                                error = TypeError(
                                    f"Cannot cancel an event from an {preposition} handler"
                                )
                                error.__cause__ = cause

                        # Annotate the exception and reraise to halt execution
                        error.__event_result__ = EventResult(
                            connector=self,
                            event=event,
                            preposition=preposition,
                            handler=event_handler,
                            value=error,
                        )

                        if return_exceptions:
                            results.append(error.__event_result__)
                        else:
                            raise error

        return results


_is_base_class_defined = True


class _DispatchEvent:
    def __init__(
        self,
        event: Union[Event, str],
        connectors: List[Mixin],
        parent: servo.pubsub.Mixin,
        args: List[Any],
        first: bool = False,
        include: Optional[List[Union[str, Mixin]]] = None,
        exclude: Optional[List[Union[str, Mixin]]] = None,
        return_exceptions: bool = False,
        _prepositions: Preposition = (
            Preposition.before | Preposition.on | Preposition.after
        ),
        kwargs: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self._event = event
        self._connectors = connectors
        self._args = args
        self._first = first
        self._include = include
        self._exclude = exclude
        self._return_exceptions = return_exceptions
        self._prepositions = _prepositions
        self._parent = parent
        self._kwargs = kwargs

        # Execution state
        self._run = False
        self._results = None
        self._channel = None

    @property
    def event(self) -> Event:
        """The Event being dispatched."""
        return self._event

    @property
    def channel(self) -> Optional[servo.pubsub.Channel]:
        """The temporary Channel associated with this Event dispatch."""
        return self._channel

    @property
    def results(self) -> Optional[List[EventResult]]:
        """The results retured by other connectors that responded to the dispatched Event."""
        return self._results

    @property
    def done(self) -> bool:
        """Returns True when the Event dispatch operation has completed."""
        return self._run is True

    @property
    def success(self) -> bool:
        """Returns True when the Event dispatch operation completed successfully."""
        return self.done and self.results is not None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.event}: done={self.done}, success={self.success}, channel='{self.channel and self.channel.name}'>"

    def __await__(self):
        # NOTE: If we are awaited, make the caller wait on run() instead
        return self.run().__await__()

    def subscribe(self, *args):
        """Subscribe to the Event dispatch operation.

        This method is usable as a callable, context manager, or decorator.
        """
        subscriber_method = servo.pubsub._SubscriberMethod(
            self._parent,
            selector=self.channel.name,
        )
        # NOTE: Enables use as a decorator or callable
        if len(args) == 0:
            return subscriber_method
        else:
            return subscriber_method(*args)

    async def __aenter__(self) -> None:
        self._channel = self._parent.pubsub_exchange.create_channel(
            f"servo.events.{self.event.name}.{id(self)}"
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if not self.done:
            self._results = await self.run()

    def __aiter__(self):  # noqa: D105
        # Iterate through the channel
        return self._channel.__aiter__()

    async def run(self) -> List[EventResult]:
        """Run the Event dispatch operation to completion and return results."""
        if self.done:
            raise RuntimeError(f"Event dispatch has already run")

        self._run = True
        results: List[EventResult] = []

        # Invoke the before event handlers
        if self._prepositions & Preposition.before:
            for connector in self._connectors:
                try:
                    results = await connector.run_event_handlers(
                        self.event,
                        Preposition.before,
                        *self._args,
                        return_exceptions=False,
                        **self._kwargs,
                    )

                except servo.errors.EventCancelledError as error:
                    # Return an empty result set
                    servo.logger.warning(
                        f'event cancelled by before event handler on connector "{connector.name}": {error}'
                    )
                    return []

        # Invoke the on event handlers and gather results
        if self._prepositions & Preposition.on:
            if self._first:
                # A single responder has been requested
                for connector in self._connectors:
                    results = await connector.run_event_handlers(
                        self.event,
                        Preposition.on,
                        *self._args,
                        return_exceptions=self._return_exceptions,
                        **self._kwargs,
                    )
                    if results:
                        break
            else:
                group = asyncio.gather(
                    *list(
                        map(
                            lambda c: c.run_event_handlers(
                                self.event,
                                Preposition.on,
                                return_exceptions=self._return_exceptions,
                                *self._args,
                                **self._kwargs,
                            ),
                            self._connectors,
                        )
                    ),
                )
                results = await group
                results = list(filter(lambda r: r is not None, results))
                results = functools.reduce(lambda x, y: x + y, results, [])

        # Invoke the after event handlers
        if self._prepositions & Preposition.after:
            await asyncio.gather(
                *list(
                    map(
                        lambda c: c.run_event_handlers(
                            self.event, Preposition.after, results
                        ),
                        self._connectors,
                    )
                )
            )

        if self.channel:
            await self.channel.close()

        return next(iter(results), None) if self._first else results

    async def __call__(self) -> Union[Optional[EventResult], List[EventResult]]:
        self._results = await self.run()

        return next(iter(self.results), None) if self._first else self.results
