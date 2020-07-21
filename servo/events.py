from datetime import datetime
from enum import Flag, auto
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, Optional, Type, TypeVar, List, Union

from pydantic import BaseModel, validator
from servo.utilities import join_to_series


class Event(BaseModel):
    """
    The Event class defines a named event that can be dispatched and 
    processed with before, on, and after handlers.
    """
    name: str
    signature: Signature

    def __hash__(self):
        return hash((self.name, self.signature,))

    def __str__(self):
        return self.name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.__str__() == other
        elif isinstance(other, Event):
            return self.name == other.name and self.signature == other.signature
        return super().__eq__(other)
        
    class Config:
        arbitrary_types_allowed = True


EventCallable = TypeVar("EventCallable", bound=Callable[..., Any])


class Preposition(Flag):
    BEFORE = auto()
    ON = auto()
    AFTER = auto()

    def __str__(self):
        if self == Preposition.BEFORE:
            return "before"
        elif self == Preposition.ON:
            return "on"
        elif self == Preposition.AFTER:
            return "after"


class EventContext(BaseModel):
    event: Event
    preposition: Preposition
    
    def is_before() -> bool:
        return self.preposition == Preposition.BEFORE

    def is_on() -> bool:
        return self.preposition == Preposition.ON
    
    def is_after() -> bool:
        return self.preposition == Preposition.AFTER
    
    def __str__(self):
        return f"{self.preposition}:{self.event.name}"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return other in (self.__str__(), self.event.name)
        return super().__eq__(other)


class EventHandler(BaseModel):
    event: Event
    preposition: Preposition
    kwargs: Dict[str, Any]
    connector_type: Optional[Type["Connector"]]
    handler: EventCallable

    def __str__(self):
        return f"{self.connector_type}({self.preposition}:{self.event}->{self.handler})"


class EventResult(BaseModel):
    """
    Encapsulates the result of a dispatched Connector event
    """

    event: Event
    preposition: Preposition
    handler: EventHandler
    connector: "Connector"
    created_at: datetime = None
    value: Any

    @validator("created_at", pre=True, always=True)
    def set_created_at_now(cls, v):
        return v or datetime.now()


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


def create_event(name: str, signature: Union[Callable, Signature]) -> Event:
    """
    Create an event programmatically from a name and function signature.
    """
    if _events.get(name, None):
        raise ValueError(f"Event '{name}' has already been created")

    signature = (
        signature
        if isinstance(signature, Signature)
        else Signature.from_callable(signature)
    )
    if list(
        filter(
            lambda param: param.kind == Parameter.VAR_POSITIONAL,
            signature.parameters.values(),
        )
    ):
        raise TypeError(
            f"Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
        )

    event = Event(name=name, signature=signature)
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
        create_event(event_name, fn)

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
        handler_signature = Signature.from_callable(fn)

        if preposition == Preposition.BEFORE:
            before_handler_signature = Signature.from_callable(__before_handler)
            _validate_handler_signature(
                handler_signature,
                event_signature=before_handler_signature,
                handler_name=name,
            )
        elif preposition == Preposition.ON:
            _validate_handler_signature(
                handler_signature, event_signature=event.signature, handler_name=name
            )
        elif preposition == Preposition.AFTER:
            after_handler_signature = Signature.from_callable(__after_handler)
            _validate_handler_signature(
                handler_signature,
                event_signature=after_handler_signature,
                handler_name=name,
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


def _validate_handler_signature(
    handler_signature: Signature, *, event_signature: Signature, handler_name: str
) -> None:
    """
    Validates that the given handler signature is compatible with the event signature. Validation
    checks the parameter and return value types using annotations. The intent is to immediately
    expose errors in event handlers rather than encountering them at runtime (which may take 
    an arbitrary amount of time to trigger a given event). Raises a TypeError when an incompatibility 
    is encountered.

    :param handler_signature: The event handler signature to validate.
    :param event_signature: The reference event signature to validate against.
    :param handler_name: The name of the handler for inclusion in error messages & logs.
    """

    # Skip the work if the signatures are identical
    if handler_signature == event_signature:
        return

    handler_parameters: Mapping[str, Parameter] = handler_signature.parameters
    handler_positional_parameters = list(
        filter(
            lambda param: param.kind
            in [Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL],
            handler_parameters.values(),
        )
    )
    handler_keyword_parameters = dict(
        filter(
            lambda item: item[1].kind
            in [
                Parameter.KEYWORD_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_KEYWORD,
            ],
            handler_parameters.items(),
        )
    )

    event_parameters: Mapping[str, Parameter] = event_signature.parameters
    event_positional_parameters = list(
        filter(
            lambda param: param.kind
            in [Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL],
            event_parameters.values(),
        )
    )
    event_keyword_parameters = dict(
        filter(
            lambda item: item[1].kind
            in [
                Parameter.KEYWORD_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_KEYWORD,
            ],
            event_parameters.items(),
        )
    )

    # We assume instance methods
    args = list(handler_parameters.keys())
    first_arg = args.pop(0) if args else None
    if first_arg != "self":
        raise TypeError(
            f"Invalid signature for '{handler_name}' event handler: {handler_signature}, \"self\" must be the first argument"
        )

    # Check return type annotation
    if handler_signature.return_annotation != event_signature.return_annotation:
        raise TypeError(
            f"Invalid return type annotation for '{handler_name}' event handler: expected {event_signature.return_annotation}, but found {handler_signature.return_annotation}"
        )

    # Check for extraneous positional parameters on the handler
    handler_positional_only = list(
        filter(
            lambda param: param.kind == Parameter.POSITIONAL_ONLY,
            handler_positional_parameters,
        )
    )
    event_positional_only = list(
        filter(
            lambda param: param.kind == Parameter.POSITIONAL_ONLY,
            event_positional_parameters,
        )
    )
    if len(handler_positional_only) > len(event_positional_only):
        extra_param_names = sorted(
            list(
                set(map(lambda p: p.name, handler_positional_only))
                - set(map(lambda p: p.name, event_positional_only))
            )
        )
        raise TypeError(
            f"Invalid type annotation for '{handler_name}' event handler: encountered extra positional parameters ({join_to_series(extra_param_names)})"
        )

    # Check for extraneous keyword parameters on the handler
    handler_keyword_nonvar = dict(
        filter(
            lambda item: item[1].kind != Parameter.VAR_KEYWORD,
            handler_keyword_parameters.items(),
        )
    )
    event_keyword_nonvar = dict(
        filter(
            lambda item: item[1].kind != Parameter.VAR_KEYWORD,
            event_keyword_parameters.items(),
        )
    )
    extraneous_keywords = sorted(
        list(set(handler_keyword_nonvar.keys()) - set(event_keyword_nonvar.keys()))
    )
    if extraneous_keywords:
        raise TypeError(
            f"Invalid type annotation for '{handler_name}' event handler: encountered extra parameters ({join_to_series(extraneous_keywords)})"
        )

    # Iterate the event signature parameters and see if the handler's signature satisfies each one
    for index, (parameter_name, event_parameter) in enumerate(event_parameters.items()):
        if event_parameter.kind == Parameter.POSITIONAL_ONLY:
            if index > len(handler_positional_parameters) - 1:
                if handler_positional_parameters[-1].kind != Parameter.VAR_POSITIONAL:
                    raise TypeError(
                        f"Missing required positional parameter: '{parameter_name}'"
                    )

            handler_parameter = handler_positional_parameters[index]
            if handler_parameter != Parameter.VAR_POSITIONAL:
                # Compare types
                if handler_parameter.annotation != event_parameter.annotation:
                    raise TypeError(
                        f"Incorrect type annotation for positional parameter '{parameter_name}': expected {event_parameter.annotation}, but found {handler_parameter.annotation}"
                    )

                if (
                    handler_parameter.return_annotation
                    != event_parameter.return_annotation
                ):
                    raise TypeError(
                        f"Incorrect return type annotation for positional parameter '{parameter_name}': expected {event_parameter.return_annotation}, but found {handler_parameter.return_annotation}"
                    )

        elif event_parameter.kind == Parameter.VAR_POSITIONAL:
            # NOTE: This should never happen
            raise TypeError(
                "Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
            )

        elif event_parameter.kind in [
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ]:
            if handler_parameter := handler_keyword_parameters.get(
                parameter_name, None
            ):
                # We have the keyword arg, check the types
                if handler_parameter.annotation != event_parameter.annotation:
                    raise TypeError(
                        f"Incorrect type annotation for parameter '{parameter_name}': expected {event_parameter.annotation}, but found {handler_parameter.annotation}"
                    )
            else:
                # Check if the last parameter is a VAR_KEYWORD
                if (
                    list(handler_keyword_parameters.values())[-1].kind
                    != Parameter.VAR_KEYWORD
                ):
                    raise TypeError(
                        f"Missing required parameter: '{parameter_name}': expected signature: {event_signature}"
                    )

        else:
            assert event_parameter.kind == Parameter.VAR_KEYWORD, event_parameter.kind
