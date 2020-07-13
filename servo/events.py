from datetime import datetime
from enum import Flag, auto
from inspect import Signature
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, validator


class Event(BaseModel):
    name: str
    signature: Signature

    def __hash__(self):
        return hash((self.name, self.signature,))

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


class EventHandler(BaseModel):
    event: Event
    preposition: Preposition
    kwargs: Dict[str, Any]
    connector_type: Optional[Type["Connector"]]
    handler: EventCallable

    def __str__(self):
        return f"{self.preposition} {self.event}"


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
