import contextlib
from typing import Any, Callable, Generator, Type

import pydantic
import pydantic.validators

__all__ = [
    "prepend_pydantic_validator",
    "append_pydantic_validator",
    "extra"
]


def prepend_pydantic_validator(
    type_: Type[Any], validator: Callable[[Any], Any]
) -> None:
    """Prepend a validator type to the list of standard Pydantic validators.

    Prepending validators can override default behaviors provided by Pydantic.
    """
    for _validator in pydantic.validators._VALIDATORS:
        if _validator[0] == type_:
            _validator[1].insert(0, validator)


def append_pydantic_validator(
    type_: Type[Any], validator: Callable[[Any], Any]
) -> None:
    """Append a validator to the list of standard Pydantic validators.

    Appending a validator will introduce new behavior common to all Pydantic models.
    """
    for _validator in pydantic.validators._VALIDATORS:
        if _validator[0] == type_:
            _validator[1].append(validator)


@contextlib.contextmanager
def extra(
    obj: pydantic.BaseModel, extra: pydantic.Extra = pydantic.Extra.allow
) -> Generator[pydantic.BaseModel, None, None]:
    """Temporarily override the value of the `extra` setting on a Pydantic model."""
    original = obj.__config__.extra
    obj.__config__.extra = extra
    try:
        yield obj
    finally:
        obj.__config__.extra = original


@contextlib.contextmanager
def allow_mutation(
    obj: pydantic.BaseModel, allow_mutation: bool = True
) -> Generator[pydantic.BaseModel, None, None]:
    """Temporarily override the value of the `allow_mutation` setting on a Pydantic model."""
    original = obj.__config__.allow_mutation
    obj.__config__.allow_mutation = allow_mutation
    try:
        yield obj
    finally:
        obj.__config__.allow_mutation = original
