from contextlib import contextmanager
from typing import Any, Callable, Generator, Type

from pydantic import BaseModel, Extra
from pydantic.validators import _VALIDATORS


def prepend_pydantic_validator(
    type_: Type[Any], validator: Callable[[Any], Any]
) -> None:
    for _validator in _VALIDATORS:
        if _validator[0] == type_:
            _validator[1].insert(0, validator)


def append_pydantic_validator(
    type_: Type[Any], validator: Callable[[Any], Any]
) -> None:
    for _validator in _VALIDATORS:
        if _validator[0] == type_:
            _validator[1].append(validator)

@contextmanager
def extra(obj: BaseModel, extra: Extra = Extra.allow) -> Generator[BaseModel, None, None]:
    """Temporarily overrides the value of the `extra` setting on a Pydantic model.
    """
    original = obj.__config__.extra
    obj.__config__.extra = extra
    yield obj
    obj.__config__.extra = original
