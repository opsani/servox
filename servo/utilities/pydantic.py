from typing import Any, Callable, Type

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
