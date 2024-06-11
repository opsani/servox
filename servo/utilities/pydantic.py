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

import contextlib
from typing import Any, Callable, Generator, Type

import pydantic
import pydantic.validators

__all__ = ["prepend_pydantic_validator", "append_pydantic_validator", "extra"]


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
def model_config_override(
    obj: pydantic.BaseModel, values: dict
) -> Generator[pydantic.BaseModel, None, None]:
    """Temporarily override the values on a Pydantic model config."""
    obj.model_config.update(**values)
    try:
        yield obj
    finally:
        for k in values.keys():
            obj.model_config.pop(k)


@contextlib.contextmanager
def extra(
    obj: pydantic.BaseModel, extra: str = "allow"
) -> Generator[pydantic.BaseModel, None, None]:
    """Temporarily override the value of the `extra` setting on a Pydantic model."""
    with model_config_override(obj, {"extra": extra}):
        # dynamicly changing extra isn't actually supported. have to tap into undelrying implementation to set field
        #   normally set during BaseModel.model_construct
        if obj.__pydantic_extra__ is None:
            obj.__pydantic_extra__ = {}
        yield obj


@contextlib.contextmanager
def allow_mutation(
    obj: pydantic.BaseModel, allow_mutation: bool = True
) -> Generator[pydantic.BaseModel, None, None]:
    """Temporarily override the value of the `allow_mutation` setting on a Pydantic model."""
    with model_config_override(obj, {"allow_mutation": allow_mutation}):
        yield obj
