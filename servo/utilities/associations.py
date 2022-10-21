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

"""Associations are virtual attributes maintained outside of an object instance.

Associations provide the ability to manage state for an object without polluting
its namespace. They are very useful for supporting Pydantic models without
introducing new attributes that need to be considered in the schema and validation
logic.
"""

import weakref
from typing import Any, Protocol, runtime_checkable

_associations: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class Mixin:
    """Provides support for virtual attributes."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        # NOTE: we are not hashable until after init
        super().__init__(*args, **kwargs)
        _associations[self] = {}

    def _set_association(self, name: str, obj: Any) -> None:
        """Set an object association by name.

        Args:
            name: A name for the association.
            obj: The object to associate with.
        """
        _associations[self][name] = obj

    def _get_association(self, name: str, default: Any = ...) -> Any:
        """Return an associated object by name.

        Args:
            name: The name of the association to retrieve.
            default: A default value to return instead of raising a `KeyError` if
                the association cannot be found.

        Returns:
            The associated object.

        Raises:
            KeyError: Raised if there is no associated object with the given name.
        """
        if default == ...:
            return _associations[self][name]
        else:
            return _associations[self].get(name, default)

    @property
    def _associations(self) -> dict[str, Any]:
        """Return all associated objects as a dictionary.

        Returns:
            A dictionary mapping of associated object names and values.
        """
        return _associations[self]


@runtime_checkable
class Associative(Protocol):  # pragma: no cover
    """A protocol that describes objects that support associations."""

    def _set_association(self, name: str, obj: Any) -> None:
        ...

    def _get_association(self, name: str, default: Any = ...) -> Any:
        ...

    def _associations(self) -> dict[str, Any]:
        ...
