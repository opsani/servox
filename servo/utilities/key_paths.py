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

from typing import Any, Sequence

__all__ = ["values_for_keys", "value_for_key_path"]


def values_for_keys(obj: dict[Any, Any], *keys: Sequence[str]) -> list[Any]:
    """
    Return a list of values from an object with a given sequence of keys.
    """
    return list(map(obj.get, *keys))


DEFAULT_SENTINEL = object()


def value_for_key_path(obj: Any, key_path: str, default: Any = DEFAULT_SENTINEL) -> Any:
    """Return the value of a property at a given key path relative to a given object.

    Args:
        obj: The object to evaluate the key path against.
        key_path: A period delimited string that identifies the property to be returned.
        default: An optional default value to be returned if the value could not be found.

    Raises:
        ValueError: Raised if a value could not be found for the given key path.

    Returns:
        Any: The value at the given key path.
    """
    if hasattr(obj, key_path):
        if default is not DEFAULT_SENTINEL:
            return getattr(obj, key_path, default)
        else:
            return getattr(obj, key_path)
    elif key_path in obj:
        return obj[key_path]
    elif "." in key_path:
        parent_key, child_key = key_path.split(".", 2)
        child = value_for_key_path(obj, parent_key, default)
        return value_for_key_path(child, child_key, default)
    else:
        if default is not DEFAULT_SENTINEL:
            return default
        else:
            raise ValueError(f"unknown key-path '{key_path}'")
