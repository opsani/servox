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

import hashlib
from typing import Any, Callable, Union


def get_hash(data: Union[list[Any], dict[Any, Any]]) -> str:
    """md5 hash of Python data. This is limited to scalars that are convertible to string and container
    structures (list, dict) containing such scalars. Some data items are not distinguishable, if they have
    the same representation as a string, e.g., hash(b'None') == hash('None') == hash(None)
    """
    hasher = hashlib.md5()
    dump_container(data, hasher.update)
    return hasher.hexdigest()


def dump_container(
    c: Union[str, bytes, list[Any], dict[Any, Any]], func: Callable[[Any], Any]
) -> None:
    """stream the contents of a container as a string through a function
    in a repeatable order, suitable, e.g., for hashing
    """
    #
    if isinstance(c, dict):  # dict
        func("{".encode("utf-8"))
        for k in sorted(c):  # for all repeatable
            func("{}:".format(k).encode("utf-8"))
            dump_container(c[k], func)
            func(",".encode("utf-8"))
        func("}".encode("utf-8"))
    elif isinstance(c, list):  # list
        func("[".encode("utf-8"))
        for k in c:  # for all repeatable
            dump_container(k, func)
            func(",".encode("utf-8"))
        func("]".encode("utf-8"))
    else:  # everything else
        if isinstance(c, type(b"")):
            pass  # already a stream, keep as is
        elif isinstance(c, str):
            # encode to stream explicitly here to avoid implicit encoding to ascii
            c = c.encode("utf-8")
        else:
            c = str(c).encode("utf-8")  # convert to string (e.g., if integer)
        func(c)  # simple value, string or convertible-to-string
