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

"""Miscellaneous utility functions for working with strings.
"""

from typing import Optional, Pattern, Sequence, Union
import re


def join_to_series(
    items: Sequence[str], *, conjunction: str = "and", oxford_comma: bool = True
) -> str:
    """
    Concatenate a sequence of strings into a series suitable for use in English output.

    Items are joined using a comma and a configurable conjunction, defaulting to 'and'.
    """
    count = len(items)
    if count == 0:
        return ""
    elif count == 1:
        return items[0]
    elif count == 2:
        return f" {conjunction} ".join(items)
    else:
        series = ", ".join(items[0:-1])
        last_item = items[-1]
        delimiter = "," if oxford_comma else ""
        return f"{series}{delimiter} {conjunction} {last_item}"


def commandify(module_path: str) -> str:
    """Transform an input string into a command name usable in a CLI."""
    # foo.bar.this_key => this-key
    return module_path.split(".", 1)[-1].replace("_", "-").lower()


def parse_re(
    value: Optional[list[str]],
) -> Union[None, list[str], Pattern[str]]:
    if value and len(value) == 1:
        val = value[0]
        if val[:1] == "/" and val[-1] == "/":
            return re.compile(val[1:-1])

    return value


def parse_csv(
    value: Optional[list[str]],
) -> Union[None, list[str], Pattern[str]]:
    if value and len(value) == 1:
        val = value[0]
        if "," in val:
            return list(map(lambda v: v.strip(), val.split(",")))

    return value


def parse_id(
    value: Optional[list[str]],
) -> Union[None, list[str], Pattern[str]]:
    v = parse_re(value)
    if not isinstance(v, Pattern):
        return parse_csv(v)

    return v


def check_status_to_str(check) -> str:
    if check.success:
        return "√ PASSED"
    else:
        if check.warning:
            return "! WARNING"
        else:
            return "X FAILED"
