"""Miscellaneous utility functions for working with strings.
"""

from typing import Sequence


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
    """Transform an input string into a command name usable in a CLI.
    """
    # foo.bar.this_key => this-key
    return module_path.split(".", 1)[-1].replace("_", "-").lower()
