from typing import Iterable

def join_to_series(
    items: Iterable[str], *, conjunction: str = "and", oxford_comma: bool =True
) -> str:
    """
    Concatenate any number of strings into a series suitable for use in English output.

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
