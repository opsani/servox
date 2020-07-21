from typing import Any, List, Tuple, Union


def values_for_keys(obj: dict, *keys: Union[List[str], Tuple[str]]) -> List[Any]:
    """
    Returns a list of values from an object with the given keys.
    """
    return list(map(obj.get, *keys))
