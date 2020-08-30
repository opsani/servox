from typing import Any, List, Tuple, Union

__all__ = [
    "values_for_keys",
    "value_for_key_path"
]

def values_for_keys(obj: dict, *keys: Union[List[str], Tuple[str]]) -> List[Any]:
    """
    Returns a list of values from an object with the given keys.
    """
    return list(map(obj.get, *keys))

DEFAULT_SENTINEL = object()

def value_for_key_path(obj: object, key_path: str, default: Any = DEFAULT_SENTINEL) -> Any:
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
        # return child[child_key]
    else:
        if default is not DEFAULT_SENTINEL:
            return default
        else:
            raise ValueError(f"unknown key-path '{key_path}'")
