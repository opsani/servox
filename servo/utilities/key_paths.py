from typing import Any, List, Tuple, Union


def values_for_keys(obj: dict, *keys: Union[List[str], Tuple[str]]) -> List[Any]:
    """
    Returns a list of values from an object with the given keys.
    """
    return list(map(obj.get, *keys))

def value_for_key_path(obj: dict, key_path: str) -> Any:
    if hasattr(obj, key_path):    
        return getattr(obj, key_path)
    elif key_path in obj:
        return obj[key_path]
    elif "." in key_path:
        parent_key, child_key = key_path.split(".", 2)
        child = value_for_key_path(obj, parent_key)
        return value_for_key_path(child, child_key)
        # return child[child_key]
    else:
        raise ValueError(f"unknown key '{key_path}'")
