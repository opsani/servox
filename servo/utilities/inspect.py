import inspect
import warnings
from collections import ChainMap
from typing import Any, Dict, Callable, List, Optional, Type, Tuple, cast

def get_instance_methods(obj, *, stop_at_parent: Optional[Type[Any]] = None) -> Dict[str, Callable]:
    """
    Returns a mapping of method names to method callables in definition order, optionally traversing
    the inheritance hierarchy in method dispatch order.

    Note that the semantics of the values in the dictionary returned are dependent on the input object.
    When `obj` is an object instance, the values are bound method objects (as returned by `get_methods`).
    When `obj` is a class, the values are unbound function objects. Depending on what you are trying to
    do, this may have interesting ramifications (for example, the method signature of the callable will
    include `self` in the parameters list). This behavior is a side-effect of the lookup implementation
    which is utilized because it retains method definition order. To obtain a bound method object reference, 
    go through `get_methods` or call `getattr` on an instance.

    Args:
        obj: The object or class to retrieve the instance methods for.
        stop_at_parent: The parent class to halt the inheritance traversal at. When None, only
            instance methods of `obj` are returned.
    
    Returns:
        A dictionary of methods in definition order.
    """
    cls = obj if inspect.isclass(obj) else obj.__class__
    methods = ChainMap()
    stopped = False

    # search for instance specific methods before traversing the class hierarchy
    if not inspect.isclass(obj):
        methods.maps.append(
            dict(filter(lambda item: inspect.ismethod(item[1]), obj.__dict__.items()))
        )

    for c in inspect.getmro(cls):
        methods.maps.append(
            dict(filter(lambda item: inspect.isfunction(item[1]), c.__dict__.items()))
        )
        if not stop_at_parent or c == stop_at_parent:
            stopped = True
            break
    
    if not stopped:
        raise TypeError(f'invalid parent type "{stop_at_parent}": not found in inheritance hierarchy')
    
    if isinstance(obj, cls):
        # Update the values to bound method references
        return dict(map(lambda name: (name, getattr(obj, name)), methods.keys()))
    else:
        return cast(dict, methods)


def get_methods(cls: Type[Any]) -> List[Tuple[str, Any]]:
    """
    Return a list of tuple of methods for the given class in alphabetical order.

    Args:
        cls: The class to retrieve the methods of.
    
    Returns:
        A list of tuples containing method names and bound method objects.
    """
    # retrieving the members can emit deprecation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return inspect.getmembers(cls, inspect.ismethod)


def get_defining_class(method: Callable) -> Optional[Type[Any]]:
    """
    Return the class that defined the given method.

    Args:
        method: The method to return the defining class of.
    
    Return:
        The class that defined the method or None if not determined.
    """
    for cls in inspect.getmro(method.__self__.__class__):
        if method.__name__ in cls.__dict__:
            return cls

    meth = getattr(method, '__func__', method)  # fallback to __qualname__ parsing
    cls = getattr(inspect.getmodule(method),
                method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                None)
    if isinstance(cls, type):
        return cls

    return None
