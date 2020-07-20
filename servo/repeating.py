import asyncio
from typing import Union, Callable, Optional, List, Dict
from servo.types import Duration, NoneCallable
from servo.utilities import values_for_keys
from datetime import timedelta
from weakref import WeakKeyDictionary, WeakValueDictionary

__repeaters = WeakKeyDictionary()
Every = Union[int, str, Duration]

class RepeatingMixin:
    def __init__(self, *args, **kwargs) -> None:
        self._repeating_tasks: Dict[str, asyncio.Task] = WeakValueDictionary()

        # Start tasks for any methods decorated via `repeating`
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        for method_name in method_names:
            method = getattr(self, method_name)
            if repeat_params := getattr(method, "__repeating__", None):
                name, duration = values_for_keys(repeat_params, "name", "duration")
                self.start_repeating_task(name, duration, method)

    def start_repeating_task(self, name: str, every: Every, callable: Callable[[None], None]) -> asyncio.Task:
        """
        Starts a repeating task with given name to repeatedly execute on `every` duration.
        """
        if task := self.repeating_tasks.get(name, None):
            if not task.done():
                # Task may be done but hasn't dropped from our index
                raise KeyError(f"repeating task already exists named '{name}'")
        
        every = every if isinstance(every, Duration) else Duration(every)
        context_name = getattr(self, "name", self.__class__.__name__)
        task_name = f"{context_name}:{name} (repeating every {every})"
        
        async def repeating_async_fn() -> None:
            while True:
                callable()
                await asyncio.sleep(every / timedelta(microseconds=1))
        
        asyncio_task = asyncio.create_task(repeating_async_fn(), name=task_name)
        self._repeating_tasks[name] = asyncio_task
        return asyncio_task
    
    def cancel_repeating_task(self, name: str) -> Optional[bool]:
        """
        Cancel a repeating task with the given name.
        
        Returns True if the task was cancelled, False if it was found but could not be cancelled, or None if
        no task with the given name could be found.

        Note that cancellation is not guaranteed (see asyncio.Task docs: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel)
        """
        if task := self._repeating_tasks.get(name):
            if task.cancelled():
                return False
            task.cancel()
            return True
        return None

    @property
    def repeating_tasks(self) -> List[asyncio.Task]:
        return self._repeating_tasks


def repeating(every: Every, *, name=None) -> Callable[[NoneCallable], NoneCallable]:
    """
    Decorates a function for repeated execution on the given duration.

    Note that the decorated function must be a method on a subclass of `RepeatingMixin` or
    the decoration will have no effect.
    """

    def decorator(fn: NoneCallable) -> NoneCallable:
        duration = every if isinstance(every, Duration) else Duration(every)
        repeater_name = name if name else fn.__name__
        fn.__repeating__ = { "duration": duration, "name": repeater_name }
        return fn

    return decorator
