import asyncio
import datetime
import weakref
from typing import Callable, Dict, Optional, Union

import servo.utilities.key_paths
from servo.types import Duration, NoneCallable, Numeric

__all__ = ["Every", "Mixin", "repeating"]

Every = Union[Numeric, str, Duration]

_repeating_tasks_registry = weakref.WeakKeyDictionary()


class Mixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        repeaters = {}
        for name, method in cls.__dict__.items():
            if repeat_params := getattr(method, "__repeating__", None):
                repeaters[method] = repeat_params

        cls.__repeaters__ = repeaters

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        repeating_tasks: weakref.WeakValueDictionary[
            str, asyncio.Task
        ] = weakref.WeakValueDictionary()
        _repeating_tasks_registry[self] = repeating_tasks

        # Start tasks for any methods decorated via `repeating`
        for method, repeat_params in self.__class__.__repeaters__.items():
            if repeat_params := getattr(method, "__repeating__", None):
                name, duration = servo.utilities.key_paths.values_for_keys(
                    repeat_params, "name", "duration"
                )
                self.start_repeating_task(name, duration, method)

    def start_repeating_task(
        self, name: str, every: Every, callable: Callable[[None], None]
    ) -> asyncio.Task:
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
                await asyncio.sleep(every / datetime.timedelta(microseconds=1))

        asyncio_task = asyncio.create_task(repeating_async_fn(), name=task_name)
        self.repeating_tasks[name] = asyncio_task
        return asyncio_task

    def cancel_repeating_task(self, name: str) -> Optional[bool]:
        """
        Cancel a repeating task with the given name.

        Returns True if the task was cancelled, False if it was found but could not be cancelled, or None if
        no task with the given name could be found.

        Note that cancellation is not guaranteed (see asyncio.Task docs: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel)
        """
        if task := self.repeating_tasks.get(name):
            if task.cancelled():
                return False
            task.cancel()
            return True

        return None

    @property
    def repeating_tasks(self) -> Dict[str, asyncio.Task]:
        tasks = _repeating_tasks_registry.get(self, None)
        if tasks is None:
            tasks = {}
            _repeating_tasks_registry[self] = tasks

        return tasks


def repeating(every: Every, *, name=None) -> Callable[[NoneCallable], NoneCallable]:
    """
    Decorates a function for repeated execution on the given duration.

    Note that the decorated function must be a method on a subclass of `servo.repeating.Mixin` or
    the decoration will have no effect.
    """

    def decorator(fn: NoneCallable) -> NoneCallable:
        duration = every if isinstance(every, Duration) else Duration(every)
        repeater_name = name if name else fn.__name__
        fn.__repeating__ = {"duration": duration, "name": repeater_name}
        return fn

    return decorator
