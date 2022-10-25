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

"""Support for scheduling asynchronous, repeating tasks within Opsani Servo connectors.

The `servo.repeating.Mixin` provides connectors with the ability to easily manage tasks
that require periodic execution or the observation of particular runtime conditions.
"""
import asyncio
import time
from sys import float_info
from typing import Callable, Dict, Optional, Union, Awaitable

import pydantic

from servo.types import Duration, NoneCallable, Numeric

__all__ = ["Every", "Mixin", "repeating"]

Every = Union[Numeric, str, Duration]


class Mixin(pydantic.BaseModel):
    """Provides convenience interfaces for working with asyncrhonously repeating tasks."""

    __private_attributes__ = {
        "_repeating_tasks": pydantic.PrivateAttr({}),
    }

    def __init_subclass__(cls, **kwargs) -> None:  # noqa: D105
        super().__init_subclass__(**kwargs)

        repeaters = {}
        for name, method in cls.__dict__.items():
            if repeat_params := getattr(method, "__repeating__", None):
                repeaters[method] = repeat_params

        cls.__repeaters__ = repeaters

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        super().__init__(*args, **kwargs)

        # Start tasks for any methods decorated via `repeating`
        for method, repeat_params in self.__class__.__repeaters__.items():
            if repeat_params := getattr(method, "__repeating__", None):
                name, duration = repeat_params["name"], repeat_params["duration"]
                self.start_repeating_task(name, duration, method)

    def start_repeating_task(
        self,
        name: str,
        every: Every,
        function: Union[Callable[[None], None], Awaitable[None]],
        time_correction: bool = False,
    ) -> asyncio.Task:
        """Start a repeating task with the given name and duration.

        Args:
            name: A name for identifying the repeating task.
            every: The duration at which the task will repeatedly run.
            function: A callable to be executed repeatedly on the desired interval.
            time_correction: Whether to modify sleep time according to wall time taken to run
            <function>. If True, e.g. if function took 1s and every is 5s, then sleep for 4s.
            Otherwise sleep for e.g. 5s regardless.
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
                t0 = time.time()
                if asyncio.iscoroutinefunction(function):
                    await function()
                elif callable(function):
                    function()
                else:
                    if asyncio.iscoroutine(function):
                        TypeError(
                            f"function={function} must be Awaitable or Callable, but has "
                            f"type(function)={type(function)}. A developer may have invoked the "
                            "async function while/prior to passing it in here."
                        )

                    raise TypeError(
                        f"function={function} must be Awaitable or Callable, but has "
                        f"type(function)={type(function)}."
                    )
                t1 = time.time()
                sleep_time = every.total_seconds()
                if time_correction:
                    sleep_time = max(sleep_time - (t1 - t0), 0)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(
                        float_info.epsilon
                    )  # this is a concurrency failsafe.

        asyncio_task = asyncio.create_task(repeating_async_fn(), name=task_name)
        self._repeating_tasks[name] = asyncio_task
        return asyncio_task

    def cancel_repeating_task(self, name: str) -> Optional[asyncio.Task]:
        """Cancel a repeating task with the given name.

        Returns the asyncio task if one was found else None.

        Note that cancellation is not guaranteed (see asyncio.Task docs: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel)
        """
        if task := self.repeating_tasks.get(name):
            if not task.done():
                task.cancel()

            return task

        return None

    def cancel_repeating_tasks(self) -> Dict[str, asyncio.Task]:
        """Cancel a repeating task with the given name.

        Returns a Dictionary of repeating task names to asyncio.Task objects.

        Note that cancellation is not guaranteed (see asyncio.Task docs: https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.cancel)
        """
        repeating_tasks: Dict[str, asyncio.Task] = self.repeating_tasks.copy()
        for task in repeating_tasks.values():
            if not task.done():
                task.cancel()

        return repeating_tasks

    @property
    def repeating_tasks(self) -> Dict[str, asyncio.Task]:
        """Return a dictionary of repeating tasks keyed by task name."""
        return self._repeating_tasks


def repeating(every: Every, *, name=None) -> Callable[[NoneCallable], NoneCallable]:
    """Decorate a function for repeated execution on a given duration.

    Note that the decorated function must be a method on a subclass of `servo.repeating.Mixin` or
    the decoration will have no effect.
    """

    def decorator(fn: NoneCallable) -> NoneCallable:
        duration = every if isinstance(every, Duration) else Duration(every)
        repeater_name = name if name else fn.__name__
        fn.__repeating__ = {"duration": duration, "name": repeater_name}
        return fn

    return decorator
