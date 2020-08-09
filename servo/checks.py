import asyncio
from typing import List
from pydantic import BaseModel
from servo.configuration import BaseConfiguration
from servo.types import Check


__all__ = [
    "BaseChecks"
]


class BaseChecks(BaseModel):
    """
    Base class for collections of Servo Connector check implementations.

    This is a convenience class for quickly and cleanly implementing checks
    for a connector. A check is an atomic verification that a particular aspect
    of a configuration is functional and ready for deployment. Connectors can
    have an arbitrary number of prerequisites and options that need to be
    checked within the runtime environment. The BaseChecks class provides a simple 
    inheritance based interface for implementing an arbitrary number of checks.
    
    Checks are implemented through standard instance methods that are prefixed
    with `check_`, accept no arguments, and return an instance of `Check`. The method
    body tests a single aspect of the configuration (rescuing exceptions as necessary)
    and returns a `Check` object that models the results of the check performed.

    Checks are executed in method definition order within the subclass (top to bottom).
    Check methods can be implemented synchronously or asynchronously. Methods that are
    declared as coroutines via the `async def` syntax are run asynchronously.

    By default, check execution is halted upon encountering a failure. This behavior
    allows the developer to assume that the runtime environment described by preceding
    checks has been established and implement a narrowly scoped check. Halting execution
    can be overridden via the boolean `all` argument to the `run` class method entry point 
    or by setting the `required` attribute of the returned `Check` instance to `False`.

    Args:
        config: The configuration object for the connector being checked.
    """
    config: BaseConfiguration

    @classmethod
    async def run(cls, config: BaseConfiguration, *, all: bool = False) -> List[Check]:
        """
        Run all checks and return a list of Check objects reflecting the results.

        Checks are implemented as instance methods prefixed with `check_` that return a `Check`
        object. Please refer to the `BaseChecks` class documentation for details.

        Args:
            all: When True, all checks are run regardless of previous failures or being required.
        
        Returns:
            A list of `Check` objects that reflect the outcomes of the checks executed.
        """
        checker = cls(config=config)
        checks: List[Check] = []
        for attr in dir(checker):
            if not attr.startswith("check_"):
                continue

            method = getattr(checker, attr)
            if callable(method):
                check = (
                    await method() if asyncio.iscoroutinefunction(method)
                    else method()
                )
                if not isinstance(check, Check):
                    raise AssertionError(f"check implementations must return `Check` instances: `{attr}` returned `{check.__class__.__name__}`")
                checks.append(check)                
        
        return checks
