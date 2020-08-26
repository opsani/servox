import asyncio

from datetime import datetime
from hashlib import blake2b
from inspect import Signature, isclass
from typing import Callable, Generator, List, Optional, Pattern, Sequence, Set, TypeVar, Tuple, Union, get_origin, get_args

from pydantic import BaseModel, Extra, StrictStr, validator, constr
from servo.configuration import BaseConfiguration
from servo.types import Any, Duration
from servo.utilities.inspect import get_instance_methods

import loguru
from loguru import logger as default_logger


__all__ = [
    "BaseChecks",
    "Check",
    "CheckHandlerResult"
]


CheckHandlerResult = Union[bool, str, Tuple[bool, str], None]
CheckHandler = TypeVar("CheckHandler", bound=Callable[..., CheckHandlerResult])
CHECK_HANDLER_SIGNATURE = Signature(return_annotation=CheckHandlerResult)

Tag = constr(strip_whitespace=True, min_length=1, max_length=32, regex="^([0-9a-z\\.-])*$")


class Check(BaseModel):
    """
    Check objects represent the status of required runtime conditions.

    A check is an atomic verification that a particular aspect
    of a configuration is functional and ready for deployment. Connectors can
    have an arbitrary number of prerequisites and options that need to be
    checked within the runtime environment.

    Checks are used to verify the correctness of servo configuration before
    starting optimization and report on health and readiness during operation.
    """

    name: StrictStr
    """An arbitrary descriptive name of the condition being checked.
    """

    id: StrictStr = None
    """A short identifier for the check. Generated automatically if unset.
    """

    description: Optional[StrictStr]
    """An optional detailed description about the condition being checked.
    """

    required: bool = False
    """
    Indicates if the check is a pre-condition for subsequent checks.

    Required state is used to halt the execution of a sequence of checks
    that are part of a `Checks` object. For example, given a connector
    that connects to a remote service such as a metrics provider, you
    may wish to check that each metrics query is well formed and returns
    results. In order for any of the query checks to succeed, the servo
    must be able to connect to the service. During failure modes such as
    network partitions, service outage, or simple configuration errors
    this can result in an arbitrary number of failing checks with an 
    identical root cause that make it harder to identify the issue.
    Required checks allow you to declare these sorts of pre-conditions
    and the servo will test them before running any dependent checks,
    ensuring that you get a single failure that identifies the root cause.

    For checks that do not belong to a `Checks` object, required is
    purely advisory metadata and is ignored by the servo.
    """

    tags: Optional[Set[Tag]]
    """
    An optional set of tags for filtering checks.

    Tags are strings between 1 and 32 characters in length and may contain 
    only lowercase alphanumeric characters, hyphens '-', and periods '.'.
    """

    success: Optional[bool]
    """
    Indicates if the condition being checked was met or not. 
    """

    message: Optional[StrictStr]
    """
    An optional message describing the outcome of the check.

    The message is presented to users and should be informative. Long
    messages may be truncated on display.
    """

    exception: Optional[Exception]
    """
    An optional exception encountered while running the check.

    When checks encounter an exception condition, it is recommended to
    store the exception so that diagnostic metadata such as the stack trace 
    can be presented to the user.
    """
    
    created_at: datetime = None
    """When the check was created (set automatically).
    """

    run_at: Optional[datetime]
    """An optional timestamp indicating when the check was run.
    """

    runtime: Optional[Duration]
    """An optional duration indicating how long it took for the check to run.
    """

    @classmethod
    def run(cls, name: str, *, handler: CheckHandler, description: Optional[str] = None) -> 'Check':
        """Runs a check handler and returns a Check object reporting the outcome.

        This method is useful for quickly implementing checks in connectors that
        do not have enough checkable conditions to warrant implementing a `Checks`
        subclass.

        Args:
            name: A name for the check being run.
            handler: The callable to run to perform the check. Must accept no arguments
                and return a `bool`, `str`, `Tuple[bool, str]`, or `None`. Boolean values
                indicate success or failure and string values are assigned to the `message`
                attribute of the Check object returned. Exceptions are rescued, mark the check
                as a failure, and assigned to the `exception` attribute.
            description: An optional detailed description about the check being run.

        Returns:
            A check object reporting the outcome of running the handler.
        """
        check = Check(name=name, description=description)
        run_check_handler(check, handler)
        return check

    @property
    def failed(self) -> bool:
        """
        Indicates if the check was unsuccessful.
        """
        return not self.success

    @validator("created_at", pre=True, always=True)
    @classmethod
    def _set_created_at_now(cls, v):
        return v or datetime.now()
    
    @validator("id", pre=True, always=True)
    @classmethod
    def _generated_id(cls, v,  values):        
        return v or blake2b(values["name"].encode('utf-8'), digest_size=4).hexdigest()
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            Exception: lambda v: repr(v),
        }


def run_check_handler(check: Check, handler: CheckHandler, *args):
    """Runs a check handler and records the result into a Check object.

    Args:
        check: The check to record execution results.
        handler: A callable handler to perform the check.

    Raises:
        ValueError: Raised if an invalid value is returned by the handler.
    """
    check.run_at = datetime.now()
    try:
        result = handler(*args)
        check.success = True
        
        if isinstance(result, str):
            check.message = result
        elif isinstance(result, bool):
            check.success = result
        elif isinstance(result, tuple):
            check.success, check.message = result
        elif result is None:
            pass
        else:
            raise ValueError(f"check method returned unexpected value of type \"{result.__class__.__name__}\"")

    except Exception as e:
        check.success = False
        check.exception = e
        check.message = f"caught exception: {repr(e)}"

    check.runtime = Duration(datetime.now() - check.run_at)

CheckRunner = TypeVar("CheckRunner", bound=Callable[..., Check])

def check(
    name: str, 
    *, 
    description: Optional[str] = None,
    id: Optional[str] = None,
    required: bool = False,
    tags: Optional[List[str]] = None) -> Callable[[CheckHandler], CheckRunner]:
    """
    Transforms a function or method into a check.
    
    Checks are used to test the availability, readiness, and health of resources and
    services that used during optimization. The `Check` class models the status of a
    check that has been run. The `check` function is a decorator that transforms a
    function or method that returns a `bool`, `str`, `Tuple[bool, str]`, or `None` 
    into a check function or method.

    The decorator requires a `name` parameter to identify the check as well as an optional
    informative `description`, an `id` for succintly referencing the check, and a `required`
    boolean value that determines if a failure with halt execution of subsequent checks.
    The body of the decorated function is used to perform the business logic of running
    the check. The decorator wraps the original function body into a handler that runs the
    check and marshalls the value returned or exception caught into a `Check` representation.
    The `run_at` and `runtime` properties are automatically set, providing execution timing of
    the check. The signature of the transformed function is `() -> Check`.

    Args:
        name: Human readable name of the check.
        description: Optional additional details about the check.
        id: A short identifier for referencing the check (e.g. from the CLI interface).
        required: When True, failure of the check will halt execution of subsequent checks.
        tags: An optional list of tags for filtering checks. Tags may contain only lowercase
            alphanumeric characters, hyphens '-', and periods '.'.
    
    Returns:
        A decorator function for transforming a function into a check.
    
    Raises:
        TypeError: Raised if the signature of the decorated function is incompatible.
    """
    def decorator(fn: CheckHandler) -> CheckRunner:
        validate_check_handler(fn)
        __check__ = Check(
            name=name, 
            description=description, 
            id=(id or fn.__name__),
            required=required,
            tags=tags,
        )

        # note: use a default value for self to wrap funcs & methods
        def run_check(self: Optional[Any] = None) -> Check:
            check = __check__.copy()
            args = [self] if self else []
            run_check_handler(check, fn, *args)
            return check

        run_check.__check__ = __check__
        return run_check

    return decorator


CHECK_SIGNATURE = Signature(return_annotation=Check)
CHECK_SIGNATURE_ANNOTATED = Signature(return_annotation='Check')


class BaseChecks(BaseModel):
    """
    Base class for collections of Check objects.

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
    """The configuration object for the connector being checked.
    """

    @classmethod
    async def check(cls, 
        config: BaseConfiguration, 
        *, 
        logger: 'loguru.Logger' = default_logger,
        name: Union[None, str, Sequence[str], Pattern[str]] = None,
        id: Union[None, str, Sequence[str], Pattern[str]] = None,
        tags: Optional[Set[str]] = None,
        all: bool = False
    ) -> List[Check]:
        """
        Runs checks and returns a list of Check objects reflecting the results.

        Checks are implemented as instance methods prefixed with `check_` that return a `Check`
        object. Please refer to the `BaseChecks` class documentation for details.

        Keyword arguments are used to filter the set of checks to be run. See the documentation
        on the `run` instance method for details on filtering.

        Args:
            config: The connector configuration to initialize the checks instance with.
            logger: 
            name: A name, sequence of names, or regex pattern for selecting checks by name.
            id:  A name, sequence of names, or regex pattern for selecting checks by name.
            tags: A set of tags for selecting checks to be run. Checks matching any tag in the set
                are selected.
            all: When True, continue running checks even if a required check has failed.
        
        Returns:
            A list of `Check` objects that reflect the outcome of the checks executed.
        """
        return await cls(config, logger=logger).run(name=name, id=id, tags=tags, all=all)

    async def run(self, 
        *, 
        name: Union[None, str, Sequence[str], Pattern[str]] = None,
        id: Union[None, str, Sequence[str], Pattern[str]] = None,
        tags: Optional[Set[str]] = None,
        all: bool = False
    ) -> List[Check]:
        """
        Runs checks and returns the results.

        Specific checks can be targetted for execution using the metadata attributes of `name`,
        `id`, and `tags`. Metadata filters are evaluated using AND semantics. Names and ids
        are matched case-sensitively. Tags are always lowercase. Names and ids can be targetted 
        using regular expression patterns. Checks are evaluated and returned in method definition
        order.

        Args:
            name: A name, sequence of names, or regex pattern for selecting checks by name.
            id:  A name, sequence of names, or regex pattern for selecting checks by name.
            tags: A set of tags for selecting checks to be run. Checks matching any tag in the set
                are selected.
            all: When True, continue running checks even if a required check has failed.
        
        Returns:
            A list of checks that were run.
        """

        class Filter(BaseModel):
            name: Union[None, str, Sequence[str], Pattern[str]] = None,
            id: Union[None, str, Sequence[str], Pattern[str]] = None,
            tags: Optional[Set[str]] = None

            @property
            def any(self) -> bool:
                return not self.empty
            
            @property
            def empty(self) -> bool:
                return bool(
                    self.name is None 
                    and self.id is None 
                    and self.tags is None
                )
            
            def matches(self, check: Check) -> bool:
                if self.empty:
                    return True

                return (                    
                    self._matches_name(check)
                    and self._matches_id(check)
                    and self._matches_tags(check)
                )

            def _matches_name(self, check: Check) -> bool:
                return self._matches_str_attr(self.name, check.name)
            
            def _matches_id(self, check: Check) -> bool:
                return self._matches_str_attr(self.id, check.id)
            
            def _matches_tags(self, check: Check) -> bool:
                if self.tags is None:
                    return True
                
                # exclude untagged checks if filtering by tag
                if check.tags is None:
                    return False
                
                # look for an intersection in our sets
                return bool(self.tags.intersection(check.tags))
            
            def _matches_str_attr(
                self, 
                attr: Union[None, str, Sequence[str], Pattern[str]],
                value: str
            ) -> bool:
                if attr is None:
                    return True
                elif isinstance(attr, str):
                    return value == attr
                elif isinstance(attr, Sequence):
                    return value in attr
                elif isinstance(attr, Pattern):
                    return bool(attr.search(value))
                else:
                    raise ValueError(f"unexpected value of type \"{attr.__class__.__name__}\": {attr}")
        
            class Config:
                arbitrary_types_allowed = True
        
        checks = []
        filter = Filter(name=name, id=id, tags=tags)
        for method_name, method in self.check_methods():
            if filter.any:
                spec = getattr(method, '__check__', None)
                if not spec:
                    self.logger.warning(f"filtering requested but encountered non-filterable check method \"{method_name}\"")
                    continue
                
                if not filter.matches(spec):
                    continue

            check = (
                await method() if asyncio.iscoroutinefunction(method)
                else method()
            )
            if not isinstance(check, Check):
                raise TypeError(f"check methods must return `Check` objects: `{method_name}` returned `{check.__class__.__name__}`")
            
            checks.append(check)

            # halt if a required check has failed
            if check.failed and check.required:
                break
        
        return checks
    
    def check_methods(self) -> Generator[Tuple[str, CheckRunner], None, None]:
        """
        Enumerates all check methods and yields the check method names and callable instances 
        in method definition order.

        Check method names are prefixed with "check_", accept no parameters, and return a
        `Check` object reporting the outcome of the check operation.
        """
        for name, method in get_instance_methods(self).items():
            if not name.startswith(("_", "check_")):
                raise ValueError(f'method names of Checks subtypes must start with "_" or "check_"')
            
            sig = Signature.from_callable(method)
            if sig not in (CHECK_SIGNATURE, CHECK_SIGNATURE_ANNOTATED):
                raise TypeError(f'invalid signature for method "{name}": expected {repr(CHECK_SIGNATURE)}, but found {repr(sig)}')
            
            yield (name, method)

    
    def __init__(self, config: BaseConfiguration, *, logger: 'loguru.Logger' = default_logger, **kwargs) -> None:
        super().__init__(config=config, logger=logger, **kwargs)
    
    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow


def validate_check_handler(fn: CheckHandler) -> None:
    """
    Validates that a function or method is usable as a check handler.

    Check handlers accept no arguments and return a `bool`, `str`, 
    `Tuple[bool, str]`, or `None`.

    Args:
        fn: The check handler to be validated.
    
    Raises:
        TypeError: Raised if the handler function is invalid.
    """
    signature = Signature.from_callable(fn)
    if len(signature.parameters) >= 1:
        for param in signature.parameters.values():
            if param.name == "self" and param.kind == param.POSITIONAL_OR_KEYWORD:
                continue

            raise TypeError(f"invalid check handler \"{fn.__name__}\": unexpected parameter \"{param.name}\" in signature {repr(signature)}, expected {repr(CHECK_HANDLER_SIGNATURE)}")

    error = TypeError(f"invalid check handler \"{fn.__name__}\": incompatible return type annotation in signature {repr(signature)}, expected to match {repr(CHECK_HANDLER_SIGNATURE)}")
    acceptable_types = set(get_args(CheckHandlerResult))
    origin = get_origin(signature.return_annotation)
    args = get_args(signature.return_annotation)
    if origin is not None:
        if origin == Union:
            handler_types = set(args)
            if handler_types - acceptable_types:
                raise error
        elif origin is tuple:
            if args != (bool, str):
                raise error
        else:
            raise error
    else:
        cls = signature.return_annotation if isclass(signature.return_annotation) else signature.return_annotation.__class__
        if not cls in acceptable_types:
            raise error
