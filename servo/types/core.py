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

"""The `servo.types` module defines the essential data types shared by all
consumers of the servo package.
"""
from __future__ import annotations

import abc
import asyncio
import contextlib
import datetime
import enum
import inspect
import operator
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import orjson
import pydantic
import pydantic.error_wrappers
import pygments.lexers
import semver

import servo.utilities.duration_str


def _orjson_dumps(
    v, *, default, indent: Optional[int] = None, sort_keys: bool = False
) -> str:
    """Serializes an input object into JSON via the `orjson` library.

    Returns:
        A JSON string representation of the input object.

    Raises:
        TypeError: Raised if the input object could not be serialized to a JSON representation.
    """
    option = orjson.OPT_PASSTHROUGH_SUBCLASS
    if indent and indent == 2:
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS

    def default_handler(obj) -> Any:
        # TODO hook OpsaniRepr into this as well
        try:
            if isinstance(obj, HumanReadable):
                return obj.human_readable()

            return default(obj)
        except TypeError:
            # TODO increase visibility into this error as this fallback can cause issues with round tripping (eg. AnyHttpUrl)
            return orjson.dumps(obj).decode()

    try:
        return orjson.dumps(v, default=default_handler, option=option).decode()
    except TypeError as err:
        raise err


DEFAULT_JSON_ENCODERS = {
    pydantic.SecretStr: lambda v: v.get_secret_value() if v else None,
    pydantic.AnyHttpUrl: str,
}


class BaseModelConfig:
    """The `BaseModelConfig` class provides a common set of Pydantic model
    configuration shared across the library.
    """

    json_encoders = DEFAULT_JSON_ENCODERS
    json_loads = orjson.loads
    json_dumps = _orjson_dumps
    validate_assignment = True


class BaseModel(pydantic.BaseModel):
    """The `BaseModel` class is the base class implementation of Pydantic model
    types utilized throughout the library.
    """

    class Config(BaseModelConfig):
        validate_all = True


class License(enum.Enum):
    """The License enumeration defines a set of licenses that describe the
    terms under which software components are released for use."""

    mit = "MIT"
    apache2 = "Apache 2.0"
    proprietary = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g., "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self) -> str:
        return self.value


class Maturity(enum.Enum):
    """The Maturity enumeration defines a set of tiers that describe how mature
    and stable a software component is considered by its developers."""

    experimental = "Experimental"
    """Experimental components are in an early state of development or are
    otherwise not fully supported by the developers.

    APIs should be considered as potentially volatile and documentation, testing,
    and deployment concerns may not yet be fully addressed.
    """

    stable = "Stable"
    """Stable components can be considered production ready and released under
    Semantic Versioning expectations.

    APIs should be considered stable and the component is fully supported by
    the developers and recommended for use in a production environment.
    """

    robust = "Robust"
    """Robust components are fully mature, stable, well documented, and battle
    tested in a variety of production environments.
    """

    @classmethod
    def from_str(cls, identifier: str) -> "Maturity":
        """
        Returns a `Maturity` object for the given string identifier (e.g., "Stable").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No maturity level identified by "{identifier}".')

    def __str__(self) -> str:
        return self.value


Version = semver.VersionInfo

# NOTE: Strict values will not be type coerced by Pydantic (e.g., from "1" to 1)
Numeric = Union[pydantic.StrictFloat, pydantic.StrictInt]
NoneCallable = TypeVar("NoneCallable", bound=Callable[[None], None])

# Describing time durations in various forms is very common
DurationDescriptor = Union[datetime.timedelta, str, Numeric]


class Duration(datetime.timedelta):
    """
    Duration is a subclass of datetime.timedelta that is serialized as a Golang duration string.

    Duration objects can be initialized with a duration string, a numeric seconds value,
    a timedelta object, and with the time component keywords of timedelta.

    Refer to `servo.utilities.duration_str` for details about duration strings.
    """

    def __new__(
        cls,
        duration: Union[str, Numeric, datetime.timedelta] = 0,
        **kwargs,
    ) -> datetime.timedelta:
        seconds = kwargs.pop("seconds", 0)
        microseconds = kwargs.pop("microseconds", 0)

        if isinstance(duration, str):
            # Parse microseconds from the string
            microseconds = (
                microseconds
                + servo.utilities.duration_str.microseconds_from_duration_str(duration)
            )
        elif isinstance(duration, datetime.timedelta):
            # convert the timedelta into a microseconds float
            microseconds = microseconds + (
                duration / datetime.timedelta(microseconds=1)
            )
        elif isinstance(duration, (int, float)):
            # Numeric first arg maps to seconds on timedelta initializer
            # NOTE: We are diverging from the behavior of timedelta here
            seconds = seconds + duration

        return datetime.timedelta.__new__(
            cls, seconds=seconds, microseconds=microseconds, **kwargs
        )

    def __init__(
        self, duration: Union[str, datetime.timedelta, Numeric] = 0, **kwargs
    ) -> None:  # noqa: D107
        # Add a type signature so we don't get warning from linters. Implementation is not used (see __new__)
        ...

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: dict[Any, Any]) -> None:
        field_schema.update(
            type="string",
            format="duration",
            pattern="([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
            examples=["300ms", "5m", "2h45m", "72h3m0.5s"],
        )

    @classmethod
    def validate(cls, value) -> "Duration":
        if isinstance(value, (str, datetime.timedelta, int, float)):
            return cls(value)

        # Parse into a timedelta with Pydantic parser
        td = pydantic.datetime_parse.parse_duration(value)
        microseconds: float = td / datetime.timedelta(microseconds=1)
        return cls(microseconds=microseconds)

    @classmethod
    def since(cls, time: datetime.datetime) -> "Duration":
        """Returns a Duration object representing the elapsed time since a given start time."""
        return cls(datetime.datetime.now() - time)

    def __str__(self) -> str:
        return servo.utilities.duration_str.timedelta_to_duration_str(
            self, extended=True
        )

    def __repr__(self) -> str:
        return f"Duration('{self}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.__str__() == other
        elif isinstance(other, datetime.timedelta):
            return super().__eq__(other)
        elif isinstance(other, (int, float)):
            return self.total_seconds() == other

        return False

    def human_readable(self) -> str:
        return str(self)


class BaseProgress(abc.ABC, BaseModel):
    started_at: Optional[datetime.datetime]
    """The time that progress tracking was started."""

    def start(self) -> None:
        """Start progress tracking.

        The current time when `start` is called is used as the starting point to track progress.

        Raises:
            RuntimeError: Raised if the object has already been started.
        """
        if self.started:
            raise RuntimeError(
                "cannot start a progress object that has already been started"
            )
        self.started_at = datetime.datetime.now()

    @property
    def started(self) -> bool:
        """Return a boolean value that indicates if progress tracking has started."""
        return self.started_at is not None

    @property
    def finished(self) -> bool:
        """Return a boolean value that indicates if the progress has reached 100%."""
        return self.progress and self.progress >= 100

    async def watch(
        self,
        notify: Callable[["DurationProgress"], Union[None, Awaitable[None]]],
        every: Duration = Duration("5s"),
    ) -> None:
        """Asynchronously watch progress tracking and invoke a callback to periodically report on progress.

        Args:
            notify: An (optionally asynchronous) callable object to periodically invoke for progress reporting.
            every: The Duration to periodically invoke the notify callback to report progress.
        """

        async def async_notifier() -> None:
            if asyncio.iscoroutinefunction(notify):
                await notify(self)
            else:
                notify(self)

        if not self.started:
            self.start()

        while True:
            if self.finished:
                break

            await asyncio.sleep(every.total_seconds())
            await async_notifier()

    def every(self, duration: DurationDescriptor) -> AsyncIterator[BaseProgress]:
        """Return an async iterator yielding a progress update every duration seconds.

        Args:
            duration: The Duration on which to yield progress updates.
        """

        class _Iterator:
            def __init__(
                self, progress: servo.BaseProgress, duration: servo.Duration
            ) -> None:
                self.progress = progress
                self.duration = duration

            def __aiter__(self):  # noqa: D105
                return self

            async def __anext__(self) -> Optional[int]:
                while True:
                    if self.progress.finished:
                        raise StopAsyncIteration

                    await asyncio.sleep(self.duration.total_seconds())
                    return self.progress

        self.start()
        return _Iterator(self, servo.Duration(duration))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        ...

    def __float__(self) -> float:
        return self.progress

    def __int__(self) -> int:
        return int(self.progress)

    @property
    def elapsed(self) -> Optional[Duration]:
        """Return the total time elapsed since progress tracking was started as a Duration value."""
        return Duration.since(self.started_at) if self.started else None

    def annotate(self, str_to_annotate: str, prefix: bool = True) -> str:
        """Return a string annotated with details about progress status.

        Args:
            str_to_annotate: The string to annotate with progress status.

        Returns:
            A new string annotated with progress status info.
        """
        status = f"{self.progress:.2f}% complete, {self.elapsed} elapsed"
        if prefix:
            return f"{status} - {str_to_annotate}"
        else:
            return f"{str_to_annotate} ({status})"

    ##
    # Abstract methods

    @property
    @abc.abstractmethod
    def progress(self) -> float:
        """Return completion progress percentage as a floating point value from 0.0 to 100.0"""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset progress back to zero."""

    @abc.abstractmethod
    async def wait(self) -> None:
        """Asynchronously wait for the progress to finish."""


class DurationProgress(BaseProgress):
    """DurationProgress objects track progress across a fixed time duration."""

    duration: Duration
    """The duration of the operation for which progress is being tracked."""

    def __init__(self, duration: "Duration" = 0, **kwargs) -> None:  # noqa: D107
        super().__init__(duration=duration, **kwargs)

    @property
    def progress(self) -> float:
        """Return completion progress percentage as a floating point value from 0.0 to 100.0"""
        if self.started:
            return (
                min(100.0, 100.0 * (self.elapsed / self.duration))
                if self.duration
                else 100.0
            )
        else:
            return 0.0

    def reset(self) -> None:
        """Reset progress back to zero."""
        self.started_at = datetime.datetime.now()

    async def wait(self) -> None:
        """Asynchronously wait for the duration to elapse."""
        await asyncio.sleep(self.duration - self.elapsed)


class EventProgress(BaseProgress):
    """EventProgress objects track progress against an indeterminate event."""

    timeout: Optional[Duration] = None
    """The maximum amount of time to wait for the event to be triggered.

    When None, the event will be awaited forever.
    """

    settlement: Optional[Duration] = None
    """The amount of time to wait for progress to be reset following an event trigger before returning early.

    When None, progress is returned immediately upon the event being triggered.
    """

    _event: asyncio.Event = pydantic.PrivateAttr(default_factory=asyncio.Event)
    _settlement_timer: Optional[asyncio.TimerHandle] = pydantic.PrivateAttr(None)
    _settlement_started_at: Optional[datetime.datetime] = pydantic.PrivateAttr(None)

    def __init__(
        self,
        timeout: Optional["Duration"] = None,
        settlement: Optional["Duration"] = None,
        **kwargs,
    ) -> None:  # noqa: D107
        super().__init__(timeout=timeout, settlement=settlement, **kwargs)

    def complete(self) -> None:
        """Advance progress immediately to completion.

        This method does not respect settlement time. Typical operation should utilize the `trigger`
        method.
        """
        self._event.set()

    @property
    def completed(self) -> bool:
        """Return True if the progress has been completed."""
        return self._event.is_set()

    @property
    def timed_out(self) -> bool:
        """Return True if the timeout has elapsed.

        Return False if there is no timeout configured or the progress has not been started.
        """
        if self.timeout == 0 and self.started:
            return True
        if not self.timeout or not self.started:
            return False
        return Duration.since(self.started_at) >= self.timeout

    @property
    def finished(self) -> bool:
        return self.timed_out or super().finished

    def trigger(self) -> None:
        """Trigger the event to advance progress toward completion.

        When the event is triggered, the behavior is dependent upon whether or not a
        settlement duration is configured. When None, progress is immediately advanced to 100%
        and progress is finished, notifying all observers.

        When a settlement duration is configured, progress will begin advancing across the settlement
        duration to allow for the progress to be reset.
        """
        if self.settlement:
            self._settlement_started_at = datetime.datetime.now()
            self._settlement_timer = asyncio.get_event_loop().call_later(
                self.settlement.total_seconds(), self.complete
            )
        else:
            self.complete()

    def reset(self) -> None:
        """Reset progress to zero by clearing the event trigger.

        Resetting progress does not affect the timeout which will eventually finalize progress
        when elapsed.
        """
        if self._settlement_timer:
            self._settlement_timer.cancel()
        self._settlement_started_at = None
        self._event.clear()

    async def wait(self) -> None:
        """Asynchronously wait until the event condition has been triggered.

        If the progress was initialized with a timeout, raises a TimeoutError when the timeout is
        elapsed.

        Raises:
            TimeoutError: Raised if the timeout elapses before the event is triggered.
        """
        timeout = self.timeout.total_seconds() if self.timeout else None
        await asyncio.wait_for(self._event.wait(), timeout=timeout)

    @property
    def settling(self) -> bool:
        """Return True if the progress has been triggered but is awaiting settlement before completion."""
        return self._settlement_started_at is not None

    @property
    def settlement_remaining(self) -> Optional[Duration]:
        """Return the amount of settlement time remaining before completion."""
        if self.settling:
            duration = Duration(
                self.settlement - Duration.since(self._settlement_started_at)
            )
            return duration if duration.total_seconds() >= 0 else None
        else:
            return None

    @property
    def progress(self) -> float:
        """Return completion progress percentage as a floating point value from 0.0 to 100.0

        If the event has been triggered, immediately returns 100.0.
        When progress has started but has not yet completed, the behavior is conditional upon
        the configuration of a timeout and/or settlement time.

        When settlement is in effect, progress is relative to the amount of time remaining in the
        settlement duration. This can result in progress that goes backward as the finish moves
        forward based on the event condition being triggered.
        """
        if self._event.is_set():
            return 100.0
        elif self.started:
            if self.settling:
                return min(
                    100.0,
                    100.0
                    * (Duration.since(self._settlement_started_at) / self.settlement),
                )
            elif self.timeout:
                return min(100.0, 100.0 * (self.elapsed / self.timeout))

        # NOTE: Without a timeout or settlement duration we advance from 0 to 100. Like a true gangsta
        return 0.0

    async def watch(
        self,
        notify: Callable[["DurationProgress"], Union[None, Awaitable[None]]],
        every: Optional[Duration] = None,
    ) -> None:
        # NOTE: Handle the case where reporting interval < timeout (matters mostly for tests)
        if every is None:
            if self.timeout is None:
                every = Duration("60s")
            else:
                every = min(Duration("60s"), self.timeout)

        # return await super().watch(notify, every)
        async def async_notifier() -> None:
            if asyncio.iscoroutinefunction(notify):
                await notify(self)
            else:
                notify(self)

        if not self.started:
            self.start()

        while True:
            if self.finished:
                break

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._event.wait(), timeout=every.total_seconds()
                )
            await async_notifier()


class Unit(str, enum.Enum):
    """An enumeration of standard units of measure for metrics.

    Member names are the name of the unit and values are an abbreviation of the unit.

    ### Members:
        float: A generic floating point value.
        int: A generic integer value.
        count: An unsigned integer count of the number of times something has happened.
        rate: The frequency of an event across a time interval.
        percentage: A ratio of one value as compared to another (e.g., errors as compared to
            total requests processed).
        milliseconds: A time value at millisecond resolution.
        bytes: Digital data size in bytes.
        requests_per_minute: Application throughput in terms of requests processed per minute.
        requests_per_second: Application throughput in terms of requests processed per second.
    """

    float = ""
    int = ""
    count = ""
    rate = ""
    percentage = "%"
    milliseconds = "ms"
    bytes = "bytes"
    gibibytes = "GiB"
    cores = "cores"
    requests_per_minute = "rpm"
    requests_per_second = "rps"

    def __repr__(self) -> str:
        if self.value:
            return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"
        else:
            return f"{self.__class__.__name__}.{self.name}"


class Metric(BaseModel):
    """Metric objects model optimizeable value types in a specific Unit of measure.

    Args:
        name: The name of the metric.
        unit: The unit that the metric is measured in (e.g., requests per second).

    Returns:
        A new Metric object.
    """

    name: str
    """The name of the metric.
    """

    unit: Unit = Unit.float
    """The unit that the metric is measured in (e.g., requests per second).
    """

    def __init__(
        self, name: str, unit: Unit = Unit.float, **kwargs
    ) -> None:  # noqa: D107
        super().__init__(name=name, unit=unit, **kwargs)

    def __hash__(self):
        return hash(
            (
                self.name,
                self.unit,
            )
        )


class DataPoint(BaseModel):
    """DataPoint objects model a scalar value reading of a Metric.

    DataPoints are iterable and indexed and behave as tuple-like objects
    of the form `(time, value)`. The metric attribute is omitted from
    iteration and indexing to allow data point objects to be handled as
    programmatically interchangeable with a tuple representation.

    Args:
        metric: The metric being measured.
        time: The time that the value was read for the metric.
        value: The value that was read for the metric.

    Returns:
        A new DataPoint object modeling a scalar value reading of a Metric.
    """

    metric: Metric
    """The metric that the data point was measured from."""

    time: datetime.datetime
    """The time that the data point was measured."""

    value: float
    """The value that was measured for the metric."""

    def __init__(
        self, metric: Metric, time: datetime.datetime, value: float, **kwargs
    ) -> None:  # noqa: D107
        super().__init__(metric=metric, time=time, value=value, **kwargs)

    def __iter__(self):
        return iter((self.time, self.value))

    def __getitem__(self, index: int) -> Union[datetime.datetime, float]:
        if not isinstance(index, int):
            raise TypeError("values can only be retrieved by integer index")
        if index not in (0, 1):
            raise KeyError(f"index out of bounds: {index} not in (0, 1)")
        return operator.getitem((self.time, self.value), index)

    @property
    def unit(self) -> Unit:
        """Return the unit of the measured value."""
        return self.metric.unit

    def __str__(self) -> str:
        return f"{self.metric.name}: {self.value:.2f}{self.unit.value} @ {self.time}"

    def __repr__(self) -> str:
        abbrv = f" ({self.unit.value})" if self.unit.value else ""
        return f"DataPoint({self.metric.name}{abbrv}, ({self.time}, {self.value}))"


class NormalizationPolicy(str, enum.Enum):
    """NormalizationPolicy is an enumeration that describes how measurements
    are normalized before being reported to the optimizer.

    Members:
        passthrough: Measurements are reported as is returned by the connectors
            without applying any normalization routines.
        intersect: Measurements are reduced to a common set of interesecting
            time series data. Data points measured at times that do not have
            data points across all time series in the measurement are dropped.
        fill: Time series in the measurement are brought into alignment by
    """

    passthrough = "passthrough"
    intersect = "intersect"
    fill = "fill"


class TimeSeries(BaseModel):
    """TimeSeries objects models a sequence of data points containing
    measurements of a metric indexed in time order.

    TimeSeries objects are sized, sequenced collections of `DataPoint` objects.
    Data points are sorted on init to ensure a time indexed order.

    Attributes:
        metric: The metric that the time series was measured from.
        id: An optional identifier contextualizing the source of the time series
            among a set of peers (e.g., instance ID, IP address, etc).
        annotation: An optional human readable description about the time series.
        metadata: An optional collection of arbitrary string key-value pairs that provides
            context about the time series (e.g., the total run time of the operation, the
            server from which the readings were taken, version info about the upstream
            metrics provider, etc.).
    """

    metric: Metric = pydantic.Field(...)
    data_points: List[DataPoint] = pydantic.Field(...)
    id: Optional[str] = None
    annotation: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    def __init__(
        self, metric: Metric, data_points: List[DataPoint], **kwargs
    ) -> None:  # noqa: D107
        data_points_ = sorted(data_points, key=lambda p: p.time)
        super().__init__(metric=metric, data_points=data_points_, **kwargs)

    def __len__(self) -> int:
        return len(self.data_points)

    def __iter__(self):
        return iter(self.data_points)

    def __getitem__(self, index: int) -> servo.DataPoint:
        if not isinstance(index, int):
            raise TypeError("values can only be retrieved by integer index")
        return self.data_points[index]

    @property
    def min(self) -> Optional[DataPoint]:
        """Return the minimum data point in the series."""
        return min(self.data_points, key=operator.itemgetter(1), default=None)

    @property
    def max(self) -> Optional[DataPoint]:
        """Return the maximum data point in the series."""
        return max(self.data_points, key=operator.itemgetter(1), default=None)

    @property
    def timespan(self) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        """Return a tuple of the earliest and latest times in the series."""
        if self.data_points:
            return (self.data_points[0].time, self.data_points[-1].time)
        else:
            return None

    @property
    def duration(self) -> Optional[Duration]:
        """Return a Duration object reflecting the time span of the series."""
        if self.data_points:
            return Duration(self.data_points[-1].time - self.data_points[0].time)
        else:
            return None

    def __repr_args__(self):
        args = super().__repr_args__()
        additional = dict(
            map(lambda attr: (attr, getattr(self, attr)), ("timespan", "duration"))
        )
        return {**dict(args), **additional}.items()


Reading = Union[DataPoint, TimeSeries]
Readings = List[Reading]


@runtime_checkable
class HumanReadable(Protocol):
    """
    HumanReadable is a protocol that declares the `human_readable` method for objects
    that can be represented as a human readable string for user output.
    """

    def human_readable(**kwargs) -> str:
        """
        Return a human readable representation of the object.
        """
        ...


@runtime_checkable
class OpsaniRepr(Protocol):
    """OpsaniRepr is a protocol that declares the `__opsani_repr__` method for
    objects that can be serialized into a representation usable in Opsani API
    requests.
    """

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        """Return a representation of the object serialized for use in Opsani
        API requests.
        """
        ...


# Common output formats
YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DICT_FORMAT = "dict"
HTML_FORMAT = "html"
TEXT_FORMAT = "text"
MARKDOWN_FORMAT = "markdown"
CONFIGMAP_FORMAT = "configmap"


class AbstractOutputFormat(str, enum.Enum):
    """Defines common behaviors for command specific output format enumerations"""

    def lexer(self) -> Optional["pygments.Lexer"]:
        if self.value in [YAML_FORMAT, CONFIGMAP_FORMAT]:
            return pygments.lexers.YamlLexer()
        elif self.value == JSON_FORMAT:
            return pygments.lexers.JsonLexer()
        elif self.value == DICT_FORMAT:
            return pygments.lexers.PythonLexer()
        elif self.value == TEXT_FORMAT:
            return None
        else:
            raise RuntimeError("no lexer configured for output format {self.value}")


HTTP_METHODS = (
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "OPTIONS",
    "TRACE",
    "HEAD",
    "DELETE",
    "CONNECT",
)


class ErrorSeverity(str, enum.Enum):
    """ErrorSeverity is an enumeration the describes the severity of an error
    and establishes semantics about how it should be handled."""

    warning = "warning"
    """Warnings are advisory and do not indicate an inability to operate. By
    default, warnings will not halt execution and emit actionable messages about
    potential problems.
    """

    common = "common"
    """Common errors are atomic failures that have no bearing on the outcome of
    other operatios. By default, errors are non-blocking and other available checks
    will be executed.
    """

    critical = "critical"
    """Critical errors block the execution of dependent operations.

    Critical failures halt the execution of a sequence of checks that are part
    of a `Checks` object. For example, given a connector that connects to a
    remote service such as a metrics provider, you may wish to check that each
    metrics query is well formed and returns results. In order for any of the
    query checks to succeed, the servo must be able to connect to the service.
    During failure modes such as network partitions, service outage, or simple
    configuration errors this can result in an arbitrary number of failing
    checks with an identical root cause that make it harder to identify the
    issue. Required checks allow you to declare these sorts of pre-conditions
    and the servo will test them before running any dependent checks, ensuring
    that you get a single failure that identifies the root cause.
    """


# An `asyncio.Future` or an object that can be wrapped into an `asyncio.Future`
# via `asyncio.ensure_future()`. See `isfuturistic()`.
Futuristic = Union[asyncio.Future, Awaitable]


def isfuturistic(obj: Any) -> bool:
    """Returns True when obj is an asyncio Future or can be wrapped into one.

    Futuristic objects can be passed into `asyncio.ensure_future` and methods
    that accept awaitables such as `asyncio.gather` and `asyncio.wait_for`
    without triggering a `TypeError`.
    """
    return asyncio.isfuture(obj) or asyncio.iscoroutine(obj) or inspect.isawaitable(obj)
