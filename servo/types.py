"""The `servo.types` module defines the essential data types shared by all
consumers of the servo package.
"""
from __future__ import annotations
import asyncio

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, Union, cast, runtime_checkable

import orjson
import semver
import pydantic
from pydantic import Extra, validator, datetime_parse, root_validator
from pygments.lexers import JsonLexer, PythonLexer, YamlLexer

from servo.logging import logger
from servo.utilities import microseconds_from_duration_str, timedelta_to_duration_str


def _orjson_dumps(v, *, default, indent: Optional[int] = None, sort_keys: bool = False) -> str:
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
        try:
            if isinstance(obj, HumanReadable):
                return obj.human_readable()

            return default(obj)
        except TypeError as err:
            return orjson.dumps(obj).decode()

    try:
        return orjson.dumps(v, default=default_handler, option=option).decode()
    except TypeError as err:
        raise err

DEFAULT_JSON_ENCODERS = {}

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
        ...

class License(Enum):
    """The License enumeration defines a set of licenses that describe the
    terms under which software components are released for use."""

    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g., "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self):
        return self.value


class Maturity(Enum):
    """The Maturity enumeration defines a set of tiers that describe how mature
    and stable a software component is considered by its developers."""

    EXPERIMENTAL = "Experimental"
    """Experimental components are in an early state of development or are
otherwise not fully supported by the developers.

    APIs should be considered as potentially volatile and documentation, testing,
    and deployment concerns may not yet be fully addressed.
    """

    STABLE = "Stable"
    """Stable components can be considered production ready and released under
Semantic Versioning expectations.

    APIs should be considered stable and the component is fully supported by
    the developers and recommended for use in a production environment.
    """

    ROBUST = "Robust"
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

    def __str__(self):
        return self.value


Version = semver.VersionInfo


Numeric = Union[float, int]
NoneCallable = TypeVar("NoneCallable", bound=Callable[[None], None])


class Duration(timedelta):
    """
    Duration is a subclass of datetime.timedelta that is serialized as a Golang duration string.

    Duration objects can be initialized with a duration string, a numeric seconds value,
    a timedelta object, and with the time component keywords of timedelta.

    Refer to `servo.utilities.duration_str` for details about duration strings.
    """

    def __new__(
        cls, duration: Union[str, Numeric, timedelta] = 0, **kwargs,
    ):
        seconds = kwargs.pop("seconds", 0)
        microseconds = kwargs.pop("microseconds", 0)

        if isinstance(duration, str):
            # Parse microseconds from the string
            microseconds = microseconds + microseconds_from_duration_str(duration)
        elif isinstance(duration, timedelta):
            # convert the timedelta into a microseconds float
            microseconds = microseconds + (duration / timedelta(microseconds=1))
        elif isinstance(duration, (int, float)):
            # Numeric first arg maps to seconds on timedelta initializer
            # NOTE: We are diverging from the behavior of timedelta here
            seconds = seconds + duration

        return timedelta.__new__(
            cls, seconds=seconds, microseconds=microseconds, **kwargs
        )

    def __init__(self, duration: Union[str, timedelta, Numeric] = 0, **kwargs) -> None:
        # Add a type signature so we don't get warning from linters. Implementation is not used (see __new__)
        ...

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: dict) -> None:
        field_schema.update(
            type="string",
            format="duration",
            pattern="([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
            examples=["300ms", "5m", "2h45m", "72h3m0.5s"],
        )

    @classmethod
    def validate(cls, value) -> "Duration":
        if isinstance(value, (str, timedelta, int, float)):
            return cls(value)

        # Parse into a timedelta with Pydantic parser
        td = datetime_parse.parse_duration(value)
        microseconds: float = td / timedelta(microseconds=1)
        return cls(microseconds=microseconds)

    def __str__(self):
        return timedelta_to_duration_str(self, extended=True)

    def __repr__(self):
        return f"Duration('{self}' {super().__str__()})"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.__str__() == other
        elif isinstance(other, timedelta):
            return super().__eq__(other)
        elif isinstance(other, (int, float)):
            return self.total_seconds() == other

        return False

    def human_readable(self) -> str:
        return str(self)


class DurationProgress(BaseModel):
    """
    DurationProgress objects track progress across a fixed time duration.
    """

    duration: Duration
    """The duration of the operation for which progress is being tracked.
    """

    started_at: Optional[datetime]
    """The time that progress tracking was started.
    """

    def __init__(self, duration: "Duration", **kwargs) -> None:
        super().__init__(duration=duration, **kwargs)

    def start(self) -> None:
        """
        Starts progress tracking.

        The current time when `start` is called is used as the starting point to track progress.

        Raises:
            AssertionError: Raised if the object has already been started.
        """
        assert not self.started
        self.started_at = datetime.now()

    @property
    def started(self) -> bool:
        """
        Returns a boolean value that indicates if progress tracking has started.
        """
        return self.started_at is not None

    @property
    def finished(self) -> bool:
        """
        Returns a boolean value that indicates if the duration has elapsed and progress is 100%.
        """
        return self.progress >= 100

    async def watch(
        self,
        notify: Callable[['DurationProgress'], Union[None, Awaitable[None]]],
        every: Duration = Duration("5s"),
        ) -> None:
        """
        Asynchronously watches progress tracking and invoke a callback to periodically report on progress.

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

    @property
    def progress(self) -> float:
        """Returns completion progress percentage as a floating point value from 0.0 to
        100.0"""
        if self.started:
            return min(100.0, 100.0 * (self.elapsed / self.duration)) if self.duration else 100.0
        else:
            return 0.0

    @property
    def elapsed(self) -> Duration:
        """Returns the total time elapsed since progress tracking was started as a Duration value."""
        return Duration(datetime.now() - self.started_at) if self.started else Duration(0)

    def annotate(self, str_to_annotate: str, prefix=True) -> str:
        """
        Annotates and returns a string with details about progress status.

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


class Unit(str, Enum):
    """The Unit enumeration defines a standard set of units of measure for
optimizable metrics.
    """

    BYTES = "bytes"
    """Digital data size in bytes.
    """

    COUNT = "count"
    """An unsigned integer count of the number of times something has happened.
    """

    REQUESTS_PER_MINUTE = "rpm"
    """Application throughput in terms of requests processed per minute.
    """

    REQUESTS_PER_SECOND = "rps"
    """Application throughput in terms of requests processed per second.
    """

    PERCENTAGE = "%"
    """A ratio of one value as compared to another (e.g., errors as compared to
total requests processed).
    """

    MILLISECONDS = "ms"
    """A time value at millisecond resolution."""


class Metric(BaseModel):
    """Metric objects model optimizable value types in a specific Unit of measure.

    Args:
        name: The name of the metric.
        unit: The unit that the metric is measured in (e.g., requests per second).

    Returns:
        A new Metric object.
    """

    name: str
    """The name of the metric.
    """

    unit: Unit
    """The unit that the metric is measured in (e.g., requests per second).
    """

    def __init__(self, name: str, unit: Unit, **kwargs) -> None:
        super().__init__(name=name, unit=unit, **kwargs)

    def __hash__(self):
        return hash((self.name, self.unit,))


class DataPoint(BaseModel):
    """DataPoint objects model a scalar value reading of a Metric.

    Args:
        metric: The metric being measured.
        value: The value that was read for the metric.

    Returns:
        A new DataPoint object modeling a scalar value reading of a Metric.
    """

    metric: Metric
    """The metric being measured.
    """

    value: Numeric
    """The value that was read for the metric.
    """

    def __init__(self, metric: Metric, value: Numeric, **kwargs) -> None:
        super().__init__(metric=metric, value=value, **kwargs)

    def __str__(self) -> str:
        return f"{self.value:.2f}{self.unit.value}"


class TimeSeries(BaseModel):
    """TimeSeries objects represent a sequence of readings taken for a Metric
over a period of time.
    """

    metric: Metric
    """The metric being measured.
    """

    values: List[Tuple[datetime, Numeric]]
    """The values read for the metric at specific moments in time.
    """

    annotation: Optional[str]
    """An optional advisory annotation providing supplemental context
information about the time series.
    """

    id: Optional[str]
    """An optional identifier contextualizing the source of the time series
among a set of peers.
    """

    metadata: Optional[Dict[str, Any]]
    """An optional collection of arbitrary key-value metadata that provides
context about the time series (e.g., the total run time of the operation, the
server from which the readings were taken, version info about the upstream
metrics provider, etc.).
    """

    def __init__(self, metric: Metric, values: List[Tuple[datetime, Numeric]], **kwargs) -> None:
        super().__init__(metric=metric, values=values, **kwargs)

Reading = Union[DataPoint, TimeSeries]
Readings = List[Reading]

class SettingType(str, Enum):
    """The SettingType enumeration defines type of adjustable settings supported
by the servo.
    """

    RANGE = "range"
    """Range settings describe an inclusive span of numeric values that can be
applied to a setting.
    """

    ENUM = "enum"
    """Enum settings describe a fixed set of values that can be applied to a
setting. Enum settings are not necessarily numeric and cover use-cases such as
instance types where the applicable values are part of a fixed taxonomy.
    """

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


class Setting(BaseModel):
    """Setting objects represent adjustable parameters that have a meaningful
    impact on application performance and are tuned by the optimizer.
    """

    name: str
    """A component scoped unique name of the setting.
    """

    type: SettingType
    """The type of setting, which defines the semantics of how adjustable values
    are computed by the optimizer."""

    min: Numeric
    """The lower bound of acceptable values for a range setting.
    """

    max: Numeric
    """The upper bound of acceptable values for a range setting.
    """

    step: Numeric
    """The delta between values that the optimizer will apply to the setting.
    """

    value: Optional[Union[Numeric, str]]
    """The current value of the setting, if any.

    Connectors are responsible for hydrating the value of their settings as appropriate
    for their use-case. A value of None does not imply any particular state or failure
    mode and the value may be stale and should not be considered authoritative. Directly
    interact with connectors via events to introspect current state before driving logic
    off of setting values.
    """

    pinned: bool = False
    """When True, the value of the setting will not be changed by the optimizer.

    Canary optimization strategies will implicitly pin settings and settings can be pinned
    via configuration on the optimizer side.
    """

    def __str__(self):
        if self.type == SettingType.RANGE:
            return f"{self.name} ({self.type} {self.min}-{self.max}, {self.step})"

        return f"{self.name} ({self.type})"

    @property
    def human_readable_value(self, **kwargs) -> str:
        """
        Returns a human readable representation of the value for use in output.

        The default implementation calls the `human_readable` method on the value
        property if one exists, else coerces the value into a string. Subclasses
        can provide arbitrary implementations to directly control the representation.
        """
        if isinstance(self.value, HumanReadable):
            return cast(HumanReadable, self.value).human_readable(**kwargs)
        return str(self.value)

    def opsani_dict(self) -> dict:
        return {
            self.name: self.dict(include={"type", "min", "max", "step", "pinned", "value"})
        }


class Component(BaseModel):
    """Component objects describe optimizable applications or services that
expose adjustable settings.
    """

    name: str
    """The unique name of the component.
    """

    settings: List[Setting]
    """The list of adjustable settings that are available for optimizing the
component.
    """

    def __init__(self, name: str, settings: List[Setting], **kwargs) -> None:
        super().__init__(name=name, settings=settings, **kwargs)

    def get_setting(self, name: str) -> Optional[Setting]:
        """Returns a setting by name or None if it could not be found.

        Args:
            name: The name of the setting to retrieve.

        Returns:
            The setting within the component with the given name or None if such
            a setting could not be found.
        """
        return next(filter(lambda m: m.name == name, self.settings), None)

    def opsani_dict(self) -> dict:
        settings_dict = {"settings": {}}
        for setting in self.settings:
            settings_dict["settings"].update(setting.opsani_dict())
        return {self.name: settings_dict}


class Control(BaseModel):
    """Control objects model parameters returned by the optimizer that govern
aspects of the operation to be performed.
    """

    duration: Duration = cast(Duration, 0)
    """How long the operation should execute.
    """

    warmup: Duration = cast(Duration, 0)
    """How long to wait before starting the operation in order to allow the
application to reach a steady state (e.g., filling read through caches, loading
class files into memory, just-in-time compilation being appliied to critical
code paths, etc.).
    """

    delay: Duration = cast(Duration, 0)
    """How long to wait beyond the duration in order to ensure that the metrics
for the target interval have been aggregated and are available for reading.
    """

    load: Optional[Dict[str, Any]] = None
    """An optional dictionary describing the parameters of the desired load
profile.
    """

    userdata: Optional[Dict[str, Any]] = None
    """An optional dictionary of supplemental metadata with no platform defined
semantics.
    """

    @root_validator(pre=True)
    def validate_past_and_delay(cls, values):
        if 'past' in values:
            # NOTE: past is an alias for delay in the API
            if 'delay' in values:
                assert values['past'] == values['delay'], "past and delay attributes must be equal"

            values['delay'] = values.pop('past')

        return values

    @validator('duration', 'warmup', 'delay', always=True, pre=True)
    @classmethod
    def validate_durations(cls, value) -> Duration:
        return value or Duration(0)


class Description(BaseModel):
    """Description objects model the essential elements of a servo
configuration that the optimizer must be aware of in order to process
measurements and prescribe adjustments.
    """

    components: List[Component] = []
    """The set of adjustable components and settings that are available for
optimization.
    """

    metrics: List[Metric] = []
    """The set of measurable metrics that are available for optimization.
    """

    def get_component(self, name: str) -> Optional[Component]:
        """Returns the component with the given name or `None` if the component
could not be found.

        Args:
            name: The name of the component to retrieve.

        Returns:
            The component with the given name or None if it could not be found.
        """
        return next(filter(lambda m: m.name == name, self.components), None)

    def get_setting(self, name: str) -> Optional[Setting]:
        """
        Returns a setting from a fully qualified name of the form `component_name.setting_name`.

        Returns:
            The setting with the given name or None if it could not be found.

        Raises:
            ValueError: Raised if the name is not fully qualified.
        """
        if not "." in name:
            raise ValueError("name must include component name and setting name")
        component_name, setting_name = name.split(".", 1)
        if component := self.get_component(component_name):
            return component.get_setting(setting_name)
        return None

    def get_metric(self, name: str) -> Optional[Metric]:
        """Returns the metric with the given name or `None` if the metric
could not be found.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            The metric with the given name or None if it could not be found.
        """
        return next(filter(lambda m: m.name == name, self.metrics), None)

    def opsani_dict(self) -> dict:
        dict = {"application": {"components": {}}, "measurement": {"metrics": {}}}
        for component in self.components:
            dict["application"]["components"].update(component.opsani_dict())
        for metric in self.metrics:
            dict["measurement"]["metrics"][metric.name] = {"unit": metric.unit.value}
        return dict


class Measurement(BaseModel):
    """Measurement objects model the outcome of a measure operation and contain
a set of readings for the metrics that were measured.
    """

    readings: Readings = []
    """A list of readings taken of target metrics during the measurement
operation.

    Readings can either be `DataPoint` objects modeling scalar values or
    `TimeSeries` objects modeling a sequence of values captured over time.
    """
    annotations: Dict[str, str] = {}

    @validator('readings', always=True, pre=True)
    def validate_readings_type(cls, value) -> Readings:
        if value:
            reading_type = None
            for obj in value:
                if reading_type:
                    assert isinstance(obj, reading_type), f"all readings must be of the same type: expected \"{reading_type.__name__}\" but found \"{obj.__class__.__name__}\""
                else:
                    reading_type = obj.__class__

        return value

    @validator('readings', always=True, pre=True)
    def validate_time_series_dimensionality(cls, value) -> Readings:
        if value:
            expected_count = None
            for obj in value:
                if isinstance(obj, TimeSeries):
                    actual_count = len(obj.values)
                    if expected_count:
                        if actual_count != expected_count:
                            logger.warning(f"all TimeSeries readings must contain the same number of values: expected {expected_count} values but found {actual_count} on TimeSeries id \"{obj.id}\"")
                    else:
                        expected_count = actual_count

        return value

    def opsani_dict(self) -> dict:
        readings = {}

        for reading in self.readings:
            if isinstance(reading, TimeSeries):
                data = {
                    "unit": reading.metric.unit.value,
                    "values": [{"id": str(int(time.time())), "data": []}],
                }

                # Fill the values with arrays of [timestamp, value] sampled from the reports
                for date, value in reading.values:
                    data["values"][0]["data"].append([int(date.timestamp()), value])

                readings[reading.metric.name] = data
            else:
                raise NotImplementedError("Not done yet")

        return dict(metrics=readings, annotations=self.annotations)


class Adjustment(BaseModel):
    """Adjustment objects model an instruction from the optimizer to apply a
specific change to a setting of a component.
    """

    component_name: str
    """The name of the component to be adjusted.
    """

    setting_name: str
    """The name of the setting to be adjusted.
    """

    value: Union[str, Numeric]
    """The value to be applied to the setting being adjusted.
    """

    @property
    def selector(self) -> str:
        """Returns a fully qualified string identifier for accessing the referenced resource.
        """
        return f"{self.component_name}.{self.setting_name}"

    def __str__(self) -> str:
        return f"{self.component_name}.{self.setting_name}={self.value}"


# Common output formats
YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DICT_FORMAT = "dict"
HTML_FORMAT = "html"
TEXT_FORMAT = "text"
MARKDOWN_FORMAT = "markdown"
CONFIGMAP_FORMAT = "configmap"


class AbstractOutputFormat(str, Enum):
    """Defines common behaviors for command specific output format enumerations"""

    def lexer(self) -> Optional["pygments.Lexer"]:
        if self.value in [YAML_FORMAT, CONFIGMAP_FORMAT]:
            return YamlLexer()
        elif self.value == JSON_FORMAT:
            return JsonLexer()
        elif self.value == DICT_FORMAT:
            return PythonLexer()
        elif self.value == TEXT_FORMAT:
            return None
        else:
            raise RuntimeError("no lexer configured for output format {self.value}")

Control.update_forward_refs()


DurationType = Union[Duration, timedelta, str, bytes, int, float]


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
