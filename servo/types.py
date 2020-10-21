"""The `servo.types` module defines the essential data types shared by all
consumers of the servo package.
"""
from __future__ import annotations

import abc
import asyncio
import datetime
import enum
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
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
        try:
            if isinstance(obj, HumanReadable):
                return obj.human_readable()

            return default(obj)
        except TypeError:
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


class License(enum.Enum):
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


class Maturity(enum.Enum):
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

# NOTE: Strict values will not be type coerced by Pydantic (e.g., from "1" to 1)
Numeric = Union[pydantic.StrictFloat, pydantic.StrictInt]
NoneCallable = TypeVar("NoneCallable", bound=Callable[[None], None])


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
    ):
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
    ) -> None: # noqa: D107
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

    def __str__(self):
        return servo.utilities.duration_str.timedelta_to_duration_str(
            self, extended=True
        )

    def __repr__(self):
        return f"Duration('{self}' {super().__str__()})"

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


class DurationProgress(BaseModel):
    """
    DurationProgress objects track progress across a fixed time duration.
    """

    duration: Duration
    """The duration of the operation for which progress is being tracked.
    """

    started_at: Optional[datetime.datetime]
    """The time that progress tracking was started.
    """

    def __init__(self, duration: "Duration", **kwargs) -> None: # noqa: D107
        super().__init__(duration=duration, **kwargs)

    def start(self) -> None:
        """
        Starts progress tracking.

        The current time when `start` is called is used as the starting point to track progress.

        Raises:
            AssertionError: Raised if the object has already been started.
        """
        assert not self.started
        self.started_at = datetime.datetime.now()

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
        notify: Callable[["DurationProgress"], Union[None, Awaitable[None]]],
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
            return (
                min(100.0, 100.0 * (self.elapsed / self.duration))
                if self.duration
                else 100.0
            )
        else:
            return 0.0

    @property
    def elapsed(self) -> Duration:
        """Returns the total time elapsed since progress tracking was started as a Duration value."""
        return Duration.since(self.started_at) if self.started else Duration(0)

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


class Unit(str, enum.Enum):
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

    def __init__(self, name: str, unit: Unit, **kwargs) -> None: # noqa: D107
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

    Args:
        metric: The metric being measured.
        value: The value that was read for the metric.

    Returns:
        A new DataPoint object modeling a scalar value reading of a Metric.
    """

    metric: Metric
    """The metric being measured.
    """

    value: float
    """The value that was read for the metric.
    """

    def __init__(self, metric: Metric, value: float, **kwargs) -> None: # noqa: D107
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

    values: List[Tuple[datetime.datetime, float]]
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

    def __init__(
        self, metric: Metric, values: List[Tuple[datetime.datetime, float]], **kwargs
    ) -> None: # noqa: D107
        super().__init__(metric=metric, values=values, **kwargs)


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

    def __opsani_repr__(self) -> dict:
        """Return a representation of the object serialized for use in Opsani
        API requests.
        """
        ...


class Setting(BaseModel, abc.ABC):
    """Setting is an abstract base class for models that represent adjustable
    parameters of an application under optimization.

    Concrete implementations of `RangeSetting` and `EnumSetting` are also
    provided in the `servo.types` module.

    Setting subclasses must define a `type` string identifier unique to the
    new setting and must be understandable by the optimizer backend the servo
    is collaborating with.
    """

    name: str = pydantic.Field(..., description="Name of the setting.")
    type: str = pydantic.Field(
        ..., description="Type of the setting, defining the attributes and semantics."
    )
    pinned: bool = pydantic.Field(
        False,
        description="Whether the value of the setting has been pinned, marking it as off limits for modification by the optimizer.",
    )
    value: Optional[Union[Numeric, str]] = pydantic.Field(
        None,
        description="The value of the setting as set by the servo during a measurement or set by the optimizer during an adjustment.",
    )

    @abc.abstractmethod
    def __opsani_repr__(self) -> dict:
        """Return a representation of the setting serialized for use in Opsani
        API requests.
        """
        ...

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

    def __setattr__(self, name, value) -> None:
        if name == "value":
            self._validate_pinned_values_cannot_be_changed(value)
        super().__setattr__(name, value)

    def _validate_pinned_values_cannot_be_changed(self, new_value) -> None:
        if not self.pinned or self.value is None:
            return

        if new_value != self.value:
            error = ValueError(
                f"value of pinned settings cannot be changed: assigned value {repr(new_value)} is not equal to existing value {repr(self.value)}"
            )
            error_ = pydantic.error_wrappers.ErrorWrapper(error, loc="value")
            raise pydantic.ValidationError([error_], self.__class__)

    class Config:
        validate_all = True
        validate_assignment = True


class EnumSetting(Setting):
    """EnumSetting objects describe a fixed set of values that can be applied to an
    adjustable parameter. Enum settings are not necessarily numeric and cover use-cases such as
    instance types where the applicable values are part of a fixed taxonomy.

    Validations:
        values: Cannot be an empty list.
        value:  Must be a value that appears in the `values` list.

    Raises:
        ValidationError: Raised if any field fails validation.
    """

    type = pydantic.Field(
        "enum",
        const=True,
        description="Identifies the setting as an enumeration setting.",
    )
    unit: Optional[str] = pydantic.Field(
        None,
        description="An optional unit describing the semantics or context of the values.",
    )
    values: pydantic.conlist(Union[str, Numeric], min_items=1) = pydantic.Field(
        ..., description="A list of the available options for the value of the setting."
    )
    value: Optional[Union[str, Numeric]] = pydantic.Field(
        None,
        description="The value of the setting as set by the servo during a measurement or set by the optimizer during an adjustment. When set, must a value in the `values` attribute.",
    )

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def validate_value_in_values(cls, values: dict) -> Optional[Union[str, Numeric]]:
        value, options = values["value"], values["values"]
        if value is not None and value not in options:
            raise ValueError(
                f"invalid value: {repr(value)} is not in the values list {repr(options)}"
            )

        return values

    def __opsani_repr__(self) -> dict:
        return {
            self.name: self.dict(
                include={"type", "unit", "values", "pinned", "value"}, exclude_none=True
            )
        }


class RangeSetting(Setting):
    """RangeSetting objects describe an inclusive span of numeric values that can be
    applied to an adjustable parameter.

    Validations:
        min, max, step: Each element of the range must be of the same type.
        value: Must inclusively fall within the range defined by min and max.

    A warning is emitted if the value is not aligned with the step (division
    modulus > 0).

    Raises:
        ValidationError: Raised if any field fails validation.
    """

    type = pydantic.Field(
        "range", const=True, description="Identifies the setting as a range setting."
    )
    min: Numeric = pydantic.Field(
        ...,
        description="The inclusive minimum of the adjustable range of values for the setting.",
    )
    max: Numeric = pydantic.Field(
        ...,
        description="The inclusive maximum of the adjustable range of values for the setting.",
    )
    step: Numeric = pydantic.Field(
        ...,
        description="The step value of adjustments up or down within the range. Adjustments will always be a multiplier of the step. The step defines the size of the adjustable range by constraining the available options to multipliers of the step within the range.",
    )
    value: Optional[Numeric] = pydantic.Field(
        None, description="The optional value of the setting as reported by the servo"
    )

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def range_must_be_of_same_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        range_types: Dict[TypeVar[int, float], List[str]] = {}
        for attr in ("min", "max", "step"):
            value = values[attr] if attr in values else cls.__fields__[attr].default
            attr_cls = value.__class__
            if attr_cls in range_types:
                range_types[attr_cls].append(attr)
            else:
                range_types[attr_cls] = [attr]

        if len(range_types) > 1:
            desc = ""
            for type_, fields in range_types.items():
                if len(desc):
                    desc += " "
                names = ", ".join(fields)
                desc += f"{type_.__name__}: {names}."

            raise TypeError(
                f"invalid range: min, max, and step must all be of the same Numeric type ({desc})"
            )

        return values

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def value_must_fall_in_range(cls, values) -> Numeric:
        value, min, max = values["value"], values["min"], values["max"]
        if value is not None and (value < min or value > max):
            raise ValueError(
                f"invalid value: {value} is outside of the range {min}-{max}"
            )

        return values

    @pydantic.validator("max")
    @classmethod
    def test_max_defines_valid_range(cls, value: Numeric, values) -> Numeric:
        if not "min" in values:
            # can't validate if we don't have a min (likely failed validation)
            return value

        max_ = value
        min_ = values["min"]

        if min_ == max_:
            raise ValueError(f"min and max cannot be equal ({min_} == {max_})")

        if min_ > max_:
            raise ValueError(f"min cannot be greater than max ({min_} > {max_})")

        return value

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def warn_if_value_is_not_step_aligned(cls, values: dict) -> dict:
        name, min_, max_, step, value = (
            values["name"],
            values["min"],
            values["max"],
            values["step"],
            values["value"],
        )

        if value is not None and value % step != 0:
            from servo.logging import logger

            desc = f"{cls.__name__}({repr(name)} {min_}-{max_}, {step})"
            logger.warning(
                f"{desc} value is not step aligned: {value} is not divisible by {step}"
            )

        return values

    def __str__(self) -> str:
        return f"{self.name} ({self.type} {self.min}-{self.max}, {self.step})"

    def __opsani_repr__(self) -> dict:
        return {
            self.name: self.dict(
                include={"type", "min", "max", "step", "pinned", "value"}
            )
        }


class CPU(RangeSetting):
    """CPU is a Setting that describes an adjustable range of values for CPU
    resources on a container or virtual machine.

    CPU is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing computing resources as a CPU object or
    type derived thereof.
    """

    name = pydantic.Field(
        "cpu", const=True, description="Identifies the setting as a CPU setting."
    )
    min: float = pydantic.Field(
        ..., gt=0, description="The inclusive minimum number of vCPUs or cores to run."
    )
    max: float = pydantic.Field(
        ..., description="The inclusive maximum number of vCPUs or cores to run."
    )
    step: float = pydantic.Field(
        0.125,
        description="The multiplier of vCPUs or cores to add or remove during adjustments.",
    )
    value: Optional[float] = pydantic.Field(
        None,
        description="The number of vCPUs or cores as measured by the servo or adjusted by the optimizer, if any.",
    )


class Memory(RangeSetting):
    """Memory is a Setting that describes an adjustable range of values for
    memory resources on a container or virtual machine.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name = pydantic.Field(
        "mem", const=True, description="Identifies the setting as a Memory setting."
    )

    @pydantic.validator("min")
    @classmethod
    def ensure_min_greater_than_zero(cls, value: Numeric) -> Numeric:
        if value == 0:
            raise ValueError("min must be greater than zero")

        return value


class Replicas(RangeSetting):
    """Memory is a Setting that describes an adjustable range of values for
    memory resources on a container or virtual machine.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name = pydantic.Field(
        "replicas",
        const=True,
        description="Identifies the setting as a replicas setting.",
    )
    min: pydantic.StrictInt = pydantic.Field(
        ...,
        description="The inclusive minimum number of replicas to of the application to run.",
    )
    max: pydantic.StrictInt = pydantic.Field(
        ...,
        description="The inclusive maximum number of replicas to of the application to run.",
    )
    step: pydantic.StrictInt = pydantic.Field(
        1,
        description="The multiplier of instances to add or remove during adjustments.",
    )
    value: Optional[pydantic.StrictInt] = pydantic.Field(
        None,
        description="The optional number of replicas running as measured by the servo or to be adjusted to as commanded by the optimizer.",
    )


class InstanceTypeUnits(str, enum.Enum):
    """InstanceTypeUnits is an enumeration that defines sources of compute instances."""

    EC2 = "ec2"


class InstanceType(EnumSetting):
    """InstanceType is a Setting that describes an adjustable enumeration of
    values for instance types of nodes or virtual machines.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name = pydantic.Field(
        "inst_type",
        const=True,
        description="Identifies the setting as an instance type enum setting.",
    )
    unit: InstanceTypeUnits = pydantic.Field(
        InstanceTypeUnits.EC2,
        description="The unit of instance types identifying the provider.",
    )


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

    def __init__(self, name: str, settings: List[Setting], **kwargs) -> None: # noqa: D107
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

    def __opsani_repr__(self) -> dict:
        settings_dict = {"settings": {}}
        for setting in self.settings:
            settings_dict["settings"].update(setting.__opsani_repr__())
        return {self.name: settings_dict}


class Control(BaseModel):
    """Control objects model parameters returned by the optimizer that govern
    aspects of the operation to be performed.
    """

    duration: Duration = cast(Duration, 0)
    """How long the operation should execute.
    """

    delay: Duration = cast(Duration, 0)
    """How long to wait beyond the duration in order to ensure that the metrics
    for the target interval have been aggregated and are available for reading.
    """

    warmup: Duration = cast(Duration, 0)
    """How long to wait before starting the operation in order to allow the
    application to reach a steady state (e.g., filling read through caches, loading
    class files into memory, just-in-time compilation being appliied to critical
    code paths, etc.).
    """

    settlement: Optional[Duration] = None
    """How long to wait after performing an operation in order to allow the
    application to reach a steady state (e.g., filling read through caches, loading
    class files into memory, just-in-time compilation being appliied to critical
    code paths, etc.).
    """

    load: Optional[Dict[str, Any]] = None
    """An optional dictionary describing the parameters of the desired load
    profile.
    """

    userdata: Optional[Dict[str, Any]] = None
    """An optional dictionary of supplemental metadata with no platform defined
    semantics.
    """

    @pydantic.root_validator(pre=True)
    def validate_past_and_delay(cls, values):
        if "past" in values:
            # NOTE: past is an alias for delay in the API
            if "delay" in values:
                assert (
                    values["past"] == values["delay"]
                ), "past and delay attributes must be equal"

            values["delay"] = values.pop("past")

        return values

    @pydantic.validator("duration", "warmup", "delay", always=True, pre=True)
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

    def __opsani_repr__(self) -> dict:
        dict = {"application": {"components": {}}, "measurement": {"metrics": {}}}
        for component in self.components:
            dict["application"]["components"].update(component.__opsani_repr__())

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

    @pydantic.validator("readings", always=True, pre=True)
    def validate_readings_type(cls, value) -> Readings:
        if value:
            reading_type = None
            for obj in value:
                if reading_type:
                    assert isinstance(
                        obj, reading_type
                    ), f'all readings must be of the same type: expected "{reading_type.__name__}" but found "{obj.__class__.__name__}"'
                else:
                    reading_type = obj.__class__

        return value

    @pydantic.validator("readings", always=True, pre=True)
    def validate_time_series_dimensionality(cls, value) -> Readings:
        from servo.logging import logger

        if value:
            expected_count = None
            for obj in value:
                if isinstance(obj, TimeSeries):
                    actual_count = len(obj.values)
                    if expected_count and actual_count != expected_count:
                        logger.warning(
                            f'all TimeSeries readings must contain the same number of values: expected {expected_count} values but found {actual_count} on TimeSeries id "{obj.id}"'
                        )
                    else:
                        expected_count = actual_count

        return value

    def __opsani_repr__(self) -> dict:
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
            elif isinstance(reading, DataPoint):
                data = {
                    "unit": reading.metric.unit.value,
                    "value": reading.value,
                }
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
        """Returns a fully qualified string identifier for accessing the referenced resource."""
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


Control.update_forward_refs()


DurationType = Union[Duration, datetime.timedelta, str, bytes, int, float]


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

    WARNING = "warning"
    """Warnings are advisory and do not indicate an inability to operate. By
    default, warnings will not halt execution and emit actionable messages about
    potential problems.
    """

    COMMON = "common"
    """Common errors are atomic failures that have no bearing on the outcome of
    other operatios. By default, errors are non-blocking and other available checks
    will be executed.
    """

    CRITICAL = "critical"
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
