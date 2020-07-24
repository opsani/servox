from __future__ import annotations

import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import semver
from pydantic import BaseModel, validator, datetime_parse
from pygments.lexers import JsonLexer, PythonLexer, YamlLexer

from servo.utilities import microseconds_from_duration_str, timedelta_to_duration_str


class License(Enum):
    """Defined licenses"""

    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self):
        return self.value


class Maturity(Enum):
    """Connector maturity level"""

    EXPERIMENTAL = "Experimental"
    STABLE = "Stable"
    ROBUST = "Robust"

    @classmethod
    def from_str(cls, identifier: str) -> "Maturity":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
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
    a timedelta object, and with the time component keywoards of timedelta.

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

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: dict) -> None:
        field_schema.update(
            type="string",
            format="duration",
            pattern="([\d\.]+h)?([\d\.]+m)?([\d\.]+s)?([\d\.]+ms)?([\d\.]+us)?([\d\.]+ns)?",
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
        return timedelta_to_duration_str(self)

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


class DurationProgress(BaseModel):
    duration: Duration
    started_at: Optional[datetime]

    def __init__(self, duration: "Duration", **kwargs) -> None:
        super().__init__(duration=duration, **kwargs)

    def start(self) -> None:
        assert not self.is_started()
        self.started_at = datetime.now()

    def is_started(self) -> bool:
        return self.started_at is not None

    def is_completed(self) -> bool:
        return self.progress() >= 100

    @property
    def progress(self) -> float:
        return min(100.0, 100.0 * (self.elapsed / self.duration))

    @property
    def elapsed(self) -> Duration:
        return Duration(datetime.now() - self.started_at)

    def annotate(self, str_to_annotate: str) -> str:
        return f"{self.progress:.2f}% complete, {self.elapsed} elapsed - {str_to_annotate}"


class Unit(str, Enum):
    BYTES = "bytes"
    REQUESTS_PER_MINUTE = "rpm"
    REQUESTS_PER_SECOND = "rps"
    PERCENTAGE = "%"
    MILLISECONDS = "ms"


class Metric(BaseModel):
    name: str
    unit: Unit

    def __init__(self, name: str, unit: Unit, **kwargs) -> None:
        super().__init__(name=name, unit=unit, **kwargs)
    
    def __hash__(self):
        return hash((self.name, self.unit,))


class DataPoint(BaseModel):
    metric: Metric
    value: Numeric

    def __init__(self, metric: Metric, value: Numeric, **kwargs) -> None:
        super().__init__(metric=metric, value=value, **kwargs)
    
    def __str__(self) -> str:
        return f"{self.value:.2f}{self.unit.value}"


class TimeSeries(BaseModel):
    metric: Metric
    values: List[Tuple[datetime, Numeric]]
    annotation: Optional[str]
    id: Optional[str] # TODO: source, id, context
    metadata: Optional[Dict[str, str]]

    def __init__(self, metric: Metric, values: List[Tuple[datetime, Numeric]], **kwargs) -> None:
        super().__init__(metric=metric, values=values, **kwargs)

Reading = Union[DataPoint, TimeSeries]
Readings = List[Reading]

class SettingType(str, Enum):
    RANGE = "range"
    ENUM = "enum"


class Setting(BaseModel):
    name: str
    type: SettingType
    min: Numeric
    max: Numeric
    step: Numeric
    value: Optional[Union[Numeric, str]]

    def __str__(self):
        if self.type == SettingType.RANGE:
            return f"{self.name} ({self.type} {self.min}-{self.max}, {self.step})"

        return f"{self.name} ({self.type})"


class Component(BaseModel):
    name: str
    settings: List[Setting]

    def __init__(self, name: str, settings: List[Setting], **kwargs) -> None:
        super().__init__(name=name, settings=settings, **kwargs)

    def opsani_dict(self) -> dict:
        settings_dict = {"settings": {}}
        for setting in self.settings:
            settings_dict["settings"][setting.name] = setting.dict(
                exclude={"name"}, exclude_unset=True
            )
        return {self.name: settings_dict}


class Control(BaseModel):
    duration: Optional[Duration]
    past: Duration = None
    warmup: Duration = None
    delay: Duration = None
    load: Optional[dict]

    @validator('past', 'warmup', 'delay', always=True, pre=True)
    @classmethod
    def validate_durations(cls, value) -> Duration:
        if value:
            return value
        return Duration(0)


class Description(BaseModel):
    components: List[Component] = []
    metrics: List[Metric] = []

    def opsani_dict(self) -> dict:
        dict = {"application": {"components": {}}, "measurement": {"metrics": {}}}
        for component in self.components:
            dict["application"]["components"].update(component.opsani_dict())
        for metric in self.metrics:
            dict["measurement"]["metrics"][metric.name] = {"unit": metric.unit.value}
        return dict


class Measurement(BaseModel):
    readings: Readings = []
    annotations: Dict[str, str] = {}

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


class Check(BaseModel):
    name: str
    description: Optional[str]
    success: bool
    comment: Optional[str]
    created_at: datetime = None

    @validator("created_at", pre=True, always=True)
    @classmethod
    def set_created_at_now(cls, v):
        return v or datetime.now()


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
