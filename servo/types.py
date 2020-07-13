from __future__ import annotations

import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import semver
from pydantic import BaseModel
from pygments.lexers import JsonLexer, PythonLexer, YamlLexer


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


class Unit(str, Enum):
    REQUESTS_PER_MINUTE = "rpm"
    REQUESTS_PER_SECOND = "rps"
    PERCENTAGE = "%"
    MILLISECONDS = "ms"


class Metric(BaseModel):
    name: str
    unit: Unit

    def __init__(self, name: str, unit: Unit, **kwargs) -> None:
        super().__init__(name=name, unit=unit, **kwargs)


class DataPoint(BaseModel):
    metric: Metric
    value: Numeric


class TimeSeries(BaseModel):
    metric: Metric
    values: List[Tuple[datetime, Numeric]]


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

    def opsani_dict(self) -> dict:
        settings_dict = {"settings": {}}
        for setting in self.settings:
            settings_dict["settings"][setting.name] = setting.dict(
                exclude={"name"}, exclude_unset=True
            )
        return {self.name: settings_dict}


class Control(BaseModel):
    duration: Optional[int]
    past: int = 0
    warmup: int = 0
    delay: int = 0
    load: Optional[dict]


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
    readings: List[Union[DataPoint, TimeSeries]] = []
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


class CheckResult(BaseModel):
    name: str
    success: bool
    comment: Optional[str]


# Common output formats
YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DICT_FORMAT = "dict"
HTML_FORMAT = "html"
TEXT_FORMAT = "text"
MARKDOWN_FORMAT = "markdown"


class AbstractOutputFormat(str, Enum):
    """Defines common behaviors for command specific output format enumerations"""

    def lexer(self) -> Optional["pygments.Lexer"]:
        if self.value == YAML_FORMAT:
            return YamlLexer()
        elif self.value == JSON_FORMAT:
            return JsonLexer()
        elif self.value == DICT_FORMAT:
            return PythonLexer()
        elif self.value == TEXT_FORMAT:
            return None
        else:
            raise RuntimeError("no lexer configured for output format {self.value}")
