import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from pygments.lexers import JsonLexer, PythonLexer, YamlLexer

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


# {"application": {"components": {"web": {"settings": {"cpu": {"value": 0.25, "min": 0.1, "max": 1.8, "step": 0.1, "type": "range"}, "replicas": {"value": 1, "min": 1, "max": 2, "step": 1, "type": "range"}}}}}, "measurement": {"metrics": {"requests_total": {"unit": "count"}, "throughput": {"unit": "rpm"}, "error_rate": {"unit": "percent"}, "latency_total": {"unit": "milliseconds"}, "latency_mean": {"unit": "milliseconds"}, "latency_50th": {"unit": "milliseconds"}, "latency_90th": {"unit": "milliseconds"}, "latency_95th": {"unit": "milliseconds"}, "latency_99th": {"unit": "milliseconds"}, "latency_max": {"unit": "milliseconds"}, "latency_min": {"unit": "milliseconds"}}}}
class Setting(BaseModel):
    name: str
    type: SettingType
    min: Numeric
    max: Numeric
    step: Numeric
    value: Optional[
        Union[Numeric, str]
    ]  # TODO: This may need a subclass to require the value for describe


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
