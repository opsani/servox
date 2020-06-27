from enum import Enum
from datetime import datetime, timedelta
import abc
from pydantic import BaseModel
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_type_hints, Union, Set, Tuple

Numeric = Union[int, float]

class Unit(str, Enum):
    REQUESTS_PER_MINUTE = 'rpm'
    PERCENTAGE = '%'
    MILLISECONDS = 'ms'

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
    RANGE = 'range'
    ENUM = 'enum'
# {"application": {"components": {"web": {"settings": {"cpu": {"value": 0.25, "min": 0.1, "max": 1.8, "step": 0.1, "type": "range"}, "replicas": {"value": 1, "min": 1, "max": 2, "step": 1, "type": "range"}}}}}, "measurement": {"metrics": {"requests_total": {"unit": "count"}, "throughput": {"unit": "rpm"}, "error_rate": {"unit": "percent"}, "latency_total": {"unit": "milliseconds"}, "latency_mean": {"unit": "milliseconds"}, "latency_50th": {"unit": "milliseconds"}, "latency_90th": {"unit": "milliseconds"}, "latency_95th": {"unit": "milliseconds"}, "latency_99th": {"unit": "milliseconds"}, "latency_max": {"unit": "milliseconds"}, "latency_min": {"unit": "milliseconds"}}}}
class Setting(BaseModel):
    name: str
    type: SettingType
    min: Numeric
    max: Numeric
    step: Numeric
    value: Optional[Union[Numeric, str]] # TODO: This may need a subclass to require the value for describe

class Component(BaseModel):
    name: str
    settings: List[Setting]

    def opsani_dict(self) -> dict:
        settings_dict = { 'settings': {} }
        for setting in self.settings:
            settings_dict['settings'][setting.name] = setting.dict(exclude={'name'}, exclude_unset=True)
        return { self.name: settings_dict }

class Control(BaseModel):
    duration: Optional[int]
    past: int = 0
    warmup: int = 0
    delay: int = 0
    load: Optional[dict]    

class Description(BaseModel):
    components: List[Component]
    metrics: List[Metric]

    def opsani_dict(self) -> dict:
        dict = { 'application': { 'components': {} }, 'measurement': { 'metrics': {} } }
        for component in self.components:
            dict['application']['components'].update(component.opsani_dict())
        for metric in self.metrics:
            dict['measurement']['metrics'][metric.name] = { "unit": metric.unit.value }
        return dict

class Measurement(BaseModel):
    readings: List[Union[DataPoint, TimeSeries]]
    annotations: Dict[str, str] = []
