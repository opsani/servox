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

"""Types for communication with backend API"""

import pydantic
import datetime
import time
from typing import Any, Optional, Union, cast

from .core import BaseModel, DataPoint, Duration, Metric, Numeric, Readings, TimeSeries
from .settings import Setting
from .slo import SloInput


class Component(BaseModel):
    """Component objects describe optimizable applications or services that
    expose adjustable settings.
    """

    name: str
    """The unique name of the component.
    """

    settings: list[Setting]
    """The list of adjustable settings that are available for optimizing the
component.
    """

    def __init__(
        self, name: str, settings: list[Setting], **kwargs
    ) -> None:  # noqa: D107
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

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        settings_dict = {"settings": {}}
        for setting in self.settings:
            settings_dict["settings"].update(setting.__opsani_repr__())
        return {self.name: settings_dict}


class UserData(BaseModel):
    slo: Optional[SloInput] = None

    class Config(BaseModel.Config):
        # Support connector level experimentation without needing to update core servox code
        extra = pydantic.Extra.allow


class Control(BaseModel):
    """Control objects model parameters returned by the optimizer that govern
    aspects of the operation to be performed.
    """

    duration: Duration = cast(Duration, 1)
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

    load: Optional[dict[str, Any]] = None
    """An optional dictionary describing the parameters of the desired load
    profile.
    """

    userdata: Optional[UserData] = None
    """An optional dictionary of supplemental metadata with no platform defined
    semantics for most keys.
    """

    environment: Optional[dict[str, Any]] = None
    """Optional mode control.
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

    components: list[Component] = []
    """The set of adjustable components and settings that are available for
    optimization.
    """

    metrics: list[Metric] = []
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

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        dict = {"application": {"components": {}}, "measurement": {"metrics": {}}}
        for component in self.components:
            dict["application"]["components"].update(component.__opsani_repr__())

        for metric in self.metrics:
            dict["measurement"]["metrics"][metric.name] = {"unit": metric.unit.value}

        return dict


class Measurement(BaseModel):
    """Measurement objects model the outcome of a measure operation and contain
    a set of readings for the metrics that were measured.

    Measurements are sized and sequenced collections of readings.
    """

    readings: Readings = []
    """A list of readings taken of target metrics during the measurement
    operation.

    Readings can either be `DataPoint` objects modeling scalar values or
    `TimeSeries` objects modeling a sequence of values captured over time.
    """
    annotations: dict[str, str] = {}

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
                    actual_count = len(obj.data_points)
                    if expected_count and actual_count != expected_count:
                        logger.debug(
                            f'all TimeSeries readings must contain the same number of values: expected {expected_count} values but found {actual_count} on TimeSeries id "{obj.id}"'
                        )
                    else:
                        expected_count = actual_count

        return value

    def __len__(self) -> int:
        return len(self.readings)

    def __iter__(self):
        return iter(self.readings)

    def __getitem__(self, index: int) -> Union[DataPoint, TimeSeries]:
        if not isinstance(index, int):
            raise TypeError("readings can only be retrieved by integer index")
        return self.readings[index]

    def __opsani_repr__(self) -> dict[str, dict[str, Any]]:
        readings = {}

        for reading in self.readings:
            if isinstance(reading, TimeSeries):
                data = {
                    "unit": reading.metric.unit.value,
                    "values": [{"id": str(int(time.time())), "data": []}],
                }

                # Fill the values with arrays of [timestamp, value] sampled from the reports
                for date, value in reading.data_points:
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

    value: Union[
        Numeric, str
    ]  # Numeric must come first so e.g. 42:int is not coerced to '42':str
    """The value to be applied to the setting being adjusted.
    """

    @property
    def selector(self) -> str:
        """Returns a fully qualified string identifier for accessing the referenced resource."""
        return f"{self.component_name}.{self.setting_name}"

    def __str__(self) -> str:
        return f"{self.component_name}.{self.setting_name}={self.value}"


Control.update_forward_refs()
