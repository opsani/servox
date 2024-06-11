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

import abc
import decimal
import enum
from inspect import isclass
import typing
import pydantic
import pydantic_core
import pydantic.fields
from typing import (
    List,
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_origin,
)

from .core import BaseModel, HumanReadable, Numeric, Unit
from pydantic import Field, ConfigDict
from typing_extensions import Annotated


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
    unit: Optional[str] = pydantic.Field(
        None,
        description="An optional unit describing the semantics or context of the value.",
    )
    value: Optional[Union[Numeric, str]] = pydantic.Field(
        None,
        description="The value of the setting as set by the servo during a measurement or set by the optimizer during an adjustment.",
    )

    def safe_set_value_copy(self, value: Any) -> "Setting":
        """Returns a copy of itself with the value updated, bypassing any failing validation checks.
        Ideal for readonly operations (eg. describe) where values are already in place and shouldn't be subject to validation
        """
        try:
            new_setting = self.copy()
            new_setting.value = value
            return new_setting
        except ValueError:
            import servo

            servo.logger.exception(f"Failed to parse safe_set value {repr(value)}")
            if (vt := getattr(self, "value_type", None)) is not None:
                value = vt(value)
            return self.copy(update={"value": value})

    def summary(self) -> str:
        return repr(self)

    @abc.abstractmethod
    def __opsani_repr__(self) -> dict[str, Any]:
        """Return a representation of the setting serialized for use in Opsani
        API requests.
        """
        ...

    @property
    def human_readable_value(self) -> str:
        """
        Returns a human readable representation of the value for use in output.

        The default implementation calls the `human_readable` method on the value
        property if one exists, else coerces the value into a string. Subclasses
        can provide arbitrary implementations to directly control the representation.
        """
        if getattr(self.value, "human_readable", None):
            return cast(HumanReadable, self.value).human_readable()
        return str(self.value)

    def __setattr__(self, name, value) -> None:
        if (
            name == "value"
            and self.pinned
            and self.value is not None
            and value != self.value
        ):
            raise ValueError(
                f"value of pinned settings cannot be changed: assigned value {repr(value)} is not equal to existing value {repr(self.value)}"
            )

        super().__setattr__(name, value)

    @classmethod
    def get_setting_type(cls, unwrap_union=False) -> Type[Any]:
        value_type = cls.model_fields["value"].annotation
        if unwrap_union and get_origin(value_type) is Union:
            none_filtered = [
                a for a in typing.get_args(value_type) if a is not type(None)
            ]
            if len(none_filtered) > 0:
                value_type = none_filtered[0]
            else:
                import servo

                servo.logger.warning(
                    f"unable to determine inner type for Union field value of model {cls}"
                )
                value_type = str

        return value_type

    @classmethod
    def human_readable(cls, value: Any) -> str:
        try:
            output_type = cls.get_setting_type()
            casted_value = output_type(value)
            if isinstance(casted_value, HumanReadable):
                return cast(HumanReadable, casted_value).human_readable()
        except:
            pass

        return str(value)

    model_config = ConfigDict(validate_default=True, validate_assignment=True)


# Helper methods for working with lists of settings
def find_setting(settings: list[Setting], setting_name: str) -> Optional[Setting]:
    return next(iter(s for s in settings if s.name == setting_name), None)


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

    type: Literal["enum"] = pydantic.Field(
        "enum",
        description="Identifies the setting as an enumeration setting.",
    )
    values: Annotated[List[Union[str, Numeric]], Field(min_length=1)] = pydantic.Field(
        ..., description="A list of the available options for the value of the setting."
    )
    value: Optional[Union[str, Numeric]] = pydantic.Field(
        None,
        description="The value of the setting as set by the servo during a measurement or set by the optimizer during an adjustment. When set, must a value in the `values` attribute.",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _validate_value_in_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        value, options = values.get("value", None), values["values"]
        if value is not None and value not in options:
            raise ValueError(
                f"invalid value: {repr(value)} is not in the values list {repr(options)}"
            )

        return values

    def summary(self) -> str:
        return (
            f"{self.__class__.__name__}(values={repr(self.values)}, unit={self.unit})"
        )

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        return {
            self.name: self.model_dump(
                include={"type", "unit", "values", "pinned", "value"}, exclude_none=True
            )
        }


# TODO implement generics if possible
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

    type: Literal["range"] = pydantic.Field(
        "range", description="Identifies the setting as a range setting."
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

    def summary(self) -> str:
        return f"{self.__class__.__name__}(range=[{self.human_readable(self.min)}..{self.human_readable(self.max)}], step={self.human_readable(self.step)}, unit={self.unit})"

    @pydantic.model_validator(mode="after")
    def _attributes_must_be_of_same_type(self) -> "RangeSetting":
        range_types: dict[TypeVar, list[str]] = {}
        for attr in ("min", "max", "step"):
            value = getattr(self, attr, None)
            if value is None:
                value = self.__class__.model_fields[attr].default
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

            import servo

            servo.logger.warning(
                f"invalid range: min, max, and step must all be of the same Numeric type ({desc})"
            )

        return self

    @pydantic.model_validator(mode="after")
    def _value_must_fall_in_range(self) -> "RangeSetting":
        cls = self.__class__
        if self.value is not None and (self.value < self.min or self.value > self.max):
            import servo

            servo.logger.warning(
                f"invalid value: {cls.human_readable(self.value)} is outside of the range {cls.human_readable(self.min)}-{cls.human_readable(self.max)}"
            )

        return self

    @pydantic.model_validator(mode="after")
    def _max_must_define_valid_range(self) -> "RangeSetting":
        cls = self.__class__
        if self.min > self.max:
            import servo

            servo.logger.warning(
                f"min cannot be greater than max ({cls.human_readable(self.min)} > {cls.human_readable(self.max)})"
            )

        return self

    def _suggest_step_aligned_values(self) -> tuple[Numeric, Numeric]:
        # FIXME
        range_size, step_decimal = decimal.Decimal(
            self.max - self.min
        ), decimal.Decimal(self.step)
        lower_bound, upper_bound = range_size, range_size

        # Find the values that are closest on either side of the value
        # Ensure the smaller size isn't smaller than step
        remainder = range_size % step_decimal

        # lower bound -- align by offseting by remainder
        lower_bound -= remainder
        assert lower_bound % step_decimal == 0
        if lower_bound <= step_decimal:
            lower_bound = step_decimal

        # upper bound -- start from the lower bound and find the next value
        upper_bound = lower_bound + step_decimal
        if upper_bound == range_size:
            upper_bound += step_decimal

        value_type = self.get_setting_type(unwrap_union=True)
        return (value_type(lower_bound), value_type(upper_bound))

    @pydantic.model_validator(mode="after")
    @classmethod
    def _min_and_max_must_be_step_aligned(cls, value: "RangeSetting") -> dict[str, Any]:
        name, min_, max_, step = (
            value.name,
            value.min,
            value.max,
            value.step,
        )

        if max_ is not None and min_ is not None:
            value_type = cls.get_setting_type(unwrap_union=True)
            diff = value_type(max_ - min_)
            if step == 0 and diff == 0:
                pass
            elif step != 0 and diff == 0:
                desc = f"{cls.__name__}({repr(name)} {cls.human_readable(min_)}-{cls.human_readable(max_)}, {cls.human_readable(step)})"
                import servo

                servo.logger.warning(
                    f"step must be zero when min equals max: step {step} cannot step from {min_} to {max_} "
                    "(consider using the pinned attribute of settings if you have a value you don't want changed)"
                )
            elif step == 0:
                import servo

                servo.logger.warning(f"step cannot be zero")

            if _is_step_aligned(diff, step):
                return value
            else:
                import servo

                smaller_range, larger_range = value._suggest_step_aligned_values()
                desc = f"{cls.__name__}({repr(name)} {cls.human_readable(min_)}-{cls.human_readable(max_)}, {cls.human_readable(step)})"
                # try new error handling and fall back to old if bugs
                try:
                    lower_min, upper_min, lower_max, upper_max = (
                        value_type(max_ - smaller_range),
                        value_type(max_ - larger_range),
                        value_type(min_ + smaller_range),
                        value_type(min_ + larger_range),
                    )
                except:
                    servo.logger.exception(
                        f"Failed to apply new formatting to derived RangeSetting validation"
                    )
                    servo.logger.warning(
                        f"{desc} min/max difference is not step aligned: {diff} is not a multiple of {step} (consider "
                        f"min {max_ - smaller_range} or {max_ - larger_range}, max {min_ + smaller_range} or {min_ + larger_range})."
                    )
                else:
                    servo.logger.warning(
                        f"{desc} min/max difference is not step aligned: {diff} is not a multiple of {step} (consider "
                        f"min {lower_min} or {upper_min}, max {lower_max} "
                        f"or {upper_max})."
                    )

        return value

    def __str__(self) -> str:
        return f"{self.name} ({self.type} {self.human_readable(self.min)}-{self.human_readable(self.max)}, {self.human_readable(self.step)})"

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        return {
            self.name: self.model_dump(
                include={"type", "unit", "min", "max", "step", "pinned", "value"},
                exclude_none=True,
            )
        }


def _is_step_aligned(value: Numeric, step: Numeric) -> bool:
    return (
        step == 0
        or value == step
        or (decimal.Decimal(str(float(value))) % decimal.Decimal(str(float(step))) == 0)
    )


class CPU(RangeSetting):
    """CPU is a Setting that describes an adjustable range of values for CPU
    resources on a container or virtual machine.

    CPU is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing computing resources as a CPU object or
    type derived thereof.
    """

    def __init__(self, *args, **kwargs):
        if "unit" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(unit=Unit.cores, *args, **kwargs)

    name: Literal["cpu"] = pydantic.Field(
        "cpu", description="Identifies the setting as a CPU setting."
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

    def __init__(self, *args, **kwargs):
        if "unit" in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(unit=Unit.gibibytes, *args, **kwargs)

    name: Literal["mem"] = pydantic.Field(
        "mem", description="Identifies the setting as a Memory setting."
    )

    @pydantic.field_validator("min")
    @classmethod
    def ensure_min_greater_than_zero(cls, value: Numeric) -> Numeric:
        if value == 0:
            import servo

            servo.logger.warning("min must be greater than zero")

        return value


class Replicas(RangeSetting):
    """Memory is a Setting that describes an adjustable range of values for
    memory resources on a container or virtual machine.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name: Literal["replicas"] = pydantic.Field(
        "replicas",
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


class InstanceTypeUnits(enum.StrEnum):
    """InstanceTypeUnits is an enumeration that defines sources of compute instances."""

    ec2 = "ec2"


class InstanceType(EnumSetting):
    """InstanceType is a Setting that describes an adjustable enumeration of
    values for instance types of nodes or virtual machines.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name: Literal["inst_type"] = pydantic.Field(
        "inst_type",
        description="Identifies the setting as an instance type enum setting.",
    )
    unit: InstanceTypeUnits = pydantic.Field(
        InstanceTypeUnits.ec2,
        description="The unit of instance types identifying the provider.",
    )


class EnvironmentSetting(Setting):
    literal: Optional[str] = pydantic.Field(
        None,
        description="(Optional) The environment variable name as used in the target system (this allows name to be "
        "set to a human readable string). Defaults to configured name when literal is not configured",
    )

    @property
    def variable_name(self) -> str:
        return self.literal or self.name


class EnvironmentRangeSetting(RangeSetting, EnvironmentSetting):
    # # TODO promote to RangeSetting base
    value_type: Union[Literal["int"], Literal["float"], None] = pydantic.Field(
        None, description="The optional data type of the value of the setting"
    )

    # ENV Var values are almost always represented as a str, override value parsing to accomodate
    value: Optional[Union[pydantic.StrictInt, float]] = pydantic.Field(
        None, description="The optional value of the setting as reported by the servo"
    )

    @pydantic.model_validator(mode="before")
    def _cast_value_to_value_type(cls, values: dict[Any, Any]) -> dict[Any, Any]:
        if (value := values.get("value")) is not None and (
            value_type := values.get("value_type")
        ) is not None:
            if value_type == "int":
                values["value"] = int(value)

            if value_type == "float":
                values["value"] = float(value)

        return values


class EnvironmentEnumSetting(EnumSetting, EnvironmentSetting):
    pass


# https://github.com/samuelcolvin/pydantic/issues/3714
class EnvironmentSettingList(pydantic.RootModel):
    root: list[
        Annotated[
            Union[EnvironmentRangeSetting, EnvironmentEnumSetting],
            pydantic.Field(discriminator="type"),
        ]
    ]

    # above https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


# TODO: revert to this annotation when the above is resolved
# PydanticEnvironmentSettingAnnotation = Annotated[
#     Union[EnvironmentRangeSetting, EnvironmentEnumSetting],
#     pydantic.Field(discriminator="type"),
# ]

# TODO unused references to this stub in test (TestCommandConfiguration)
# class CommandConfiguration(servo.BaseConfiguration):
#     ...
