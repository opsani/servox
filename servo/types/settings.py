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
import functools
from inspect import isclass
import pydantic
import pydantic.fields
from typing import (
    Annotated,
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_origin,
)

from .core import BaseModel, HumanReadable, Numeric, Unit


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

    @classmethod
    def get_setting_type(cls) -> Type[Any]:
        return cls.__fields__["value"].type_

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

    class Config:
        validate_all = True
        validate_assignment = True


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
        const=True,
        description="Identifies the setting as an enumeration setting.",
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
    def _validate_value_in_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        value, options = values["value"], values["values"]
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

    type: Literal["range"] = pydantic.Field(
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

    def summary(self) -> str:
        return f"{self.__class__.__name__}(range=[{self.human_readable(self.min)}..{self.human_readable(self.max)}], step={self.human_readable(self.step)}, unit={self.unit})"

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _attributes_must_be_of_same_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        range_types: dict[TypeVar, list[str]] = {}
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
    def _value_must_fall_in_range(cls, values) -> Numeric:
        value, min, max = values["value"], values["min"], values["max"]
        if value is not None and (value < min or value > max):
            raise ValueError(
                f"invalid value: {cls.human_readable(value)} is outside of the range {cls.human_readable(min)}-{cls.human_readable(max)}"
            )

        return values

    @pydantic.validator("max")
    @classmethod
    def _max_must_define_valid_range(cls, max_: Numeric, values) -> Numeric:
        if not "min" in values:
            # can't validate if we don't have a min (likely failed validation)
            return max_

        min_ = values["min"]
        if min_ > max_:
            raise ValueError(
                f"min cannot be greater than max ({cls.human_readable(min_)} > {cls.human_readable(max_)})"
            )

        return max_

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _min_and_max_must_be_step_aligned(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        name, min_, max_, step = (
            values["name"],
            values["min"],
            values["max"],
            values["step"],
        )

        if max_ is not None and min_ is not None:
            diff = max_ - min_
            if step == 0 and diff == 0:
                pass
            elif step != 0 and diff == 0:
                desc = f"{cls.__name__}({repr(name)} {cls.human_readable(min_)}-{cls.human_readable(max_)}, {cls.human_readable(step)})"
                raise ValueError(
                    f"step must be zero when min equals max: step {step} cannot step from {min_} to {max_} "
                    "(consider using the pinned attribute of settings if you have a value you don't want changed)"
                )
            elif step == 0:
                raise ValueError(f"step cannot be zero")

            if _is_step_aligned(diff, step):
                return values
            else:
                smaller_range, larger_range = _suggest_step_aligned_values(diff, step)
                desc = f"{cls.__name__}({repr(name)} {cls.human_readable(min_)}-{cls.human_readable(max_)}, {cls.human_readable(step)})"
                # try new error handling and fall back to old if bugs
                try:
                    value_type = cls.get_setting_type()
                    if get_origin(value_type) is Union:
                        value_type = str
                    cast_diff, lower_min, upper_min, lower_max, upper_max = (
                        value_type(v)
                        for v in (
                            diff,
                            max_ - smaller_range,
                            max_ - larger_range,
                            min_ + smaller_range,
                            min_ + larger_range,
                        )
                    )
                except:
                    import servo

                    servo.logger.exception(
                        f"Failed to apply new formatting to derived RangeSetting validation"
                    )
                    raise ValueError(
                        f"{desc} min/max difference is not step aligned: {diff} is not a multiple of {step} (consider "
                        f"min {max_ - smaller_range} or {max_ - larger_range}, max {min_ + smaller_range} or {min_ + larger_range})."
                    )

                raise ValueError(
                    f"{desc} min/max difference is not step aligned: {cast_diff} is not a multiple of {step} (consider "
                    f"min {lower_min} or {upper_min}, max {lower_max} "
                    f"or {upper_max})."
                )

        return values

    def __str__(self) -> str:
        return f"{self.name} ({self.type} {self.human_readable(self.min)}-{self.human_readable(self.max)}, {self.human_readable(self.step)})"

    def __opsani_repr__(self) -> dict[str, dict[Any, Any]]:
        return {
            self.name: self.dict(
                include={"type", "unit", "min", "max", "step", "pinned", "value"},
                exclude_none=True,
            )
        }


def _is_step_aligned(value: Numeric, step: Numeric) -> bool:
    return value == step or (
        decimal.Decimal(str(float(value))) % decimal.Decimal(str(float(step))) == 0
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
            return super().__init__(*args, **kwargs)
        return super().__init__(unit=Unit.cores, *args, **kwargs)

    name: str = pydantic.Field(
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

    def __init__(self, *args, **kwargs):
        if "unit" in kwargs:
            return super().__init__(*args, **kwargs)
        return super().__init__(unit=Unit.gibibytes, *args, **kwargs)

    name: str = pydantic.Field(
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

    name: str = pydantic.Field(
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

    ec2 = "ec2"


class InstanceType(EnumSetting):
    """InstanceType is a Setting that describes an adjustable enumeration of
    values for instance types of nodes or virtual machines.

    Memory is a default setting known to the Opsani optimization engine that is
    used for calculating cost impacts and carries specialized semantics and
    heuristics. Always representing memory resources as a Memory object or
    type derived thereof.
    """

    name: str = pydantic.Field(
        "inst_type",
        const=True,
        description="Identifies the setting as an instance type enum setting.",
    )
    unit: InstanceTypeUnits = pydantic.Field(
        InstanceTypeUnits.ec2,
        description="The unit of instance types identifying the provider.",
    )


def _suggest_step_aligned_values(
    value: Numeric,
    step: Numeric,
    *,
    in_repr: Optional[Callable[[Numeric], Union[str, float, int]]] = None,
) -> tuple[str, str]:
    if in_repr is None:
        # return raw data for further processing
        in_repr = lambda x: x

    # declare numeric and textual representations
    parser = functools.partial(pydantic.parse_obj_as, value.__class__)
    value_dec, step_dec = decimal.Decimal(str(float(value))), decimal.Decimal(
        str(float(step))
    )
    lower_bound, upper_bound = value_dec, value_dec
    value_repr, lower_repr, upper_repr = in_repr(parser(value_dec)), None, None

    # Find the values that are closest on either side of the value
    # Don't recommend anything smaller than step
    while value_dec < step_dec:
        value_dec += step_dec

    remainder = value_dec % step_dec

    # lower bound -- align by offseting by remainder
    lower_bound -= remainder
    assert lower_bound % step_dec == 0
    while True:
        # only decrement after first iteration
        if lower_repr is not None:
            lower_bound -= step_dec

        # if we dip below the step, anchor on it as the minimum value
        if lower_bound <= step_dec:
            lower_bound = step_dec
            lower_repr = in_repr(parser(lower_bound))
            break

        lower_repr = in_repr(parser(lower_bound))
        # if we are step aligned take the current value as lower bound
        if remainder != 0 and lower_repr == value_repr:
            continue

        # round trip the value to make sure its not a lossy representation
        repr_decimal = decimal.Decimal(str(float(parser(lower_repr))))
        if repr_decimal % step_dec == 0:
            break

    # upper bound -- start from the lower bound and find the next value
    upper_bound = lower_bound
    while True:
        upper_bound += step_dec
        upper_repr = in_repr(parser(upper_bound))
        if upper_repr == value_repr:
            continue

        # round trip the value to make sure its not a lossy representation
        repr_decimal = decimal.Decimal(str(float(parser(upper_repr))))
        if repr_decimal % step_dec == 0:
            break

    return (lower_repr, upper_repr)


class EnvironmentSetting(Setting):
    literal: Optional[str] = pydantic.Field(
        None,
        description="(Optional) The environment variable name as used in the target system (this allows name to be "
        "set to a human readable string). Defaults to configured name when literal is not configured",
    )

    @property
    def variable_name(self) -> str:
        return self.literal or self.name


class NumericType(Setting):
    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
        yield cls.validate

    @classmethod
    def validate(cls, value, field: pydantic.fields.ModelField = None):
        if isclass(value) and issubclass(value, (int, float)):
            return value

        if value == "int":
            return int
        if value == "float":
            return float

        raise ValueError(f"Unrecognized numeric type {repr(value)}")

    @classmethod
    def __modify_schema__(cls, field_schema: dict[str, Any]):
        field_schema.update(anyOf=["int", "float"])


class EnvironmentRangeSetting(RangeSetting, EnvironmentSetting):
    # # TODO promote to RangeSetting base
    value_type: NumericType = pydantic.Field(
        None, description="The optional data type of the value of the setting"
    )

    # ENV Var values are almost always represented as a str, override value parsing to accomodate
    value: Optional[Union[pydantic.StrictInt, float]] = pydantic.Field(
        None, description="The optional value of the setting as reported by the servo"
    )

    @pydantic.validator("value_type", pre=True)
    def _set_value_type_to_type(cls, value: Any) -> Union[Type[int], Type[float]]:
        if value == "int":
            return int
        if value == "float":
            return float
        return value

    @pydantic.root_validator
    def _cast_value_to_value_type(cls, values: dict[Any, Any]) -> dict[Any, Any]:
        if (value := values.get("value")) is not None and (
            value_type := values.get("value_type")
        ) is not None:
            values["value"] = value_type(value)
        return values


class EnvironmentEnumSetting(EnumSetting, EnvironmentSetting):
    pass


# https://github.com/samuelcolvin/pydantic/issues/3714
class EnvironmentSettingList(pydantic.BaseModel):
    __root__: list[
        Annotated[
            Union[EnvironmentRangeSetting, EnvironmentEnumSetting],
            pydantic.Field(discriminator="type"),
        ]
    ]

    # above https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]


# TODO: revert to this annotation when the above is resolved
# PydanticEnvironmentSettingAnnotation = Annotated[
#     Union[EnvironmentRangeSetting, EnvironmentEnumSetting],
#     pydantic.Field(discriminator="type"),
# ]

# TODO unused references to this stub in test (TestCommandConfiguration)
# class CommandConfiguration(servo.BaseConfiguration):
#     ...
