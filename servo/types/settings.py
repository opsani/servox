import abc
import decimal
import enum
import functools
import pydantic
from typing import Any, Callable, Generator, Optional, TypeVar, Union, cast

from .core import BaseModel, HumanReadable, Numeric

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
        Ideal for readonly operations (eg. describe) where values are already in place and shouldn't be subject to validation"""
        try:
            new_setting = self.copy()
            new_setting.value = value
            return new_setting
        except ValueError as ve:
            import servo
            servo.logger.error(f"Failed to parse safe_set value {value}")
            return self.copy(update={"value": value})

    def summary(self) -> str:
        return repr(self)

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

    @classmethod
    def human_readable(cls, value: Any) -> str:
        try:
            output_type = cls.__fields__["value"].type_
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

    type = pydantic.Field(
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
    def _validate_value_in_values(cls, values: dict) -> dict[str, Any]:
        value, options = values["value"], values["values"]
        if value is not None and value not in options:
            raise ValueError(
                f"invalid value: {repr(value)} is not in the values list {repr(options)}"
            )

        return values

    def summary(self) -> str:
        return f"{self.__class__.__name__}(values={repr(self.values)}, unit={self.unit})"

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

    def summary(self) -> str:
        return f"{self.__class__.__name__}(range=[{self.human_readable(self.min)}..{self.human_readable(self.max)}], step={self.human_readable(self.step)}, unit={self.unit})"

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _attributes_must_be_of_same_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        range_types: dict[TypeVar[int, float], list[str]] = {}
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

    @pydantic.validator("step")
    @classmethod
    def _step_cannot_be_zero(cls, value: Numeric) -> Numeric:
        if not value:
            raise ValueError(f"step cannot be zero")

        return value

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _min_cannot_be_less_than_step(cls, values: dict) -> dict[str, Any]:
        min, step = values["min"], values["step"]
        # NOTE: some resources can scale to zero (e.g., Replicas)
        if min != 0 and min < step:
            raise ValueError(f'min cannot be less than step ({min} < {step})')

        return values

    @pydantic.validator("max")
    @classmethod
    def _max_must_define_valid_range(cls, max_: Numeric, values) -> Numeric:
        if not "min" in values:
            # can't validate if we don't have a min (likely failed validation)
            return max_

        min_ = values["min"]
        if min_ > max_:
            raise ValueError(f"min cannot be greater than max ({cls.human_readable(min_)} > {cls.human_readable(max_)})")

        return max_

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _min_and_max_must_be_step_aligned(cls, values: dict[str, Any]) -> dict[str, Any]:
        name, min_, max_, step = (
            values["name"],
            values["min"],
            values["max"],
            values["step"],
        )

        for boundary in ('min', 'max'):
            value = values[boundary]
            if value and not _is_step_aligned(value, step):
                suggested_lower, suggested_upper = _suggest_step_aligned_values(value, step, in_repr=cls.human_readable)
                desc = f"{cls.__name__}({repr(name)} {cls.human_readable(min_)}-{cls.human_readable(max_)}, {cls.human_readable(step)})"
                raise ValueError(
                    f"{desc} {boundary} is not step aligned: {cls.human_readable(value)} is not a multiple of {cls.human_readable(step)} (consider {suggested_lower} or {suggested_upper})."
                )

        return values

    def __str__(self) -> str:
        return f"{self.name} ({self.type} {self.human_readable(self.min)}-{self.human_readable(self.max)}, {self.human_readable(self.step)})"

    def __opsani_repr__(self) -> dict:
        return {
            self.name: self.dict(
                include={"type", "unit", "min", "max", "step", "pinned", "value"}
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

    ec2 = "ec2"


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
        InstanceTypeUnits.ec2,
        description="The unit of instance types identifying the provider.",
    )

def _is_step_aligned(value: Numeric, step: Numeric) -> bool:
    if value == step:
        return True
    elif value > step:
        return decimal.Decimal(str(float(value))) % decimal.Decimal(str(float(step))) == 0
    else:
        return decimal.Decimal(str(float(step))) % decimal.Decimal(str(float(value))) == 0

def _suggest_step_aligned_values(value: Numeric, step: Numeric, *, in_repr: Optional[Callable[[Numeric], str]] = None) -> tuple[str, str]:
    if in_repr is None:
        # return string identity by default
        in_repr = lambda x: str(x)

    # declare numeric and textual representations
    parser = functools.partial(pydantic.parse_obj_as, value.__class__)
    value_dec, step_dec = decimal.Decimal(str(float(value))), decimal.Decimal(str(float(step)))
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

class EnvironmentSetting(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def value(self) -> Optional[Numeric]:
        ...

    @classmethod
    def __get_validators__(cls: "EnvironmentSetting") -> Generator[Callable[..., Any], None, None]:
        yield cls.validate

    @classmethod
    def validate(cls: "EnvironmentSetting", value: Any) -> Union["EnvironmentEnumSetting", "EnvironmentRangeSetting"]:
        if isinstance(value, dict):
            _type = value.get('type')
            if _type == 'range':
                return EnvironmentRangeSetting(value)
            elif _type == 'enum':
                return EnvironmentEnumSetting(value)
            else:
                raise ValueError(f'Unknown type for environment variable settings {_type}')
        else:
            raise ValueError(f'Unable to parse Environment setting, cannot get type from {type(value)} value "{value}"')


    literal: Optional[str] = pydantic.Field(
        None, description="(Optional) The environment variable name as used in the target system (this allows name to be "
                         "set to a human readable string). Defaults to configured name when literal is not configured"
    )

    @property
    def variable_name(self) -> str:
        return self.literal or self.name

class EnvironmentRangeSetting(RangeSetting, EnvironmentSetting):
    pass

class EnvironmentEnumSetting(EnumSetting, EnvironmentSetting):
    pass

# TODO unused references to this stub in test (TestCommandConfiguration)
# class CommandConfiguration(servo.BaseConfiguration):
#     ...
