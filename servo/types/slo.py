"""Types pertaining to Service Level Objective (SLO) adherence"""

import collections
import decimal
import enum
import pydantic
from typing import Optional

from .core import BaseModel, Numeric


class SloKeep(str, enum.Enum):
    above = "above"
    below = "below"


class SloCondition(BaseModel):
    description: Optional[str] = None
    metric: str
    slo_metric_minimum: float = 0.25
    threshold_multiplier: decimal.Decimal = decimal.Decimal(1)
    keep: SloKeep = SloKeep.below
    trigger_count: pydantic.conint(ge=1, multiple_of=1) = 1
    trigger_window: pydantic.conint(ge=1, multiple_of=1) = None
    threshold: Optional[decimal.Decimal]
    threshold_metric: Optional[str]
    slo_threshold_minimum: float = 0.25

    @pydantic.root_validator
    @classmethod
    def _check_threshold_values(cls, values):
        if (
            values.get("threshold") is not None
            and values.get("threshold_metric") is not None
        ):
            raise ValueError(
                "SLO Condition cannot specify both threshold and threshold_metric"
            )

        if values.get("threshold") is None and values.get("threshold_metric") is None:
            raise ValueError(
                "SLO Condition must specify either threshold or threshold_metric"
            )

        return values

    @pydantic.root_validator
    @classmethod
    def _check_duplicated_minimum(cls, values):
        if (
            values.get("threshold") is not None
            and values.get("slo_threshold_minimum") is not None
        ):
            raise ValueError(
                "SLO Condition cannot specify both static threshold and metric based threshold minimum"
            )

        return values

    @pydantic.validator("trigger_window", pre=True, always=True)
    @classmethod
    def _trigger_window_defaults_to_trigger_count(cls, v, *, values, **kwargs):
        if v is None:
            return values["trigger_count"]
        return v

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def _trigger_count_cannot_be_greater_than_window(cls, values) -> Numeric:
        trigger_window, trigger_count = (
            values["trigger_window"],
            values["trigger_count"],
        )
        if trigger_count > trigger_window:
            raise ValueError(
                f"trigger_count cannot be greater than trigger_window ({trigger_count} > {trigger_window})"
            )

        return values

    def __str__(self) -> str:
        ret_str = f"{self.metric} {self.keep}"

        if self.threshold:
            ret_str = f"{ret_str} {self.threshold}"
        elif self.threshold_metric:
            ret_str = f"{ret_str} {self.threshold_metric}"

        if self.description is None:
            return f"({ret_str})"
        else:
            return f"({ret_str} -> {self.description})"

    def __hash__(self) -> int:
        return hash(str(self))

    class Config(BaseModel.Config):
        extra = pydantic.Extra.forbid


class SloInput(BaseModel):
    conditions: list[SloCondition]

    @pydantic.validator("conditions")
    def _conditions_are_unique(cls, value: list[SloCondition]):
        condition_counts = collections.defaultdict(int)
        for cond in value:
            condition_counts[str(cond)] += 1

        non_unique = list(filter(lambda item: item[1] > 1, condition_counts.items()))
        if non_unique:
            raise ValueError(
                f"Slo conditions must be unique. Redundant conditions found: {', '.join(map(lambda nu: nu[0] , non_unique))}"
            )
        return value

    class Config(BaseModel.Config):
        extra = pydantic.Extra.forbid
