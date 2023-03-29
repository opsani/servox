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

import collections
import decimal
import enum
import datetime
import operator
import pydantic
import statistics
from typing import Awaitable, Callable, Dict, List, Optional

import devtools

import servo
import servo.errors
import servo.configuration
import servo.types

SLO_FAILED_REASON = "slo-violation"


class SloOutcomeStatus(str, enum.Enum):
    passed = "passed"
    failed = "failed"
    zero_metric = "zero_metric"
    zero_threshold = "zero_threshold"
    missing_metric = "missing_metric"
    missing_threshold = "missing_threshold"


class SloOutcome(pydantic.BaseModel):
    status: SloOutcomeStatus
    metric_readings: Optional[List[servo.types.Reading]]
    threshold_readings: Optional[List[servo.types.Reading]]
    metric_value: Optional[decimal.Decimal]
    threshold_value: Optional[decimal.Decimal]
    checked_at: datetime.datetime

    def to_message(self, condition: servo.types.SloCondition):
        if self.status == SloOutcomeStatus.missing_metric:
            message = f"Metric {condition.metric} was not found in readings"
        elif self.status == SloOutcomeStatus.missing_threshold:
            message = f"Threshold metric {condition.threshold_metric} was not found in readings"
        elif self.status == SloOutcomeStatus.passed:
            message = "SLO passed"
        elif self.status == SloOutcomeStatus.failed:
            message = f"SLO failed metric value {self.metric_value} was not {condition.keep} threshold value {self.threshold_value}"
        elif self.status == SloOutcomeStatus.zero_metric:
            message = f"Skipping SLO {condition.metric} due to near-zero metric value of {self.metric_value}"
        elif self.status == SloOutcomeStatus.zero_threshold:
            message = f"Skipping SLO {condition.metric} due to near-zero threshold value of {self.threshold_value}"
        else:
            message = f"Uncrecognized outcome status {self.status}"
        return f"{self.checked_at} {message}"


class FastFailObserver(pydantic.BaseModel):
    config: servo.configuration.FastFailConfiguration
    input: servo.types.SloInput
    metrics_getter: Callable[
        [datetime.datetime, datetime.datetime],
        Awaitable[Dict[str, List[servo.types.Reading]]],
    ]

    _results: Dict[servo.types.SloCondition, List[SloOutcome]] = pydantic.PrivateAttr(
        default=collections.defaultdict(list)
    )

    async def observe(self, progress: servo.EventProgress) -> None:
        if progress.elapsed < self.config.skip:
            return

        checked_at = datetime.datetime.now()
        metrics = await self.metrics_getter(checked_at - self.config.span, checked_at)
        self.check_readings(metrics=metrics, checked_at=checked_at)

    def check_readings(
        self,
        metrics: Dict[str, List[servo.types.Reading]],
        checked_at: datetime.datetime,
    ) -> None:
        failures: Dict[servo.types.SloCondition, List[SloOutcome]] = {}
        for condition in self.input.conditions:
            result_args = dict(checked_at=checked_at)
            # Evaluate target metric
            metric_readings = metrics.get(condition.metric)
            if not metric_readings:
                self._results[condition].append(
                    SloOutcome(**result_args, status=SloOutcomeStatus.missing_metric)
                )
                continue

            metric_value = _get_scalar_from_readings(metric_readings)
            result_args.update(
                metric_value=metric_value, metric_readings=metric_readings
            )

            if (
                self.config.treat_zero_as_missing and float(metric_value) == 0
            ) or metric_value.is_nan():
                self._results[condition].append(
                    SloOutcome(**result_args, status=SloOutcomeStatus.missing_metric)
                )
                continue

            # Evaluate threshold
            threshold_readings = None
            if condition.threshold is not None:
                threshold_value = condition.threshold * condition.threshold_multiplier

                result_args.update(
                    threshold_value=threshold_value,
                    threshold_readings=threshold_readings,
                )
            elif condition.threshold_metric is not None:
                threshold_readings = metrics.get(condition.threshold_metric)
                if not threshold_readings:
                    self._results[condition].append(
                        SloOutcome(
                            **result_args, status=SloOutcomeStatus.missing_threshold
                        )
                    )
                    continue

                threshold_scalar = _get_scalar_from_readings(threshold_readings)
                threshold_value = threshold_scalar * condition.threshold_multiplier

                result_args.update(
                    threshold_value=threshold_value,
                    threshold_readings=threshold_readings,
                )

                if (
                    self.config.treat_zero_as_missing and float(threshold_value) == 0
                ) or threshold_value.is_nan():
                    self._results[condition].append(
                        SloOutcome(
                            **result_args, status=SloOutcomeStatus.missing_threshold
                        )
                    )
                    continue

                # NOTE config.treat_zero_as_missing does not apply to these checks as it is meant to account
                #   for metrics systems in which absolute 0 is returned when no data is present for that metric
                elif 0 <= metric_value <= condition.slo_metric_minimum:
                    self._results[condition].append(
                        SloOutcome(**result_args, status=SloOutcomeStatus.zero_metric)
                    )
                    continue

                elif 0 <= threshold_value <= condition.slo_threshold_minimum:
                    self._results[condition].append(
                        SloOutcome(
                            **result_args, status=SloOutcomeStatus.zero_threshold
                        )
                    )
                    continue

            # Check target against threshold
            check_passed_op = _get_keep_operator(condition.keep)
            if check_passed_op(metric_value, threshold_value):
                self._results[condition].append(
                    SloOutcome(**result_args, status=SloOutcomeStatus.passed)
                )
            else:
                self._results[condition].append(
                    SloOutcome(**result_args, status=SloOutcomeStatus.failed)
                )

            # Update window by slicing last n items from list where n is trigger_window
            self._results[condition] = self._results[condition][
                -condition.trigger_window :
            ]

            if (
                len(
                    list(
                        filter(
                            lambda res: res.status == SloOutcomeStatus.failed,
                            self._results[condition],
                        )
                    )
                )
                >= condition.trigger_count
            ):
                failures[condition] = self._results[condition]

        servo.logger.debug(f"SLO results: {devtools.pformat(self._results)}")

        # Log the latest results
        last_results_buckets: Dict[
            SloOutcomeStatus, List[str]
        ] = collections.defaultdict(list)
        for condition, results_list in self._results.items():
            last_result = results_list[-1]
            last_results_buckets[last_result.status].append(str(condition))

        last_results_messages: List[str] = []
        for status, condition_str_list in last_results_buckets.items():
            last_results_messages.append(
                f"x{len(condition_str_list)} {status} [{', '.join(condition_str_list)}]"
            )

        servo.logger.info(
            f"SLO statuses from last check: {', '.join(last_results_messages)}"
        )

        if failures:
            raise servo.errors.EventAbortedError(
                f"SLO violation(s) observed: {_get_results_str(failures)}",
                reason=SLO_FAILED_REASON,
            )


# Helper methods
def _get_keep_operator(keep: servo.types.SloKeep):
    if keep == servo.types.SloKeep.below:
        return operator.le
    elif keep == servo.types.SloKeep.above:
        return operator.ge
    else:
        raise ValueError(f"Unknown SloKeep type {keep}")


def _get_scalar_from_readings(
    metric_readings: List[servo.types.Reading],
) -> decimal.Decimal:
    instance_values = []
    for reading in metric_readings:
        # TODO: NewRelic APM returns 0 for missing metrics. Will need optional config to ignore 0 values
        #   when implementing fast fail for eventual servox newrelic connector
        if isinstance(reading, servo.types.DataPoint):
            instance_values.append(decimal.Decimal(reading.value))
        elif isinstance(reading, servo.types.TimeSeries):
            timeseries_values = list(
                map(lambda dp: decimal.Decimal(dp.value), reading.data_points)
            )
            if len(timeseries_values) > 1:
                instance_values.append(statistics.mean(timeseries_values))
            else:
                instance_values.append(timeseries_values[0])
        else:
            raise ValueError(f"Unknown metric reading type {type(reading)}")

    if len(instance_values) > 1:
        return statistics.mean(instance_values)
    else:
        return decimal.Decimal(instance_values[0])


def _get_results_str(results: Dict[servo.types.SloCondition, List[SloOutcome]]) -> str:
    fmt_outcomes = []
    for condition, outcome_list in results.items():
        outcome_str_list = list(
            map(lambda outcome: outcome.to_message(condition), outcome_list)
        )
        fmt_outcomes.append(f"{condition}[{', '.join(outcome_str_list)}]")
    return ", ".join(fmt_outcomes)
