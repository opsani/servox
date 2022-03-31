from datetime import datetime
import freezegun
import pydantic
import pytest
from typing import Callable, Dict, List

import servo
import servo.configuration
from servo.fast_fail import FastFailObserver
from servo.types import (
    DataPoint,
    Metric,
    Reading,
    SloCondition,
    SloInput,
    SloKeep,
    TimeSeries,
    Unit,
)


def test_non_unique_conditions() -> None:
    conditions = [
        SloCondition(metric="same", threshold=6000, slo_threshold_minimum=None),
        SloCondition(metric="same", threshold=6000, slo_threshold_minimum=None),
        SloCondition(
            metric="same2",
            threshold_metric="same3",
        ),
        SloCondition(
            metric="same2",
            threshold_metric="same3",
        ),
        SloCondition(metric="not_same", threshold=6000, slo_threshold_minimum=None),
        SloCondition(
            metric="not_same",
            keep=SloKeep.above,
            threshold=6000,
            slo_threshold_minimum=None,
        ),
    ]
    with pytest.raises(pydantic.ValidationError) as err_info:
        SloInput(conditions=conditions)

    assert str(err_info.value) == (
        "1 validation error for SloInput\n"
        "conditions\n"
        "  Slo conditions must be unique. Redundant conditions found: (same below 6000), (same2 below same3) (type=value_error)"
    )


def test_trigger_count_greater_than_window() -> None:
    with pytest.raises(pydantic.ValidationError) as err_info:
        SloCondition(
            metric="test",
            threshold=1,
            trigger_count=2,
            trigger_window=1,
            slo_threshold_minimum=None,
        )
    assert str(err_info.value) == (
        "1 validation error for SloCondition\n"
        "__root__\n"
        "  trigger_count cannot be greater than trigger_window (2 > 1) (type=value_error)"
    )


@pytest.fixture
def metric() -> Metric:
    return Metric("throughput", Unit.requests_per_minute)


@pytest.fixture
def tuning_metric() -> Metric:
    return Metric("tuning_throughput", Unit.requests_per_minute)


@pytest.fixture
def config() -> servo.configuration.FastFailConfiguration:
    return servo.configuration.FastFailConfiguration()


@pytest.fixture
def slo_input(metric: Metric, tuning_metric: Metric) -> SloInput:
    return SloInput(
        conditions=[
            SloCondition(
                metric=metric.name,
                threshold=6000,
                trigger_window=2,
                slo_threshold_minimum=None,
            ),
            SloCondition(
                metric=metric.name,
                threshold_metric=tuning_metric.name,
                trigger_window=2,
            ),
        ]
    )


@pytest.fixture
async def metrics_getter():
    async def test_metrics_getter(start, end) -> Dict[str, List[Reading]]:
        return []

    return test_metrics_getter


@pytest.fixture
def observer(config, slo_input, metrics_getter) -> Callable[[], FastFailObserver]:
    return FastFailObserver(
        config=config,
        input=slo_input,
        metrics_getter=metrics_getter,
    )


@freezegun.freeze_time("2020-01-21 12:00:01", auto_tick_seconds=600)
def _make_time_series_list(
    metric: Metric, values: List[List[float]]
) -> List[TimeSeries]:
    ret_list = []
    for index, val_list in enumerate(values):
        points = list(map(lambda v: DataPoint(metric, datetime.now(), v), val_list))
        ret_list.append(TimeSeries(metric=metric, data_points=points, id=index))
    return ret_list


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [[21337.0, 566.0, 87.0, 320.0, 59.0]],
            [[31337.0, 666.0, 187.0, 420.0, 69.0]],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [[21337.0, 566.0, 87.0, 320.0, 59.0], [31337.0, 666.0, 187.0, 420.0, 69.0]],
            [
                [31337.0, 666.0, 187.0, 420.0, 69.0],
                [31337.0, 666.0, 187.0, 420.0, 69.0],
            ],
        ),
        (
            datetime(2020, 1, 21, 12, 20, 1),
            [[21337.0, 566.0, 87.0, 320.0, 59.0]],
            [],
        ),
        (
            datetime(2020, 1, 21, 12, 30, 1),
            [],
            [[31337.0, 666.0, 187.0, 420.0, 69.0]],
        ),
        (
            datetime(2020, 1, 21, 12, 30, 1),
            [[1337.0], [566.0]],
            [[2337.0], [666.0]],
        ),
    ],
)
def test_timeseries_slos_pass(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[List[float]],
    tuning_values: List[List[float]],
) -> None:
    slo_check_readings: Dict[str, List[TimeSeries]] = {
        metric.name: _make_time_series_list(metric, values),
        tuning_metric.name: _make_time_series_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [[0.05, 0.35, 0.01, 0.03]],
            [[0.0, 0.0, 0.0, 0.0]],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [[0.05, 0.35, 0.01, 0.03], [0.05, 0.35, 0.01, 0.03]],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_timeseries_slos_skip_zero_metric(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[List[float]],
    tuning_values: List[List[float]],
) -> None:
    slo_check_readings: Dict[str, List[TimeSeries]] = {
        metric.name: _make_time_series_list(metric, values),
        tuning_metric.name: _make_time_series_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.05, 0.35, 0.01, 0.03]],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [[0.05, 0.35, 0.01, 0.03], [0.05, 0.35, 0.01, 0.03]],
        ),
    ],
)
def test_timeseries_slos_skip_zero_threshold(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[List[float]],
    tuning_values: List[List[float]],
) -> None:
    slo_check_readings: Dict[str, List[TimeSeries]] = {
        metric.name: _make_time_series_list(metric, values),
        tuning_metric.name: _make_time_series_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values, error_str",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [[31337.0, 666.0, 187.0, 420.0, 69.0]],
            [[21337.0, 566.0, 87.0, 320.0, 59.0]],
            "SLO violation(s) observed: (throughput below 6000)[2020-01-21 12:00:01 SLO failed metric value 6535.8 was"
            " not below threshold value 6000], (throughput below tuning_throughput)[2020-01-21 12:00:01 SLO failed metric"
            " value 6535.8 was not below threshold value 4473.8]",
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [
                [31337.0, 666.0, 187.0, 420.0, 69.0],
                [31337.0, 666.0, 187.0, 420.0, 69.0],
            ],
            [[21337.0, 566.0, 87.0, 320.0, 59.0], [31337.0, 666.0, 187.0, 420.0, 69.0]],
            "SLO violation(s) observed: (throughput below 6000)[2020-01-21 12:10:01 SLO failed metric value 6535.8 was"
            " not below threshold value 6000], (throughput below tuning_throughput)[2020-01-21 12:10:01 SLO failed metric"
            " value 6535.8 was not below threshold value 5504.8]",
        ),
        (
            datetime(2020, 1, 21, 12, 20, 1),
            [[31337.0, 666.0, 187.0, 420.0, 69.0]],
            [],
            "SLO violation(s) observed: (throughput below 6000)[2020-01-21 12:20:01 SLO failed metric value 6535.8 was"
            " not below threshold value 6000]",
        ),
        (
            datetime(2020, 1, 21, 12, 30, 1),
            [[31337.0], [666.0]],
            [[21337.0], [566.0]],
            "SLO violation(s) observed: (throughput below 6000)[2020-01-21 12:30:01 SLO failed metric value 16001.5 was"
            " not below threshold value 6000], (throughput below tuning_throughput)[2020-01-21 12:30:01 SLO failed metric"
            " value 16001.5 was not below threshold value 10951.5]",
        ),
    ],
)
def test_timeseries_slos_fail(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[List[float]],
    tuning_values: List[List[float]],
    error_str: str,
) -> None:
    slo_check_readings: Dict[str, List[TimeSeries]] = {
        metric.name: _make_time_series_list(metric, values),
        tuning_metric.name: _make_time_series_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    with pytest.raises(servo.EventAbortedError) as err_info:
        observer.check_readings(slo_check_readings, checked_at)

    assert str(err_info.value) == error_str


@freezegun.freeze_time("2020-01-21 12:00:01", auto_tick_seconds=600)
def _make_data_point_list(metric: Metric, values: List[float]) -> List[DataPoint]:
    return list(map(lambda v: DataPoint(metric, datetime.now(), v), values))


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [21337.0, 566.0, 87.0, 320.0, 59.0],
            [31337.0, 666.0, 187.0, 420.0, 69.0],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [2.0],
            [3.0],
        ),
    ],
)
def test_data_point_slos_pass(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[float],
    tuning_values: List[float],
) -> None:
    slo_check_readings: Dict[str, List[DataPoint]] = {
        metric.name: _make_data_point_list(metric, values),
        tuning_metric.name: _make_data_point_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [0.05, 0.35, 0.01, 0.03],
            [0.0, 0.0, 0.0, 0.0],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [0.24],
            [0.0],
        ),
    ],
)
def test_data_point_slos_skip_zero_metric(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[float],
    tuning_values: List[float],
) -> None:
    slo_check_readings: Dict[str, List[DataPoint]] = {
        metric.name: _make_data_point_list(metric, values),
        tuning_metric.name: _make_data_point_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [0.0, 0.0, 0.0, 0.0],
            [0.05, 0.35, 0.01, 0.03],
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [0.0],
            [0.24],
        ),
    ],
)
def test_data_point_slos_skip_zero_threshold(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[float],
    tuning_values: List[float],
) -> None:
    slo_check_readings: Dict[str, List[DataPoint]] = {
        metric.name: _make_data_point_list(metric, values),
        tuning_metric.name: _make_data_point_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    observer.check_readings(slo_check_readings, checked_at)


@pytest.mark.parametrize(
    "checked_at, values, tuning_values, error_str",
    [
        (
            datetime(2020, 1, 21, 12, 0, 1),
            [31337.0, 666.0, 187.0, 420.0, 69.0],
            [21337.0, 566.0, 87.0, 320.0, 59.0],
            "SLO violation(s) observed: (throughput below 6000)[2020-01-21 12:00:01 SLO failed metric value 6535.8 was"
            " not below threshold value 6000], (throughput below tuning_throughput)[2020-01-21 12:00:01 SLO failed metric"
            " value 6535.8 was not below threshold value 4473.8]",
        ),
        (
            datetime(2020, 1, 21, 12, 10, 1),
            [3.0],
            [2.0],
            "SLO violation(s) observed: (throughput below tuning_throughput)[2020-01-21 12:10:01 SLO failed metric value"
            " 3 was not below threshold value 2]",
        ),
    ],
)
def test_data_point_slos_fail(
    observer: FastFailObserver,
    checked_at: datetime,
    metric: Metric,
    tuning_metric: Metric,
    values: List[float],
    tuning_values: List[float],
    error_str: str,
) -> None:
    slo_check_readings: Dict[str, List[DataPoint]] = {
        metric.name: _make_data_point_list(metric, values),
        tuning_metric.name: _make_data_point_list(tuning_metric, tuning_values),
    }

    servo.logging.set_level("DEBUG")
    with pytest.raises(servo.EventAbortedError) as err_info:
        observer.check_readings(slo_check_readings, checked_at)

    assert str(err_info.value) == error_str
