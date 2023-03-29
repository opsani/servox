import datetime
import pytest

from servo.types.core import DataPoint, TimeSeries, Unit
from servo.types.api import *


def test_adjustment_str() -> None:
    adjustment = Adjustment(component_name="web", setting_name="cpu", value=1.25)
    assert adjustment.__str__() == "web.cpu=1.25"
    assert str(adjustment) == "web.cpu=1.25"
    assert f"adjustment=({adjustment})" == "adjustment=(web.cpu=1.25)"
    assert (
        f"[adjustments=({', '.join(list(map(str, [adjustment])))})]"
        == "[adjustments=(web.cpu=1.25)]"
    )


class TestMeasurement:
    @pytest.fixture
    def metric(self) -> Metric:
        return Metric("throughput", Unit.requests_per_minute)

    def test_rejects_empty_data_point(self, metric: Metric) -> None:
        with pytest.raises(ValueError) as e:
            readings = [DataPoint(metric, datetime.datetime.now(), None)]
            Measurement(readings=readings)
        assert e
        assert "none is not an allowed value" in str(e.value)

    def test_accepts_empty_readings(self, metric: Metric) -> None:
        Measurement(readings=[])

    def test_accepts_empty_time_series(self, metric: Metric) -> None:
        readings = [TimeSeries(metric, [])]
        Measurement(readings=readings)

    @pytest.mark.xfail
    def test_rejects_mismatched_time_series_readings(self, metric: Metric) -> None:
        readings = [
            TimeSeries(
                metric, [(datetime.datetime.now(), 1), (datetime.datetime.now(), 2)]
            ),
            TimeSeries(
                metric,
                [
                    (datetime.datetime.now(), 1),
                    (datetime.datetime.now(), 2),
                    (datetime.datetime.now(), 3),
                ],
                id="foo",
            ),
        ]
        with pytest.raises(ValueError) as e:
            Measurement(readings=readings)
        assert e
        assert (
            'all TimeSeries readings must contain the same number of values: expected 2 values but found 3 on TimeSeries id "foo"'
            in str(e.value)
        )

    @pytest.mark.xfail
    def test_rejects_mixed_empty_and_nonempty_readings(self, metric: Metric) -> None:
        readings = [
            TimeSeries(
                metric, [(datetime.datetime.now(), 1), (datetime.datetime.now(), 2)]
            ),
            TimeSeries(metric=metric, data_points=[]),
        ]
        with pytest.raises(ValueError) as e:
            Measurement(readings=readings)
        assert e
        assert (
            'all TimeSeries readings must contain the same number of values: expected 2 values but found 0 on TimeSeries id "None"'
            in str(e.value)
        )

    def test_rejects_mixed_types_of_readings(self, metric: Metric) -> None:
        readings = [
            TimeSeries(
                metric,
                [
                    DataPoint(metric, datetime.datetime.now(), 1),
                    DataPoint(metric, datetime.datetime.now(), 2),
                ],
            ),
            DataPoint(metric, datetime.datetime.now(), 123),
        ]
        with pytest.raises(ValueError) as e:
            Measurement(readings=readings)
        assert e
        assert (
            'all readings must be of the same type: expected "TimeSeries" but found "DataPoint"'
            in str(e.value)
        )


class TestControl:
    def test_validation_fails_if_delay_past_do_not_agree(self) -> None:
        with pytest.raises(ValueError) as e:
            Control(past=123, delay=456)
        assert e
        assert "past and delay attributes must be equal" in str(e.value)

    def test_past_value_is_coerced_to_delay(self) -> None:
        control = Control(past=123)
        assert control.delay == Duration("2m3s")
