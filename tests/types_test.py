from datetime import timedelta

import pytest
from pydantic import create_model

from servo.types import Adjustment, Component, Duration


class TestDuration:
    def test_init_with_seconds(self) -> None:
        duration = Duration(120)
        assert duration.total_seconds() == 120

    def test_init_with_timedelta(self) -> None:
        td = timedelta(seconds=120)
        duration = Duration(td)
        assert duration.total_seconds() == 120

    def test_init_with_duration_str(self) -> None:
        duration = Duration("5m")
        assert duration.total_seconds() == 300

    def test_init_with_invalid_str(self) -> None:
        with pytest.raises(ValueError) as error:
            Duration("invalid")
        assert str(error.value) == "Invalid duration 'invalid'"

    def test_init_with_time_components(self) -> None:
        duration = Duration(hours=10, seconds=25)
        assert duration.total_seconds() == 36025.0

    def test_eq_str(self) -> None:
        duration = Duration(300)
        assert duration == "5m"

    def test_eq_timedelta(self) -> None:
        duration = Duration(18_000)
        assert duration == timedelta(hours=5)

    def test_eq_numeric(self) -> None:
        duration = Duration("5h")
        assert duration == 18_000

    def test_repr(self) -> None:
        duration = Duration("5h")
        assert duration.__repr__() == "Duration('5h' 5:00:00)"

    def test_str(self) -> None:
        duration = Duration("5h37m15s")
        assert duration.__str__() == "5h37m15s"
    
    def test_parse_extended(self) -> None:
        duration = Duration("2y4mm3d8h24m")
        assert duration.__str__() == "2y4mm3d8h24m"
        assert duration.total_seconds() == 73729440.0
        assert Duration(73729440.0).__str__() == "2y4mm3d8h24m"

    def test_pydantic_schema(self) -> None:
        model = create_model("duration_model", duration=(Duration, ...))
        schema = model.schema()
        assert schema["properties"]["duration"] == {
            "title": "Duration",
            "type": "string",
            "format": "duration",
            "pattern": "([\\d\\.]+y)?([\\d\\.]+mm)?(([\\d\\.]+w)?[\\d\\.]+d)?([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
            "examples": ["300ms", "5m", "2h45m", "72h3m0.5s",],
        }

def test_adjustment_str() -> None:
    adjustment = Adjustment(component_name="web", setting_name="cpu", value=1.25)
    assert adjustment.__str__() == "web.cpu=1.25"
    assert str(adjustment) == "web.cpu=1.25"
    assert f"adjustment=({adjustment})" == "adjustment=(web.cpu=1.25)"
    assert f"[adjustments=({', '.join(list(map(str, [adjustment])))})]" == "[adjustments=(web.cpu=1.25)]"


# TODO: Move to api_test.py
def test_parse_measure_command_response_including_units() -> None:
    from pydantic import parse_obj_as
    from typing import Union
    from servo.api import CommandResponse, MeasureParams, Status
    payload = {
        'cmd': 'MEASURE',
        'param': {
            'control': {
                'delay': 10,
                'warmup': 30,
                'duration': 180,
            },
            'metrics': {
                'throughput': {
                    'unit': 'rpm',
                },
                'error_rate': {
                    'unit': '%',
                },
                'latency_total': {
                    'unit': 'ms',
                },
                'latency_mean': {
                    'unit': 'ms',
                },
                'latency_50th': {
                    'unit': 'ms',
                },
                'latency_90th': {
                    'unit': 'ms',
                },
                'latency_95th': {
                    'unit': 'ms',
                },
                'latency_99th': {
                    'unit': 'ms',
                },
                'latency_max': {
                    'unit': 'ms',
                },
                'latency_min': {
                    'unit': 'ms',
                },
            },
        }
    }
    obj = parse_obj_as(Union[CommandResponse, Status], payload)
    assert isinstance(obj, CommandResponse)
    assert isinstance(obj.param, MeasureParams)
    assert obj.param.metrics == ['throughput', 'error_rate', 'latency_total', 'latency_mean', 'latency_50th', 'latency_90th', 'latency_95th', 'latency_99th', 'latency_max', 'latency_min']
