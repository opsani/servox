import pytest
import servo
import servo.api


class TestStatus:
    def test_adjustment_rejected_from_error(self) -> None:
        error = servo.errors.AdjustmentRejectedError("foo")
        status = servo.api.Status.from_error(error)
        assert status.message == 'foo'
        assert status.status == 'rejected'

    def test_event_cancelled_from_error(self) -> None:
        error = servo.errors.EventCancelledError("Command cancelled")
        status = servo.api.Status.from_error(error)
        assert status.message == 'Command cancelled'
        assert status.status == 'cancelled'


from servo.api import descriptor_to_adjustments, CommandResponse, MeasureParams, Status
from servo.types import Adjustment, Control

def _check_measure_parse(obj: CommandResponse):
    assert isinstance(obj, CommandResponse)
    assert isinstance(obj.param, MeasureParams)
    assert obj.param.metrics == [
        "throughput",
        "error_rate",
        "latency_total",
        "latency_mean",
        "latency_50th",
        "latency_90th",
        "latency_95th",
        "latency_99th",
        "latency_max",
        "latency_min",
    ]

def _check_adjust_parse(obj: CommandResponse):
    assert isinstance(obj, CommandResponse)
    adjustments = descriptor_to_adjustments(obj.param["state"])
    adj_strs = []
    for a in adjustments:
        assert isinstance(a, Adjustment)
        adj_strs.append(str(a))

    assert adj_strs == [
        'main.cpu=1.0',
        'main.mem=1.0',
        'main.replicas=2.0',
        'tuning.cpu=1.0',
        'tuning.mem=1.0',
        'tuning.replicas=1.0'
    ]

    control = Control.parse_obj(obj.param["control"])
    assert control.settlement == 60


@pytest.mark.parametrize(
    ('validator', 'payload'),
    [
        (
            _check_measure_parse,
            {
                "cmd": "MEASURE",
                "param": {
                    "control": {
                        "delay": 10,
                        "warmup": 30,
                        "duration": 180,
                    },
                    "metrics": {
                        "throughput": {
                            "unit": "rpm",
                        },
                        "error_rate": {
                            "unit": "%",
                        },
                        "latency_total": {
                            "unit": "ms",
                        },
                        "latency_mean": {
                            "unit": "ms",
                        },
                        "latency_50th": {
                            "unit": "ms",
                        },
                        "latency_90th": {
                            "unit": "ms",
                        },
                        "latency_95th": {
                            "unit": "ms",
                        },
                        "latency_99th": {
                            "unit": "ms",
                        },
                        "latency_max": {
                            "unit": "ms",
                        },
                        "latency_min": {
                            "unit": "ms",
                        },
                    },
                },
            },
        ),
        (
            _check_adjust_parse,
            {
                "cmd": "ADJUST",
                "param": {
                    "state": {
                        "application": {
                            "components": {
                                "main": {
                                    "settings": {
                                        "cpu": {
                                            "value": 1.0
                                        },
                                        "mem": {
                                            "value": 1.0
                                        },
                                        "replicas": {
                                            "value": 2.0
                                        }
                                    }
                                },
                                "tuning": {
                                    "settings": {
                                        "cpu": {
                                            "value": 1.0
                                        },
                                        "mem": {
                                            "value": 1.0
                                        },
                                        "replicas": {
                                            "value": 1.0
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "control": {
                        "settlement": 60
                    }
                }
            },
        )
    ]
)
def test_parse_command_response_including_units_control(payload, validator) -> None:
    from typing import Union

    from pydantic import parse_obj_as

    obj = parse_obj_as(Union[CommandResponse, Status], payload)
    validator(obj)
