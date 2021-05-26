import pytest
import platform
import respx

import servo
import servo.api
from servo.api import descriptor_to_adjustments, CommandResponse, MeasureParams, Status
from servo.types import Adjustment, Control

class TestStatus:
    def test_from_error(self) -> None:
        error = servo.errors.AdjustmentRejectedError("foo")
        status = servo.api.Status.from_error(error)
        assert status.message == 'foo'
        assert status.status == 'rejected'

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

@respx.mock
async def test_user_agent(monkeypatch) -> None:
    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")
    expected = f"github.com/opsani/servox/{servo.__version__} (platform {platform.platform()}; namespace test-namespace)"

    optimizer = optimizer = servo.Optimizer(
        id="servox.opsani.com/tests",
        token="00000000-0000-0000-0000-000000000000",
    )

    # Validate correct construction
    assert optimizer.user_agent == expected

    servo_ = servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald"),
        connectors=[], # Init empty servo
        optimizer=optimizer,
    )
    request = respx.post("https://api.opsani.com/accounts/servox.opsani.com/applications/tests/servo")
    await servo_.dispatch_event("check", matching=None)

    # Validate UA string included in headers
    assert request.called
    assert request.calls.last.request.headers['user-agent'] == expected
