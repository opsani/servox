from datetime import timedelta

import pytest
from pydantic import create_model

from servo.types import Duration


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

    def test_pydantic_schema(self) -> None:
        model = create_model("duration_model", duration=(Duration, ...))
        schema = model.schema()
        assert schema["properties"]["duration"] == {
            "title": "Duration",
            "type": "string",
            "format": "duration",
            "pattern": "([\\d\\.]+h)?([\\d\\.]+m)?([\\d\\.]+s)?([\\d\\.]+ms)?([\\d\\.]+us)?([\\d\\.]+ns)?",
            "examples": ["300ms", "5m", "2h45m", "72h3m0.5s",],
        }
