import servo
import servo.api

class TestStatus:
    def test_from_error(self) -> None:
        error = servo.errors.AdjustmentRejectedError("foo")
        status = servo.api.Status.from_error(error)
        assert status.message == 'foo'
        assert status.status == 'rejected'
