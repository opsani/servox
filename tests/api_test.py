import servo
import servo.api

class TestStatus:
    def test_rejected_from_error(self) -> None:
        error = servo.errors.AdjustmentRejectedError("foo")
        status = servo.api.Status.from_error(error)
        assert status.message == 'foo'
        assert status.status == 'rejected'

    def test_env_failed_from_error(self) -> None:
        error = servo.errors.EnvironmentFailedError("bar")
        status = servo.api.Status.from_error(error)
        assert status.message == 'bar'
        assert status.status == 'environment-failed'

    def test_failed_from_error(self) -> None:
        error = servo.errors.BaseError("baz")
        status = servo.api.Status.from_error(error)
        assert status.message == 'baz'
        assert status.status == 'failed'
