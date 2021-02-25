import servo
import servo.api

import platform
import respx

class TestStatus:
    def test_from_error(self) -> None:
        error = servo.errors.AdjustmentRejectedError("foo")
        status = servo.api.Status.from_error(error)
        assert status.message == 'foo'
        assert status.status == 'rejected'

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
