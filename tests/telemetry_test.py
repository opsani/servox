import httpx
import pytest
import platform
import respx

import servo
import servo.api
import servo.events
import servo.configuration
import servo.runner

@respx.mock
async def test_telemetry_hello(monkeypatch, optimizer: servo.configuration.Optimizer) -> None:
    expected = f'"telemetry": {{"servox.version": "{servo.__version__}", "servox.platform": "{platform.platform()}", "servox.namespace": "test-namespace"}}'

    # Simulate running as a k8s pod
    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    # NOTE: Can't use servo_runner fixture; its init happens prior to setting of env var above
    servo_runner = servo.runner.ServoRunner(servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald", optimizer=optimizer),
        connectors=[], # Init empty servo
    ))

    request = respx.post(
        "https://api.opsani.com/accounts/dev.opsani.com/applications/servox/servo"
    ).mock(return_value=httpx.Response(200, text=f'{{"status": "{servo.api.OptimizerStatuses.ok}"}}'))
    await servo_runner._post_event(servo.api.Events.hello, dict(
        agent=servo.api.user_agent(),
        telemetry=servo_runner.servo.telemetry.values
    ))

    assert request.called
    assert expected in request.calls.last.request.content.decode()
