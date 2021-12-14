import httpx
import pytest
import respx

import servo
import servo.telemetry


class TestDiagnosticStates:
    def test_withhold_diagnostic_state(self) -> None:
        state = servo.telemetry.DiagnosticStates("WITHHOLD")
        assert state == servo.telemetry.DiagnosticStates.withhold

    def test_send_diagnostic_state(self) -> None:
        state = servo.telemetry.DiagnosticStates("SEND")
        assert state == servo.telemetry.DiagnosticStates.send

    def test_stop_diagnostic_state(self) -> None:
        state = servo.telemetry.DiagnosticStates("STOP")
        assert state == servo.telemetry.DiagnosticStates.stop

@respx.mock
async def test_diagnostics_request(monkeypatch, optimizer: servo.configuration.Optimizer) -> None:

    # Simulate running as a k8s pod
    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    # NOTE: Can't use servo_runner fixture; its init happens prior to setting of env var above
    servo_runner = servo.runner.ServoRunner(servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald", optimizer=optimizer),
        connectors=[], # Init empty servo
    ))

    request = respx.get(
        "https://api.opsani.com/accounts/dev.opsani.com/applications/servox/assets/opsani.com/diagnostics-check"
    ).mock(return_value=httpx.Response(200, text=f'{{"data": "WITHHOLD"}}'))
    request = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._diagnostics_request()
    assert isinstance(request, servo.telemetry.DiagnosticStates)
    assert request == servo.telemetry.DiagnosticStates.withhold

@respx.mock
async def test_diagnostics_post(monkeypatch, optimizer: servo.configuration.Optimizer) -> None:

    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    servo_runner = servo.runner.ServoRunner(servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald", optimizer=optimizer),
        connectors=[], # Init empty servo
    ))

    put = respx.put(
        "https://api.opsani.com/accounts/dev.opsani.com/applications/servox/assets/opsani.com/diagnostics-output"
    ).mock(return_value=httpx.Response(200, text=f'{{"status": "ok"}}'))
    diagnostic_data = servo.telemetry.Diagnostics(configmap={'foo': 'bar'}, logs={'foo': 'bar'})
    response = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._put_diagnostics(diagnostic_data)

    assert put.called
    assert response.status == servo.api.OptimizerStatuses.ok

@respx.mock
async def test_diagnostics_reset(monkeypatch, optimizer: servo.configuration.Optimizer) -> None:

    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    servo_runner = servo.runner.ServoRunner(servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald", optimizer=optimizer),
        connectors=[], # Init empty servo
    ))

    put = respx.put(
        "https://api.opsani.com/accounts/dev.opsani.com/applications/servox/assets/opsani.com/diagnostics-check"
    ).mock(return_value=httpx.Response(200, text=f'{{"status": "ok"}}'))
    response = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._reset_diagnostics()

    assert put.called
    assert response.status == servo.api.OptimizerStatuses.ok
