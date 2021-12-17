import httpx
import platform
import respx

import servo
import servo.api
import servo.events
import servo.configuration
import servo.runner
import servo.telemetry

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
        f"https://api.opsani.com/accounts/dev.opsani.com/applications/servox/{servo.telemetry.DIAGNOSTICS_CHECK_ENDPOINT}"
    ).mock(return_value=httpx.Response(200, text=f'{{"data": "WITHHOLD"}}'))
    request = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._diagnostics_api(method="GET", endpoint=servo.telemetry.DIAGNOSTICS_CHECK_ENDPOINT, output_model=servo.telemetry.DiagnosticStates)
    assert isinstance(request, servo.telemetry.DiagnosticStates)
    assert request == servo.telemetry.DiagnosticStates.withhold

@respx.mock
async def test_diagnostics_put(monkeypatch, optimizer: servo.configuration.Optimizer) -> None:

    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    servo_runner = servo.runner.ServoRunner(servo.Servo(
        config=servo.BaseServoConfiguration(name="archibald", optimizer=optimizer),
        connectors=[], # Init empty servo
    ))

    put = respx.put(
        f"https://api.opsani.com/accounts/dev.opsani.com/applications/servox/{servo.telemetry.DIAGNOSTICS_OUTPUT_ENDPOINT}"
    ).mock(return_value=httpx.Response(200, text=f'{{"status": "ok"}}'))
    diagnostic_data = servo.telemetry.Diagnostics(configmap={'foo': 'bar'}, logs={'foo': 'bar'})
    response = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._diagnostics_api(method="PUT", endpoint=servo.telemetry.DIAGNOSTICS_OUTPUT_ENDPOINT, output_model=servo.api.Status, json=diagnostic_data.dict())

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
        f"https://api.opsani.com/accounts/dev.opsani.com/applications/servox/{servo.telemetry.DIAGNOSTICS_CHECK_ENDPOINT}"
    ).mock(return_value=httpx.Response(200, text=f'{{"status": "ok"}}'))
    reset_state = servo.telemetry.DiagnosticStates.withhold
    response = await servo.telemetry.DiagnosticsHandler(servo_runner.servo)._diagnostics_api(method="PUT", endpoint=servo.telemetry.DIAGNOSTICS_CHECK_ENDPOINT, output_model=servo.api.Status, json=reset_state)

    assert put.called
    assert response.status == servo.api.OptimizerStatuses.ok
