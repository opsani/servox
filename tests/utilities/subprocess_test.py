import asyncio
import pytest
import servo.utilities.subprocess

async def test_stream_subprocess_exec():
    output = []
    status_code = await servo.utilities.subprocess.stream_subprocess_exec(
        "/bin/echo", "test", stdout_callback=lambda m: output.append(m)
    )
    assert status_code == 0
    assert output == ["test"]


async def test_run_subprocess_exec():
    status_code, stdout, stderr = await servo.utilities.subprocess.run_subprocess_exec("/bin/echo", "test")
    assert status_code == 0
    assert stdout == ["test"]
    assert stderr == []


async def test_stream_subprocess_shell():
    output = []
    status_code = await servo.utilities.subprocess.stream_subprocess_shell(
        "echo test", stdout_callback=lambda m: output.append(m)
    )
    assert status_code == 0
    assert output == ["test"]


async def test_run_subprocess_shell():
    status_code, stdout, stderr = await servo.utilities.subprocess.run_subprocess_shell("echo test")
    assert status_code == 0
    assert stdout == ["test"]
    assert stderr == []


async def test_named_tuple_output():
    result = await servo.utilities.subprocess.run_subprocess_shell("echo test")
    assert result.return_code == 0
    assert result.stdout == ["test"]
    assert result.stderr == []


async def test_run_subprocess_timeout():
    with pytest.raises(asyncio.TimeoutError) as e:
        await servo.utilities.subprocess.stream_subprocess_shell("sleep 60.0", timeout=0.0001)
    assert e


async def test_run_subprocess_timeout():
    output = []
    await servo.utilities.subprocess.stream_subprocess_shell(
        "echo 'test'", stdout_callback=lambda m: output.append(m), timeout=10.0
    )
    assert output == ["test"]
