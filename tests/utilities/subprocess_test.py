import asyncio

import pytest

from servo.utilities.subprocess import *


async def test_stream_subprocess_exec():
    output = []
    status_code = await stream_subprocess_exec(
        "/bin/echo", "test", stdout_callback=lambda m: output.append(m)
    )
    assert status_code == 0
    assert output == ["test"]


async def test_run_subprocess_exec():
    status_code, stdout, stderr = await run_subprocess_exec("/bin/echo", "test")
    assert status_code == 0
    assert stdout == ["test"]
    assert stderr == []


async def test_stream_subprocess_shell():
    output = []
    status_code = await stream_subprocess_shell(
        "cd ~/ && echo test", stdout_callback=lambda m: output.append(m)
    )
    assert status_code == 0
    assert output == ["test"]


async def test_run_subprocess_shell():
    status_code, stdout, stderr = await run_subprocess_shell("cd ~/ && echo test")
    assert status_code == 0
    assert stdout == ["test"]
    assert stderr == []


async def test_named_tuple_output():
    result = await run_subprocess_shell("echo test")
    assert result.return_code == 0
    assert result.stdout == ["test"]
    assert result.stderr == []


async def test_run_subprocess_timeout():
    with pytest.raises(asyncio.TimeoutError) as e:
        await stream_subprocess_shell("sleep 60.0", timeout=0.0001)
    assert e


async def test_run_subprocess_timeout():
    output = []
    await stream_subprocess_shell(
        "echo 'test'", stdout_callback=lambda m: output.append(m), timeout=10.0
    )
    assert output == ["test"]
