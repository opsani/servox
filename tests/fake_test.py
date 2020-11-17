import pytest
import tests.fake
import servo
import random
import pathlib
import fastapi
from typing import Optional, Union

import servo.runner
import servo.connectors.emulator

METRICS = [
    servo.Metric("throughput", servo.Unit.REQUESTS_PER_MINUTE),
    servo.Metric("error_rate", servo.Unit.PERCENTAGE),
]

COMPONENTS = [
    servo.Component(
        "fake-app",
        settings=[
            servo.CPU(
                min=1,
                max=5
            ),
            servo.Memory(
                min=0.25,
                max=8.0,
                step=0.125
            ),
            servo.Replicas(
                min=1,
                max=10
            )
        ]
    )
]
    
async def test_state_machine() -> None:
    state_machine = await tests.fake.StateMachine.create()
    # await state_machine.enter_state(tests.fake.StateMachine.States.ready)
    # debug(state_machine.states)
    # debug(state_machine.events)
    debug(state_machine.history)
    await state_machine.say_hello()
    debug(state_machine.__class__.__fields__)
    debug(state_machine.__slots__)
    debug(state_machine.history)
    
    await state_machine.ask_whats_next()
    debug(state_machine.__class__.__fields__)
    debug(state_machine.__slots__)
    debug(state_machine.history)
    
    await state_machine.request_description()
    debug(state_machine.__class__.__fields__)
    debug(state_machine.__slots__)
    debug(state_machine.history)
    
    await state_machine.say_goodbye()
    debug(state_machine.__class__.__fields__)
    debug(state_machine.__slots__)
    debug(state_machine.history)
    # print("asdasda")
    # optimizer = FakeOptimizer(id="dev.opsani.com/emulator")
    
    # print("asdasda")
    # # debug(optimizer)
    # debug(optimizer.state)
    # print("Calling Hello!")
    # # optimizer.hello()
    # await optimizer.hello()
    
    # transition = dict(trigger="start", source="Start", dest="Done", prepare="prepare_model",
    #               before=["before_change"] * 5 + ["sync_before_change"],
    #               after="after_change")  # execute before function in asynchronously 5 times    
    # machine = AsyncMachine(optimizer, states=["Start", "Done"], transitions=[transition], initial='Start')

async def test_states() -> None:
    ...

@pytest.mark.parametrize(
    ("state", "expected_command"),
    [
        (tests.fake.StateMachine.States.ready, tests.fake.Commands.sleep),
        (tests.fake.StateMachine.States.analyzing, tests.fake.Commands.sleep),
        (tests.fake.StateMachine.States.awaiting_description, tests.fake.Commands.describe),
        (tests.fake.StateMachine.States.awaiting_measurement, tests.fake.Commands.measure),
        (tests.fake.StateMachine.States.awaiting_adjustment, tests.fake.Commands.adjust),
        (tests.fake.StateMachine.States.done, tests.fake.Commands.sleep),
        (tests.fake.StateMachine.States.failed, tests.fake.Commands.sleep),        
    ]
)
async def test_command(state: tests.fake.StateMachine.States, expected_command: Optional[tests.fake.Commands]) -> None:
    state_machine = await tests.fake.StateMachine.create(state=state)
    assert state_machine.command == expected_command, f"Expected command of {expected_command} but found {state_machine.command}"

async def test_submit_description_stores_description() -> None:
    description = _random_description()
    state_machine = await tests.fake.StateMachine.create(state=tests.fake.StateMachine.States.awaiting_description)
    await state_machine.submit_description(description)
    assert state_machine.description == description
    
async def test_whats_next_returns_command_response() -> None:
    state_machine = await tests.fake.StateMachine.create(state=tests.fake.StateMachine.States.awaiting_description)
    response = await state_machine.ask_whats_next()
    assert response.cmd == tests.fake.Commands.describe
    assert response.param == {}
    assert state_machine.command == tests.fake.Commands.describe    

@pytest.fixture()
def state_machine() -> tests.fake.StateMachine:
    return tests.fake.StateMachine()

@pytest.fixture()
def measurement() -> servo.Measurement:
    return servo.Measurement(
        readings=[
            servo.DataPoint(
                value=31337,
                metric=servo.Metric(
                    name="Some Metric",
                    unit=servo.Unit.REQUESTS_PER_MINUTE,                        
                )
            )
        ]
    )

@pytest.mark.parametrize(
    ("initial_state", "event", "progress", "ending_state"),
    [
        (tests.fake.StateMachine.States.awaiting_measurement, "submit_measurement", 0, tests.fake.StateMachine.States.awaiting_measurement),
        (tests.fake.StateMachine.States.awaiting_measurement, "submit_measurement", None, tests.fake.StateMachine.States.analyzing),
        (tests.fake.StateMachine.States.awaiting_measurement, "submit_measurement", 35, tests.fake.StateMachine.States.awaiting_measurement),
        (tests.fake.StateMachine.States.awaiting_measurement, "submit_measurement", 100, tests.fake.StateMachine.States.analyzing),
        
        (tests.fake.StateMachine.States.awaiting_adjustment, "complete_adjustment", 0, tests.fake.StateMachine.States.awaiting_adjustment),
        (tests.fake.StateMachine.States.awaiting_adjustment, "complete_adjustment", None, tests.fake.StateMachine.States.analyzing),
        (tests.fake.StateMachine.States.awaiting_adjustment, "complete_adjustment", 35, tests.fake.StateMachine.States.awaiting_adjustment),
        (tests.fake.StateMachine.States.awaiting_adjustment, "complete_adjustment", 100, tests.fake.StateMachine.States.analyzing),
    ]
)
async def test_progress_tracking(state_machine: tests.fake.StateMachine, measurement: servo.Measurement, initial_state, event, progress, ending_state) -> None:
    await state_machine.enter_state(initial_state)
    await state_machine.trigger(event, measurement, progress=progress)
    assert state_machine.state == ending_state

async def test_progress_cant_go_backwards(state_machine: tests.fake.StateMachine, measurement: servo.Measurement) -> None:
    await state_machine.enter_state(tests.fake.StateMachine.States.awaiting_measurement)
    await state_machine.submit_measurement(measurement, progress=45)
    with pytest.raises(ValueError, match="progress cannot go backward: new progress value of 22 is less than existing progress value of 45"):
        await state_machine.submit_measurement(measurement, progress=22)
    
# TODO: Turn all of these into fixtures...
def _random_duration() -> servo.Duration:
    seconds = random.randrange(30, 600)
    return servo.Duration(seconds=seconds)

def _random_value_for_setting(setting: servo.Setting) -> Union[str, servo.Numeric]:
    if isinstance(setting, servo.RangeSetting):
        max = int((setting.max - setting.min) / setting.step)
        return random.randint(0, max) * setting.step + setting.min
    elif isinstance(setting, servo.EnumSetting):
        return random.choice(setting.values)
    else:
        raise ValueError(f"unexpected setting: {repr(setting)}")

def _random_description() -> servo.Description:
    components = COMPONENTS.copy()
    metrics = METRICS.copy()
    
    for component in components:
        for setting in component.settings:
            setting.value = _random_value_for_setting(setting)

    return servo.Description(metrics=metrics, components=components)

# @pytest.fixture
# def static_optimizer(**kwargs) -> tests.fake.StaticOptimizer:
#     return tests.fake.StaticOptimizer(**kwargs)

async def test_static_optimizer() -> None:
    static_optimizer = tests.fake.StaticOptimizer(id='dev.opsani.com/big-in-japan', token='31337')
    assert len(static_optimizer.events), "should not be empty"

async def test_hello_and_describe(
    servo_runner: servo.runner.ServoRunner,
    fakeapi_url: str
) -> None:
    static_optimizer = tests.fake.StaticOptimizer(id='dev.opsani.com/big-in-japan', token='31337')
    assert static_optimizer.state == tests.fake.StateMachine.States.ready
    await static_optimizer.say_hello(dict(agent=servo.api.USER_AGENT))
    assert static_optimizer.state == tests.fake.StateMachine.States.ready
    
    # manually advance to describe
    await static_optimizer.request_description()
    assert static_optimizer.state == tests.fake.StateMachine.States.awaiting_description
    await static_optimizer.submit_description(dict(agent=servo.api.USER_AGENT))
    
    servo_runner.servo.optimizer.base_url = fakeapi_url
    response = await servo_runner._post_event(
        servo.api.Event.HELLO, dict(agent=servo.api.USER_AGENT)
    )
    assert response.status == "ok"
    
    description = await servo_runner.describe()
    
    param = dict(descriptor=description.__opsani_repr__(), status="ok")
    response = await servo_runner._post_event(servo.api.Event.DESCRIPTION, param)
    debug(response)
    await asyncio.sleep(10)

# TODO: Mop up all of this shit


@pytest.fixture()
def assembly(servo_yaml: pathlib.Path) -> servo.assembly.Assembly:
    config_model = servo.assembly._create_config_model_from_routes(
        {
            # "prometheus": servo.connectors.prometheus.PrometheusConnector,
            # "adjust": tests.helpers.AdjustConnector,
            "emulator": servo.connectors.emulator.EmulatorConnector
        }
    )
    config = config_model.generate()
    servo_yaml.write_text(config.yaml())

    optimizer = servo.Optimizer(
        id="dev.opsani.com/blake-ignite111",
        token="bfcf94a6e302222eed3c73a5594badcfd53fef4b6d6a703ed32604",
        
    )
    assembly_ = servo.assembly.Assembly.assemble(
        config_file=servo_yaml, optimizer=optimizer
    )
    return assembly_


@pytest.fixture
def assembly_runner(assembly: servo.Assembly) -> servo.runner.AssemblyRunner:
    """Return an unstarted assembly runner."""
    return servo.runner.AssemblyRunner(assembly)

@pytest.fixture
async def servo_runner(assembly: servo.Assembly) -> servo.runner.ServoRunner:
    """Return an unstarted servo runner."""
    return servo.runner.ServoRunner(assembly.servos[0])




#####


api = fastapi.FastAPI()
optimizer = tests.fake.StaticOptimizer(id='dev.opsani.com/big-in-japan', token='31337')

@api.post("/accounts/{account}/applications/{app}/servo")
async def servo_get(account: str, app: str, ev: tests.fake.ServoEvent) -> Union[tests.fake.ServoNotifyResponse, tests.fake.ServoCommandResponse]:
    debug("INVOKED! account, app, ev", account, app, ev)
    if ev.event == "HELLO":
        return await optimizer.say_hello()
    elif ev.event == "GOODBYE":
        return await optimizer.say_goodbye()
    elif ev.event == "WHATS_NEXT":
        return await optimizer.ask_whats_next()
    elif ev.event == "DESCRIPTION":
        ...
    elif ev.event == "MEASUREMENT":
        ...
    elif ev.event == "ADJUSTMENT":
        ...
    else:
        raise ValueError(f"unknown event: {ev.event}")
    
    # if app not in state:
    #     if ev.event == "HELLO":
    #         state[app] = App(name = app)
    #         servo.logging.logger.info(f"Registered new application: {app}")
    #         # fall through to process event
    #     else:
    #         msg = f"Received event {ev.event} for unknown app {app}"
    #         servo.logging.logger.info(msg)
    #         raise fastapi.HTTPException(status_code=400, detail=msg)

    # try:
    #     r = state[app].feed(ev) 
    # except Exception as e:
    #     servo.logging.logger.exception(f"failed with exception: {e}")
    #     raise fastapi.HTTPException(status_code=400, detail=str(e))

    # return r
    return "Move along, nothing to see here"

@pytest.fixture
def fastapi_app() -> fastapi.FastAPI:
    return api
