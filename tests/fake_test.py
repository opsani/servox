import pytest
import tests.fake
import servo
import random
from typing import Optional, Union

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

# @pytest.fixture
# def static_optimizer(**kwargs) -> tests.fake.StaticOptimizer:
#     return tests.fake.StaticOptimizer(**kwargs)

# def test_default_initial_state() -> None:
#     optimizer = tests.fake.StaticOptimizer()
#     assert optimizer.is_idle()
    
# def test_set_initial_state() -> None:
#     optimizer = tests.fake.StaticOptimizer(initial=tests.fake.States.done)
#     assert optimizer.is_done()

# @pytest.mark.parametrize(
#     ("state", "event"),
#     [
#         (tests.fake.States.idle, tests.fake.Events.hello),
#         (tests.fake.States.done, tests.fake.Events.goodbye),
#         (tests.fake.States.awaiting_description, tests.fake.Events.whats_next),
#     ]
# )
# async def test_reflexive_events(state, event) -> None:
#     static_optimizer = tests.fake.StaticOptimizer(initial=state)
#     assert static_optimizer.in_state(state)
#     await static_optimizer.dispatch(event)
#     assert static_optimizer.in_state(state)

# import transitions

# @pytest.mark.parametrize(
#     ("initial", "event", "dest", "error_message"),
#     [
#         (tests.fake.States.idle, tests.fake.Events.description, tests.fake.States.idle, '"Can\'t trigger event description from state idle!"'),
#         (tests.fake.States.awaiting_description, tests.fake.Events.description, tests.fake.States.thinking, None),
        
#         (tests.fake.States.idle, tests.fake.Events.measurement, tests.fake.States.idle, '"Can\'t trigger event measurement from state idle!"'),
#         (tests.fake.States.awaiting_measurement, tests.fake.Events.measurement, tests.fake.States.thinking, None),
        
#         (tests.fake.States.idle, tests.fake.Events.adjustment, tests.fake.States.idle, '"Can\'t trigger event adjustment from state idle!"'),
#         (tests.fake.States.awaiting_adjustment, tests.fake.Events.adjustment, tests.fake.States.thinking, None),
#     ]
# )
# async def test_command_events(initial, event, dest, error_message) -> None:    
#     static_optimizer = tests.fake.StaticOptimizer(initial=initial)
#     if error_message is not None:
#         with pytest.raises(transitions.core.MachineError) as e:
#             await static_optimizer.dispatch(event)
#         assert str(e.value) == error_message
#         assert static_optimizer.in_state(dest)
#     else:
#         await static_optimizer.dispatch(event)
#         assert static_optimizer.in_state(dest)

# @pytest.mark.parametrize(
#     ("initial", "event", "progress", "dest", "error_message"),
#     [
#         # progress is ignored for describe ops
#         (tests.fake.States.awaiting_description, tests.fake.Events.describe, None, tests.fake.States.thinking, None),
#         (tests.fake.States.awaiting_description, tests.fake.Events.describe, 10, tests.fake.States.thinking, None),
#         (tests.fake.States.awaiting_description, tests.fake.Events.describe, 100, tests.fake.States.thinking, None),
        
#         (tests.fake.States.awaiting_measurement, tests.fake.Events.measure, None, tests.fake.States.thinking, None),
#         (tests.fake.States.awaiting_measurement, tests.fake.Events.measure, 10, tests.fake.States.awaiting_measurement, None),
#         (tests.fake.States.awaiting_measurement, tests.fake.Events.measure, 100, tests.fake.States.thinking, None),
        
#         (tests.fake.States.awaiting_adjustment, tests.fake.Events.adjust, None, tests.fake.States.thinking, None),
#         (tests.fake.States.awaiting_adjustment, tests.fake.Events.adjust, 10, tests.fake.States.awaiting_adjustment, None),
#         (tests.fake.States.awaiting_adjustment, tests.fake.Events.adjust, 100, tests.fake.States.thinking, None),
#     ]
# )
# async def test_command_events_progress(initial, event, progress, dest, error_message) -> None:    
#     static_optimizer = tests.fake.StaticOptimizer(initial=initial)
#     await static_optimizer.dispatch(event, progress)
#     assert static_optimizer.in_state(dest)
    

# async def test_hello_and_describe() -> None:
#     static_optimizer = tests.fake.StaticOptimizer()
#     assert static_optimizer.in_state(tests.fake.States.idle)
#     await static_optimizer.hello(dict(agent=servo.api.USER_AGENT))
#     assert static_optimizer.in_state(tests.fake.States.idle)
    
#     # manually advance to describe
#     await static_optimizer.to_awaiting_description()
#     assert static_optimizer.in_state(tests.fake.States.awaiting_description)
#     await static_optimizer.describe(dict(agent=servo.api.USER_AGENT))
    
#     # servo_runner.servo.optimizer.base_url = fakeapi_url
#     # response = await servo_runner._post_event(
#     #     servo.api.Event.HELLO, dict(agent=servo.api.USER_AGENT)
#     # )
#     # assert response.status == "ok"
    
#     # description = await servo_runner.describe()
    
#     # param = dict(descriptor=description.__opsani_repr__(), status="ok")
#     # response = await servo_runner._post_event(servo.api.Event.DESCRIPTION, param)
#     # debug(response)
#     # await asyncio.sleep(10)
    
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
        (tests.fake.StateMachine.States.ready, None),
        (tests.fake.StateMachine.States.analyzing, None),
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
    
async def test_whats_next_returns_command() -> None:
    state_machine = await tests.fake.StateMachine.create(state=tests.fake.StateMachine.States.awaiting_description)
    command = await state_machine.ask_whats_next()
    assert command == tests.fake.Commands.describe
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