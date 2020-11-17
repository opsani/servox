import abc
import asyncio
import datetime
import enum
import random
import yaml
from typing import Any, Dict, List, Optional, Union

import pydantic
import statesman

import servo

# TODO: Replace these with native models?
class ServoEvent(pydantic.BaseModel):
    event: str
    param: Union[None, Dict]

class ServoNotifyResponse(pydantic.BaseModel):
    status: str
    message: Optional[str]

class ServoCommandResponse(pydantic.BaseModel):
    cmd: str
    param: Dict


class SeqError(Exception):
    pass

# TODO: Replace this with model...
ADJUST_PAYLOAD = yaml.safe_load(    # payload to send for all adjust commands - must match emulator
    """
    state:
        application:
            components:
                web:
                    settings:
                        cpu: { value: 1 }
                        replicas: { value: 3 }
                java:
                    settings:
                        mem: { value: 2 }
                        GCTimeRatio: { value: 99 }
                db:
                    settings:
                        cpu: { value: 1 }
                        commit_delay: { value: 0 }
    """
)

class Events(str, enum.Enum):
    hello = "HELLO"
    whats_next = "WHATS_NEXT"
    describe = "DESCRIPTION"
    measure = "MEASUREMENT"
    adjust = "ADJUSTMENT"
    goodbye = "GOODBYE"


class Commands(str, enum.Enum):
    describe = "DESCRIBE"
    measure = "MEASURE"
    adjust = "ADJUST"
    sleep = "SLEEP"

class StateMachine(statesman.HistoryMixin, statesman.StateMachine):
    class States(statesman.StateEnum):
        ready = statesman.InitialState("Ready")
        analyzing = "Analyzing"
        
        awaiting_description = "Awaiting Description" # issued a DESCRIBE, waiting for confirmation
        awaiting_measurement = "Awaiting Measurement" # issued a MEASURE, waiting for results    
        awaiting_adjustment = "Awaiting Adjustment" # issued an ADJUST, waiting for confirmation
        
        done = "Done" # Optimizer completed
        failed = "Failed"
    
    description: Optional[servo.Description] = None
    connected: bool = False
    progress: Optional[int] = None
    error: Optional[Exception] = None
    
    @property
    def command(self) -> Optional[Commands]:
        """Return the command for the current state."""
        if self.state == StateMachine.States.awaiting_description:
            return Commands.describe
        elif self.state == StateMachine.States.awaiting_measurement:
            return Commands.measure
        elif self.state == StateMachine.States.awaiting_adjustment:
            return Commands.adjust
        elif self.state in (StateMachine.States.done, StateMachine.States.failed):
            return Commands.sleep
        else:
            return None
    
    @statesman.enter_state(States.ready)
    async def _enter_ready(self) -> None:
        self.clear_history()
        self.description = None
        self.connected = False
        self.progress = None
        self.error = None
    
    @statesman.enter_state(States.analyzing)
    async def _enter_analyzing(self) -> None:
        self.progress = None
        
    @statesman.event("Reset Optimizer", States.__any__, States.ready)
    async def reset(self) -> None:
        """Reset the state machine to an initial ready state."""
        servo.logging.logger.info("Resetting Optimizer")
    
    @statesman.event("Say Hello", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def say_hello(self) -> None:
        """Say hello to a servo that has connected and greeted us.
        
        A servo saying hello toggles the connected state to True.
        """
        servo.logging.logger.info("Saying Hello")
        self.connected = True
    
    @statesman.event("Ask What's Next?", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def ask_whats_next(self) -> Optional[Commands]:
        """Answer an inquiry about what the next command to be executed is."""
        servo.logging.logger.info(f"Asking What's Next? => {self.command}")
        return self.command
    
    @statesman.event("Say Goodbye", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def say_goodbye(self) -> None:
        """Say goodbye to a servo that is disconnecting and has bid us farewell.
        
        A servo saying goodbye toggles the connected state to False.
        """
        servo.logging.logger.info("Saying Goodbye")
        self.connected = False
    
    async def _guard_progress_tracking(self, *, progress: Optional[int] = None) -> bool:
        if isinstance(progress, int) and progress < 100:
            if self.progress and progress < self.progress:
                raise ValueError(f"progress cannot go backward: new progress value of {progress} is less than existing progress value of {self.progress}")
            self.progress = progress
            return False

        return True
    
    ##
    # Describe
    
    @statesman.event("Request Description", [States.ready, States.analyzing], States.awaiting_description)
    async def request_description(self) -> None:
        servo.logging.logger.info("Requesting Description")
    
    async def _validate_description(self, description: servo.Description) -> None:
        servo.logging.logger.info(f"Validating Description: {description}")
        # TODO: Check that the description makes sense
        ...

    @statesman.event("Submit Description", States.awaiting_description, States.analyzing, guard=_validate_description)
    async def submit_description(self, description: servo.Description) -> None:
        servo.logging.logger.info(f"Description submitted: {description}")
        self.description = description
    
    ##
    # Measure
    
    @statesman.event("Request Measurement", [States.ready, States.analyzing], States.awaiting_measurement)
    async def request_measurement(self) -> None:
        servo.logging.logger.info("Requesting Measurement")
    
    async def _validate_measurement(self, measurement: servo.Measurement) -> None:
        servo.logging.logger.info(f"Validating Measurement: {measurement}")
        # TODO: Check that the measurement makes sense
        ...
    
    @statesman.event("Submit Measurement", States.awaiting_measurement, States.analyzing, guard=[_guard_progress_tracking, _validate_measurement])
    async def submit_measurement(self, measurement: servo.Measurement) -> None:
        servo.logging.logger.info("Submitting Measurement")
        ...
    
    ##
    # Adjust
    
    @statesman.event("Recommend Adjustment", [States.ready, States.analyzing], States.awaiting_adjustment)
    async def recommend_adjustment(self) -> None:
        servo.logging.logger.info("Recommending Adjustment")
        ...
    
    async def _validate_adjustment(self, adjustment: servo.Adjustment) -> None:
        servo.logging.logger.info(f"Validating Adjustment: {adjustment}")
        # TODO: Check that the adjustment makes sense
        ...
    
    @statesman.event("Complete Adjustment", States.awaiting_adjustment, States.analyzing, guard=[_guard_progress_tracking, _validate_adjustment])
    async def complete_adjustment(self, adjustment: servo.Adjustment) -> None:
        servo.logging.logger.info("Completing Adjustment")
        ...
    
    ##
    # Terminal transitions
    
    @statesman.event("Fail Optimization", States.__any__, States.failed)
    async def fail(self, error: Exception) -> None:
        servo.logging.logger.info("Failing Optimization")
        ...
    
    @statesman.event("Complete Optimization", States.__any__, States.done)
    async def done(self) -> None:
        servo.logging.logger.info("Completing Optimization")
        ...
    
    class Config:
        arbitrary_types_allowed = True

# class TransitionTable:
#     ...
#     # Input, Current State, Next State, Output
#     # State, Event, Input Params, Expected Next State, Output???

# state_machine = StateMachine()
# @state_machine.expect(
#     # Entry State, Expected Event, Expected Exit State, Expected Output
#     (None, state_machine.reset(), StateMachine.States.ready, None),
#     (StateMachine.States.ready, state_machine.whats_next(), StateMachine.States.awaiting_description, ...),
    
#     (StateMachine.States.awaiting_description, state_machine.submit_description(ExpectedDescription()), StateMachine.States.ready, ...),
    
#     (StateMachine.States.ready, state_machine.whats_next(), StateMachine.States.awaiting_measurement, _verify_measure_command),
#     (StateMachine.States.awaiting_measurement, state_machine.submit_measurement(ExpectedMeasurement()), StateMachine.States.analyzing, _verify_measurement_response),
# )
# async def verify(transition: statesman.Transition, expected: Any) -> None:
#     ...

# sequence = statesman.Sequence(
#      state_machine.whats_next(),
#      state_machine.request_description(),
#      state_machine.submit_description(),
     
#      state_machine.whats_next(),
#      state_machine.request_measurement(),
#      state_machine.submit_measurement(),
     
#      state_machine.whats_next(),
#      state_machine.recommend_adjustment(),
#      state_machine.complete_adjustment(),
     
#      entry=state_machine.States.ready,
#      iterations=10
# )

# await sequence.enter()
# await sequence.next()
# await sequence.run()

# state_machine = StateMachine()
# table = statesman.TransitionTable(
#     # State, Event, Expected State, Output
#     # Event, Expected State, Output???
#     # Entry State, Event, Exit State, Verifier
#     [state_machine.reset(), StateMachine.States.__any__, StateMachine.States.ready, state_machine.empty()],
#     [state_machine.say_hello(), StateMachine.states.ready, state_machine.empty()],
    
#     [None, State.ready, ]
    
#     (None, StateMachine.activate, StateMachine.States.ready, Commands.Adjust("whatever"))
    
#     (StateMachine.say_hello(None)
#     request_description
#     submit_description
    
#     request_measurement
#     submit_measurement
    
#     recommend_adjustment
#     complete_adjustment
    
#     say_goodbye,
#     Analyzing, reset
# )

class AbstractOptimizer:
    # state machine...
    ...

# class StaticOptimizer(AbstractOptimizer):
#     """A fake optimizer that requires manual state changes."""
    
#     async def to_next_state(self, *args, **kwargs) -> None:
#         debug(*args, kwargs)
#         pass

# class SequencedOptimizer(AbstractOptimizer):
#     """A fake optimizer that executes state transitions in a specific order."""
#     repeating: bool = False
    
#     async def to_next_state(self) -> None:
#         pass

# class RandomOptimizer(AbstractOptimizer):
#     """A fake optimizer that executes state transitions in random order."""
    
#     async def to_next_state(self) -> None:
#         pass

# class ChaosOptimizer(AbstractOptimizer):
#     """A fake optimizer that generates chaos.
    
#     Provides resilience testing through chaos such as non-sensical metrics,
#     invalid adjustment values, etc.
#     """
    
#     async def to_next_state(self) -> None:
#         pass

# TODO: Try to make app apply out of range, put it into weird situations
# For Kubernetes, READY -> INQUIRING/WAITING -> ADJUSTING

# class App:
#     name: str
#     state: str = "initial"
#     next_cmd: Optional[str] = None  # if None, WHATS_NEXT is invalid
#     curr_cmd: Optional[str] = None
#     curr_cmd_progress: Optional[int] = None
#     n_cycles: int = 0

#     fsmtab = {
#         # state             event           progress  new state          next command  cycle++
#         ("initial"        , "HELLO"       , False  ): ("hello-received", "DESCRIBE" , False ),
#         ("hello-received" , "DESCRIPTION" , False  ): ("start-measure" , "MEASURE"  , False ),
#         ("start-measure"  , "MEASUREMENT" , True   ): ("in-measure"    , None       , False ),
#         ("in-measure"     , "MEASUREMENT" , True   ): ("in-measure"    , None       , False ),
#         ("in-measure"     , "MEASUREMENT" , False  ): ("start-adjust"  , "ADJUST"   , True  ),
#         ("start-adjust"   , "ADJUSTMENT"  , True   ): ("in-adjust"     , None       , False ),
#         ("in-adjust"      , "ADJUSTMENT"  , True   ): ("in-adjust"     , None       , False ),
#         ("in-adjust"      , "ADJUSTMENT"  , False  ): ("start-measure" , "MEASURE"  , False ),

#         ("hello-received" , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
#         ("start-measure"  , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
#         ("in-measure"     , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
#         ("start-adjust"   , "GOODBYE"     , False  ): ("terminated"    , None       , False ),
#         ("in-adjust"      , "GOODBYE"     , False  ): ("terminated"    , None       , False ),

#         ("terminated"     , "HELLO"       , False  ): ("hello-received", "DESCRIBE" , False )
#     }

#     def __init__(self, name: str):
#         self.name = name

#     def _exc(self, msg: str):
#         return SeqError(f"Sequence error for {self.name}: {msg}")

#     def feed(self, ev: ServoEvent) -> Union[ServoNotifyResponse, ServoCommandResponse]:
#         servo.logging.logger.info(f"Processing {ev}")
#         # handle next command query
#         if ev.event == "WHATS_NEXT":
#             if self.next_cmd == None:
#                 raise self._exc(f"Unexpected WHATS_NEXT in {self.state}")
#             (cmd, self.next_cmd) = (self.next_cmd, None) # clear next_cmd
#             self.curr_cmd = None
#             self.curr_cmd_progress = None
#             if cmd == "ADJUST":
#                 params = ADJUST_PAYLOAD
#             elif cmd == "MEASURE":
#                 params = { "metrics": [], "control": {}}
#             else:
#                 params = {}
#             return ServoCommandResponse(cmd = cmd, param = params)

#         # lookup event and extract sequencer instruction
#         try:
#             progress = int(ev.param["progress"])
#         except Exception:
#             progress = None
#         key = (self.state, ev.event, progress is not None)
#         try:
#             instr = self.fsmtab[key]
#         except KeyError:
#             pmsg = "" if progress is None else f"({progress}%)"
#             raise self._exc(f"Unexpected event {ev}{pmsg} in state {self.state}")

#         # check progress
#         if progress is not None:
#             if self.curr_cmd_progress is not None and progress < self.curr_cmd_progress:
#                 raise self._exc(f"Progress going backwards: old={self.curr_cmd_progress}, new={progress}")
#             else:
#                 self.curr_cmd_progress = progress
#         else:
#             curr_cmd_progress = None # zero out progress on completed commands

#         # "execute" instruction
#         (self.state, self.next_cmd, bump_cycle) = instr 
#         if bump_cycle:
#             self.n_cycles += 1

#         servo.logging.logger.info(f"#seq# {self.name} processed {key} -> {instr}, {self.n_cycles} cycles {' [TERMINATED]' if self.state == 'terminated' else ''}")

#         return ServoNotifyResponse(status = "ok")

# app = fastapi.FastAPI()
# state = {}

# # TODO: Create FakeOptimizer class... maintain a list of them
# # Need a good API for describing the steps...

# @app.post("/accounts/{account}/applications/{app}/servo")
# async def servo_get(account: str, app: str, ev: ServoEvent) -> Union[ServoNotifyResponse, ServoCommandResponse]:
#     if app not in state:
#         if ev.event == "HELLO":
#             state[app] = App(name = app)
#             servo.logging.logger.info(f"Registered new application: {app}")
#             # fall through to process event
#         else:
#             msg = f"Received event {ev.event} for unknown app {app}"
#             servo.logging.logger.info(msg)
#             raise fastapi.HTTPException(status_code=400, detail=msg)

#     try:
#         r = state[app].feed(ev) 
#     except Exception as e:
#         servo.logging.logger.exception(f"failed with exception: {e}")
#         raise fastapi.HTTPException(status_code=400, detail=str(e))

#     return r

# @pytest.fixture
# def fastapi_app() -> fastapi.FastAPI:
#     return app
