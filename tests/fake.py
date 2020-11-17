import enum
import yaml
from typing import Any, Dict, List, Optional, Union

import pydantic
import fastapi
import pytest
import statesman

import servo

# TODO: Replace these with models from servo.types
class ServoEvent(pydantic.BaseModel):
    event: str
    param: Union[None, Dict]

class ServoNotifyResponse(pydantic.BaseModel):
    status: str
    message: Optional[str]

class ServoCommandResponse(pydantic.BaseModel):
    cmd: str
    param: Dict

# TODO: Replace this with a model...
# TODO: Need an adjustments_to_response function
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
    
    # TODO: Figure out how to tighten this up...
    command_params: Optional[Dict[str, Any]] = None
    
    @property
    def command(self) -> Commands:
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
            servo.logging.logger.error(f"in non-operational state ({self.state}): cannot command servo meaningfully")
            return Commands.sleep
    
    @statesman.enter_state(States.ready)
    async def _enter_ready(self) -> None:
        self.clear_history()
        self.description = None
        self.connected = False
        self.progress = None
        self.error = None
        self.command_params = None
    
    @statesman.enter_state(States.analyzing)
    async def _enter_analyzing(self) -> None:
        self.progress = None
        self.command_params = None
    
    @statesman.exit_state([States.awaiting_measurement, States.awaiting_adjustment])
    async def _exiting_operation(self) -> None:
        self.command_params = None
        
    @statesman.event("Reset Optimizer", States.__any__, States.ready)
    async def reset(self) -> None:
        """Reset the state machine to an initial ready state."""
        servo.logging.logger.info("Resetting Optimizer")
    
    @statesman.event("Say Hello", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def say_hello(self) -> ServoNotifyResponse:
        """Say hello to a servo that has connected and greeted us.
        
        A servo saying hello toggles the connected state to True.
        """
        servo.logging.logger.info("Saying Hello")
        self.connected = True
        
        return ServoNotifyResponse(status="ok")
    
    @statesman.event("Ask What's Next?", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def ask_whats_next(self) -> ServoCommandResponse:
        """Answer an inquiry about what the next command to be executed is."""
        servo.logging.logger.info(f"Asking What's Next? => {self.command}")
        params = self.command_params or {}
        return ServoCommandResponse(cmd=self.command, param=params)
    
    @statesman.event("Say Goodbye", States.__any__, States.__active__, transition_type=statesman.Transition.Types.self)
    async def say_goodbye(self) -> ServoNotifyResponse:
        """Say goodbye to a servo that is disconnecting and has bid us farewell.
        
        A servo saying goodbye toggles the connected state to False.
        """
        servo.logging.logger.info("Saying Goodbye")
        self.connected = False
        
        return ServoNotifyResponse(status="ok")
    
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
    async def submit_description(self, description: servo.Description) -> ServoNotifyResponse:
        servo.logging.logger.info(f"Description submitted: {description}")
        self.description = description
        
        return ServoNotifyResponse(status="ok")
    
    ##
    # Measure
    
    # TODO: Replace the metrics and control arg...
    @statesman.event("Request Measurement", [States.ready, States.analyzing], States.awaiting_measurement)
    async def request_measurement(self, params: Dict[str, Any] = { "metrics": [], "control": {}}) -> None:
        servo.logging.logger.info("Requesting Measurement")
        self.command_params = params
    
    async def _validate_measurement(self, measurement: servo.Measurement) -> None:
        servo.logging.logger.info(f"Validating Measurement: {measurement}")
        # TODO: Check that the measurement makes sense
        ...
    
    @statesman.event("Submit Measurement", States.awaiting_measurement, States.analyzing, guard=[_guard_progress_tracking, _validate_measurement])
    async def submit_measurement(self, measurement: servo.Measurement) -> ServoNotifyResponse:
        servo.logging.logger.info("Submitting Measurement")
        
        return ServoNotifyResponse(status="ok")
    
    ##
    # Adjust
    
    # TODO: Replace this whole thing with models (adjustments[])
    @statesman.event("Recommend Adjustment", [States.ready, States.analyzing], States.awaiting_adjustment)
    async def recommend_adjustment(self, params: Dict[str, Any]) -> None:
        servo.logging.logger.info("Recommending Adjustment")
        self.command_params = params
    
    async def _validate_adjustment(self, adjustment: servo.Adjustment) -> None:
        servo.logging.logger.info(f"Validating Adjustment: {adjustment}")
        # TODO: Check that the adjustment makes sense
        ...
    
    @statesman.event("Complete Adjustment", States.awaiting_adjustment, States.analyzing, guard=[_guard_progress_tracking, _validate_adjustment])
    async def complete_adjustment(self, adjustment: servo.Adjustment) -> ServoNotifyResponse:
        servo.logging.logger.info("Completing Adjustment")
        
        return ServoNotifyResponse(status="ok")
    
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

class AbstractOptimizer(StateMachine):
    id: str
    token: str
    
    # TODO: define an abstract method for next_state() ?

class StaticOptimizer(AbstractOptimizer):
    """A fake optimizer that requires manual state changes."""
    
    async def to_next_state(self, *args, **kwargs) -> None:
        debug(*args, kwargs)
        pass

class SequencedOptimizer(AbstractOptimizer):
    """A fake optimizer that executes state transitions in a specific order."""
    repeating: bool = False
    
    async def to_next_state(self) -> None:
        pass

class RandomOptimizer(AbstractOptimizer):
    """A fake optimizer that executes state transitions in random order."""
    
    async def to_next_state(self) -> None:
        pass

class ChaosOptimizer(AbstractOptimizer):
    """A fake optimizer that generates chaos.
    
    Provides resilience testing through chaos such as non-sensical metrics,
    invalid adjustment values, etc.
    """
    
    async def to_next_state(self) -> None:
        pass
