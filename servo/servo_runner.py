from servo.servo import Servo, BaseServoSettings
from servo.connector import Optimizer, logger
from enum import Enum
import httpx
import json
import signal
import sys
import time
import typing
import backoff
from pydantic import BaseModel, Field, parse_obj_as
from servo.metrics import Metric, Component, Setting, Description, Measurement, Control
from typing import List, Optional, Any, Dict, Callable, Union, Tuple
from devtools import pformat

USER_AGENT = 'github.com/opsani/servox'

class Command(str, Enum):    
    DESCRIBE = 'DESCRIBE'
    MEASURE = 'MEASURE'
    ADJUST = 'ADJUST'    
    SLEEP = 'SLEEP'

class Event(str, Enum):
    HELLO = 'HELLO'
    GOODBYE = 'GOODBYE'
    DESCRIPTION = 'DESCRIPTION'
    WHATS_NEXT = 'WHATS_NEXT'
    ADJUSTMENT = 'ADJUSTMENT'
    MEASUREMENT = 'MEASUREMENT'

class Status(BaseModel):
    status: str
    message: Optional[str]

class SleepResponse(BaseModel):
    pass
# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'

# Instructions from servo on what to measure 
class MeasureParams(BaseModel):
    metrics: List[str]
    control: Control

class EventRequest(BaseModel):
    event: Event
    param: Optional[Dict[str, Any]]      # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Event: lambda v: str(v),
        }

class CommandResponse(BaseModel):
    command: Command = Field(
        alias="cmd",
    )
    param: Optional[Union[MeasureParams, Dict[str, Any]]]      # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Command: lambda v: str(v),
        }

class StatusMessage(BaseModel):
    status: str
    message: Optional[str]

# TODO: Review and expand all the error classes
class ConnectorError(Exception):
    """E
    xception indicating that a connector failed
    """

    def __init__(self, *args, status="failed", reason="unknown"):
        self.status = status
        self.reason = reason
        super().__init__(*args)

class ServoRunner:
    servo: Servo
    interactive: bool
    _settings: BaseServoSettings
    _optimizer: Optimizer
    base_url: str
    headers: Dict[str, str]
    _stop_flag: bool

    def __init__(self, servo: Servo, **kwargs) -> None:
        self.servo = servo
        self.interactive = False
        self._settings = servo.settings
        self._optimizer = servo.settings.optimizer
        self.base_url = f'{self._optimizer.base_url}accounts/{self._optimizer.org_domain}/applications/{self._optimizer.app_name}/'
        self.headers = { 'Authorization': f'Bearer {self._optimizer.token}', 'User-Agent': USER_AGENT, 'Content-Type': 'application/json' }
        super().__init__()

    def describe(self) -> Description:
        logger.info('Describing...')

        # TODO: This message dispatch should go through a driver for in-process vs. subprocess
        aggregate_description = Description.construct()
        for connector in self.servo.connectors:
            describe_func = getattr(connector, "describe", None)
            if callable(describe_func): # TODO: This should have a tighter contract (arity, etc)
                description: Description = describe_func()
                aggregate_description.components.extend(description.components)
                aggregate_description.metrics.extend(description.metrics)

        return aggregate_description


    def measure(self, param: MeasureParams) -> Measurement:
        logger.info(f"Measuring... [merics={', '.join(param.metrics)}]")
        logger.trace(pformat(param))

        aggregate_measurement = Measurement.construct()
        for connector in self.servo.connectors:
            measure_func = getattr(connector, "measure", None)
            if callable(measure_func): # TODO: This should have a tighter contract (arity, etc)
                measurement = measure_func(metrics=param.metrics, control=param.control)
                aggregate_measurement.readings.extend(measurement.readings)
                aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    def adjust(self, param) -> dict:
        logger.info('Adjusting...')
        logger.trace(pformat(param))

        for connector in self.servo.connectors:
            adjust_func = getattr(connector, "adjust", None)
            if callable(adjust_func): # TODO: This should have a tighter contract (arity, etc)
                adjustment = adjust_func(param) # TODO: Should be modeled
                result = adjustment.get('status', 'undefined')
                if result == 'ok':
                    logger.info(f'{connector.name} - Adjustment completed')
                    return adjustment
                else:
                    raise ConnectorError('Adjustment driver failed with status "{}" and message:\n{}'.format(
                        status, str(rsp.get('message', 'undefined'))), status=status, reason=adjustment.get('reason', 'undefined'))

        # TODO: Model a response class
        return {}

    # --- Helpers -----------------------------------------------------------------
    
    def http_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self.headers)
    
    def delay(self):
        if self.interactive:
            print('Press <Enter> to continue...', end='')
            sys.stdout.flush()
            sys.stdin.readline()
        elif self.delay:
            time.sleep(1.0)

    @backoff.on_exception(backoff.expo,
                      (httpx.HTTPError),
                      max_time=180,
                      max_tries=12)
    def post_event(self, event: Event, param) -> Union[CommandResponse, Status]:
        '''
        Send request to cloud service. Retry if it fails to connect.
        '''

        event_request = EventRequest(event=event, param=param)
        with self.http_client() as client:
            try:
                response = client.post('servo', data=event_request.json())
                response.raise_for_status()
            except httpx.HTTPError as error:
                logger.exception(f"HTTP error encountered while posting {event.value} event")
                logger.trace(pformat(event_request))
                raise error
        
        return parse_obj_as(Union[CommandResponse, Status], response.json())

    def exec_command(self):
        cmd_response = self.post_event(Event.WHATS_NEXT, None)
        logger.debug(f"What's Next? => {cmd_response.command}")
        logger.trace(pformat(cmd_response))

        try:
            if cmd_response.command == Command.DESCRIBE:
                description = self.describe()
                logger.info(f"Described: {len(description.components)} components, {len(description.metrics)} metrics")
                logger.trace(pformat(description))
                param = dict(descriptor=description.opsani_dict(), status='ok')
                self.post_event(Event.DESCRIPTION, param)

            elif cmd_response.command == Command.MEASURE:                
                measurement = self.measure(cmd_response.param)
                logger.info(f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations")
                logger.trace(pformat(measurement))
                param = measurement.opsani_dict()
                self.post_event(Event.MEASUREMENT, param)

            elif cmd_response.command == Command.ADJUST:                
                # # TODO: This needs to be modeled
                #oc"{'cmd': 'ADJUST', 'param': {'state': {'application': {'components': {'web': {'settings': {'cpu': {'value': 0.225}, 'mem': {'value': 0.1}}}}}}, 'control': {}}}"                

                # TODO: Why do we do this nonsense??
                # create a new dict based on p['state'] (with its top level key
                # 'application') which also includes a top-level 'control' key, and
                # pass this to adjust()
                new_dict = cmd_response.param['state'].copy()
                new_dict['control'] = cmd_response.param.get('control', {})
                adjustment = self.adjust(new_dict)
                
                # TODO: What works like this and why?
                if 'state' not in adjustment: # if driver didn't return state, assume it is what was requested
                    adjustment['state'] = cmd_response.param['state']

                components_dict = adjustment['state']['application']['components']
                components_count = len(components_dict)                
                settings_count = sum(len(components_dict[component]['settings']) for component in components_dict)
                logger.info(f"Adjusted: {components_count} components, {settings_count} settings_count")

                self.post_event(Event.ADJUSTMENT, adjustment)

            elif cmd_response.command == Command.SLEEP:
                if not self.interactive: # ignore sleep request when interactive - let user decide
                    # TODO: Model this
                    duration = int(cmd_response.param.get('duration', 120))
                    logger.info(f'Sleeping {duration} sec.')
                    time.sleep(duration)

            else:
                raise ValueError(f"Unknown command '{cmd_response.command.value}'")
            
        except Exception as error:
            logger.exception(f"{cmd_response.command} command failed!")
            param = dict(status='failed', message=_exc_format(error))
            sys.exit(2)
            self.post_event(_event_for_command(cmd_response.command), param)    

    def run(self) -> None:        
        self._stop_flag = False
        self._init_signal_handlers()

        logger.info(f"Servo starting with {len(self.servo.connectors)} active connectors [{self._optimizer.id} @ {self._optimizer.base_url}]")

        # announce
        logger.info('Saying HELLO.', end=' ')
        self.delay()
        self.post_event(Event.HELLO, dict(agent=USER_AGENT))

        while not self._stop_flag:
            try:
                self.exec_command()
            except Exception as e:
                logger.exception("Exception encountered while executing command")

        try:
            self.post_event(Event.GOODBYE, dict(reason=self.stop_flag))
        except Exception as e:
            logger.exception('Warning: failed to send GOODBYE: {}. Exiting anyway'.format(str(e)))

    ###    
    # TODO: Factor into a signal handler class

    def _init_signal_handlers(self):
        # intercept SIGINT to provide graceful, traceback-less Ctrl-C/SIGTERM handling
        signal.signal(signal.SIGTERM, self._signal_handler) # container kill
        signal.signal(signal.SIGINT, self._signal_handler) # Ctrl-C
        signal.signal(signal.SIGUSR1, self._graceful_stop_handler)
        signal.signal(signal.SIGHUP, self._graceful_restart_handler)

    def _signal_handler(self, sig_num, unused_frame):
        # restore original signal handler (to prevent reentry)
        signal.signal(sig_num, signal.SIG_DFL)

        # determine signal name (best effort)
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = 'signal #{}'.format(sig_num)

        # log signal
        logger.info(f'*** Servo stop requested by signal "{format(sig_name)}". Sending GOODBYE')

        # send GOODBYE event (best effort)
        try:
            self.post_event(Event.GOODBYE, dict(reason=sig_name))
        except Exception as e:
            logger.exception('Warning: failed to send GOODBYE: {}. Exiting anyway'.format(str(e)))

        sys.exit(0) # abort now

    def _graceful_stop_handler(self, sig_num, unused_frame):
        """handle signal for graceful termination - simply set a flag to have the main loop exit after the current operation is completed"""
        self._stop_flag = "exit"

    def _graceful_restart_handler(self, sig_num, unused_frame):
        """handle signal for restart - simply set a flag to have the main loop exit and restart the process after the current operation is completed"""
        self._stop_flag = "restart"

def _event_for_command(command: Command) -> Optional[Event]:
    if cmd_response.command == Command.DESCRIBE:
        return Event.DESCRIPTION
    elif cmd_response.command == Command.MEASURE:
        return Event.MEASUREMENT
    elif cmd_response.command == Command.ADJUST:
        return Event.ADJUSTMENT
    else:
        return None

def _exc_format(e):
    if type(e) is Exception:  # if it's just an Exception
        return str(e) # print only the message but not the type
    return "{}: {}".format(type(e).__name__, str(e)) # print the exception type and message
