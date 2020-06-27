from servo.servo import Servo, BaseServoSettings
from servo.connector import Optimizer
from enum import Enum
import httpx
import loguru
import json
import signal
import sys
import time
import typing
import traceback
from pydantic import BaseModel, Field
from servo.metrics import Metric, Component, Setting, Description, Measurement

USER_AGENT = 'github.com/opsani/servox'

# TODO: Rename to Command
class Command(str, Enum):    
    DESCRIBE = 'DESCRIBE'
    MEASURE = 'MEASURE'
    ADJUST = 'ADJUST'    
    SLEEP = 'SLEEP'

class Event(str, Enum):
    HELLO = 'HELLO'
    GOODBY = 'GOODBYE'
    DESCRIPTION = 'DESCRIPTION'
    WHATS_NEXT = 'WHATS_NEXT'
    ADJUSTMENT = 'ADJUSTMENT'
    MEASUREMENT = 'MEASUREMENT'

class Status(BaseModel):
    status: str
    message: str

class SleepResponse(BaseModel):
    pass
# SleepResponse '{"cmd": "SLEEP", "param": {"duration": 60, "data": {"reason": "no active optimization pipeline"}}}'

# Instructions from servo on what to measure 
class MeasureParams(BaseModel):
    metrics: List[str]
    control: Control

class EventRequest(BaseModel):
    event: Event
    param: typing.Optional[typing.Dict[str, typing.Any]]      # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Event: lambda v: str(v),
        }

class CommandResponse(BaseModel):
    command: Command = Field(
        alias="cmd",
    )
    param: typing.Optional[typing.Union[MeasureParams, typing.Dict[str, typing.Any]]]      # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            Command: lambda v: str(v),
        }

class ServoRunner:
    servo: Servo
    interactive: bool
    _settings: BaseServoSettings
    _optimizer: Optimizer
    base_url: str
    headers: typing.Dict[str, str]
    _stop_flag: bool

    def __init__(self, servo: Servo, **kwargs) -> None:
        self.servo = servo
        self.interactive = False
        self._settings = servo.settings
        self._optimizer = servo.settings.optimizer
        self.base_url = f'{self._optimizer.base_url}accounts/{self._optimizer.org_domain}/applications/{self._optimizer.app_name}/'
        self.headers = { 'Authorization': f'Bearer {self._optimizer.token}', 'User-Agent': USER_AGENT, 'Content-Type': 'application/json' }
        super().__init__()


    def describe(self):
        print('describing')

        # Gather all the metrics returned and build a payload
        # TODO: This message dispatch should go through a driver for in-process vs. subprocess
        # Aggregate a set of metrics and components across all responsive connectors
        metrics = []
        components = [
            Component(name="web", settings=[
                Setting(
                    name="cpu",
                    type="range",
                    min="0.1",
                    max="10.0",
                    step="0.125",
                    value=3.0
                ),
            ]),
        ]
        for connector in self.servo.connectors:
            # TODO: This should probably be driven off a decorator (Servo defines it, connectors opt-in)
            describe_func = getattr(connector, "describe", None)
            if callable(describe_func): # TODO: This should have a tighter contract (arity, etc)
                description: Description = describe_func()
                metrics.extend(description.components)
                metrics.extend(description.metrics)

        response = Description(components=components, metrics=metrics)
        debug(response)
        return response


    def measure(self, param: MeasureParams):

        print('measuring', param)

        for connector in self.servo.connectors:
            measure_func = getattr(connector, "measure", None)
            if callable(measure_func): # TODO: This should have a tighter contract (arity, etc)
                measurement = measure_func(metrics=param.metrics, control=param.control)
                debug(measurement)
                # Send MEASUREMENT event, param is dict of (metrics, annotations)
                # # TODO: Make this shit async...
                # response = client.post('servo', json=dict(event='MEASUREMENT', param=dict(metrics=metrics, annotations=annotations)))
                # response.raise_for_status()
                
                # command = response.json()
                # debug(command)

        sys.exit(2)

        # execute measurement driver and return result
        # rsp = run_driver(DFLT_MEASURE_DRIVER, args.app_id, req=param, progress_cb=partial(report_progress, 'MEASUREMENT', time.time()))
        # status = rsp.get('status', 'undefined')
        # if status != 'ok':
        #     raise Exception('Measurement driver failed with status "{}" and message "{}"'.format(
        #         status, rsp.get('message', 'undefined')))
        # metrics = rsp.get('metrics', {})
        # annotations = rsp.get('annotations', {})

        # if not metrics:
        #     raise Exception('Measurement driver returned no metrics')

        # print('measured ', metrics)

        # ret = dict(metrics=metrics)
        # if annotations:
        #     ret["annotations"] = annotations

        return ret

    def adjust(self, param):

        print('adjusting', param)

        # execute adjustment driver and return result
        # rsp = run_driver(DFLT_ADJUST_DRIVER, args.app_id, req=param, progress_cb=partial(report_progress, 'ADJUSTMENT', time.time()))
        # status = rsp.get('status', 'undefined')
        # if status == 'ok':
        #     print('adjusted ok')
        # else:
        #     raise DriverError('Adjustment driver failed with status "{}" and message:\n{}'.format(
        #         status, str(rsp.get('message', 'undefined'))), status=status, reason=rsp.get('reason', 'undefined'))

        return {}


    # --- Helpers -----------------------------------------------------------------
    
    def http_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self.headers)
    
    # TODO: Make private...
    def delay(self):
        if self.interactive:
            print('Press <Enter> to continue...', end='')
            sys.stdout.flush()
            sys.stdin.readline()
        # elif self.delay:
        #     time.sleep(args.delay)
        print()

    def request(self, event: Event, param, *, retries=None, backoff=True):
        # TODO: add retries/timeout
        # TODO: Pretty weird that this thing loops...
        # TODO: Factor retries into helper function

        # response = client.post('servo', json=event)
        # response.raise_for_status()
        # debug(response.json())

        '''
        Send request to cloud service. Retry if it fails to connect.
        Setting retries to None means retry forever; 0 means no retries, other
        integer value defines the number of retries
        TODO: implement backoff - currently ignored
        '''

        retry_delay = 20 # seconds
        if event == Event.WHATS_NEXT:
            retry_delay = 1 # quick first-time retry - workaround
        ev = EventRequest(event=event, param=param)
        while True:
            try:
                with self.http_client() as client:
                    response = client.post('servo', data=ev.json())
                    #response.raise_for_status()
            except httpx.NetworkError as e:
                exc = Exception('Server unavailable for event {} ({}: {}).'.format(event, type(e).__name__, str(e)))
            else:
                # check if server failed with 5xx response: display and set exc
                if response.is_error:
                    try:
                        rsp_msg = response.text
                    except Exception as e:
                        rsp_msg = "(unknown: failed to parse: {}: {})".format(type(e).__name__, str(e))
                    try:
                        rsp_msg = json.loads(rsp_msg)['message'] # extract salient message if json formatted
                    except Exception:
                        pass # leave raw text
                    exc = Exception('Server rejected request {} with status {}: {}.'.format(ev, response.status_code, rsp_msg))
                else:
                    try:
                        rsp_json = response.json()
                    except Exception as e:
                        try:
                            rsp_msg = rsp.text
                        except Exception as e:
                            rsp_msg = "(unknown: failed to parse: {}: {})".format(type(e).__name__, str(e))
                        exc = Exception('Server response is not valid json: {}.'.format(rsp_msg))
                    else:
                        exc = None

            if exc is None:
                break # success, return response

            # retry or fail
            if retries is not None:
                if retries > 0:
                    retries -= 1
                else:
                    exc = Exception('No more retries left, failed to send {}.'.format(event))
                    raise exc

            print(str(exc), 'Waiting {} seconds to retry...\n'.format(retry_delay))
            time.sleep(retry_delay)   # wait for cloud service to recover; TODO add progressive backoff to ~1 minute
            retry_delay = 20 # seconds
            continue
        
        # TODO: Parse this into a response class (Pydantic model)
        debug(response.text)
        # if args.verbose:
        #     print("RESPONSE:", rsp.text)
        return rsp_json

    def test_loop(self):
        self._stop_flag = True

        # WHATS_NEXT
        with self.http_client() as client:
            response = client.post('servo', json=dict(event='WHATS_NEXT', param=dict(agent=USER_AGENT)))
            response.raise_for_status()
            command = response.json()

            if command['cmd'] == 'DESCRIBE':
                # TODO: Send back a description via: event=DESCRIPTION, param=dict(descriptor=describe(), status='ok')
                v = json.loads('{"descriptor": {"application": {"components": {"web": {"settings": {"cpu": {"value": 0.475, "min": 0.1, "max": 0.8, "step": 0.125, "type": "range"}, "mem": {"value": 0.1, "min": 0.1, "max": 0.8, "step": 0.125, "type": "range"}, "replicas": {"value": 1, "min": 1, "max": 2, "step": 1, "type": "range"}}}}}, "measurement": {"metrics": {"throughput": {"unit": "rpm"}, "error_rate": {"unit": "percent"}, "latency_total": {"unit": "milliseconds"}, "latency_mean": {"unit": "milliseconds"}, "latency_50th": {"unit": "milliseconds"}, "latency_90th": {"unit": "milliseconds"}, "latency_95th": {"unit": "milliseconds"}, "latency_99th": {"unit": "milliseconds"}, "latency_max": {"unit": "milliseconds"}, "latency_min": {"unit": "milliseconds"}, "requests_total": {"unit": "count"}}}}, "status": "ok"}')
                response = client.post('servo', json=dict(event='DESCRIPTION', param=v))
                response.raise_for_status()
                debug(response.json())
            
                # WHATS_NEXT
                response = client.post('servo', json=dict(event='WHATS_NEXT', param=dict(agent=USER_AGENT)))
                response.raise_for_status()
                
                # MEASURE
                command = response.json()
                debug(command)
                if command['cmd'] == 'MEASURE':
                    debug("ASKED TO MEASURE!!!")

                    # TODO: This is quick and dirty...
                    # Dispatch measurements
                    for connector in self.connectors:
                        measure_func = getattr(connector, "measure", None)
                        if callable(measure_func): # TODO: This should have a tighter contract (arity, etc)
                            metrics, annotations = measure_func()
                            # Send MEASUREMENT event, param is dict of (metrics, annotations)
                            # TODO: Make this shit async...
                            response = client.post('servo', json=dict(event='MEASUREMENT', param=dict(metrics=metrics, annotations=annotations)))
                            response.raise_for_status()
                            
                            command = response.json()
                            debug(command)

    def exec_command(self):

        ev = ev_param = None
        cmd_json = self.request(Event.WHATS_NEXT, None)
        debug("Got response ", cmd_json)
        cmd_response = CommandResponse(**cmd_json)
        debug(cmd_response)

        if cmd_response.command == Command.DESCRIBE:            
            try:
                v = dict(descriptor=self.describe().opsani_dict(), status='ok')
                debug(v)
            except Exception as e:
                traceback.print_exc()
                # TODO: Use Status
                v = dict(status='failed', message=exc_format(e))

            # if args.verbose or args.interactive:
            #     print('DESCRIBE {}: will return {}.'.format(p, v), end=' ')
            debug(v)
            ev = Event.DESCRIPTION
            ev_param = v
            # TODO: Use EventRequest
        elif cmd_response.command == Command.MEASURE:
            try:
                v = self.measure(cmd_response.param)
            except Exception as e:
                traceback.print_exc()
                # TODO: Use Status
                v = dict(status='failed', message=exc_format(e))

            # if args.verbose or args.interactive:
            #     print('MEASURE {}: will return {}.'.format(p, v), end=' ')
            ev = Event.MEASUREMENT
            ev_param = v
            # TODO: Use EventRequest
        elif cmd_response.command == Command.ADJUST:
            try:
                # create a new dict based on p['state'] (with its top level key
                # 'application') which also includes a top-level 'control' key, and
                # pass this to adjust()
                # TODO: This needs to be modeled
                new_dict = cmd_response.param['state'].copy()
                new_dict['control'] = cmd_response.param.get('control', {})
                v = adjust(new_dict)
                if 'state' not in v: # if driver didn't return state, assume it is what was requested
                    v['state'] = cmd_response.param['state']
            except DriverError as e:
                traceback.print_exc()
                # TODO: Use Status
                v = dict(status=e.status, message=exc_format(e))
            except Exception as e:
                traceback.print_exc()
                # TODO: Use Status
                v = dict(status='failed', message=exc_format(e))
            # if args.verbose or args.interactive:
            #     print('ADJUST to {}: will return {}.'.format(p, v), end=' ')
            ev = Event.ADJUSTMENT
            ev_param = v
            # TODO: Use EventRequest
        elif cmd_response.command == Command.SLEEP:
            # if args.verbose or args.interactive:
            #     print('SLEEP {} sec.'.format(p), end=' ')
            if not self.interactive: # ignore sleep request when interactive - let user decide
                try:
                    duration = int(cmd_response.param['duration'])
                except Exception:
                    duration = 120 # seconds
                print('sleeping {} sec.'.format(duration))
                time.sleep(duration)
        else:
            raise Exception('unknown command "{}". Ignoring'.format(cmd_response.cmd))

        # delay() # nb: in interactive mode, will display prompt and wait for keypress
        if ev:
            self.request(ev, ev_param)

    # TODO: Needs a big refactor
    def report_progress(operation: str, ts_started: float, progress: int, time_remain: int = None, message: str = None, log: list = None) -> None:
        '''
        Report progress of driver operation
        Parameters:
        (Note that the first two parameters are usually passed as part of a closure, the remainder passed on callback)
            operation: event to send progress with (ADJUSTMENT or MEASUREMENT); if None/'', no progress will be reported
            ts_started: timestamp when operation started
            progress: percent completed [0-100] (if known; None otherwise)
            time_remain: time remaining [seconds] (if known; None otherwise)
            message: message to display (e.g., 'now adjusting compnent XX' or 'measurement: warming up'), or None
            log: list of zero or more strings reporting warnings or otherwise important items to send to cloud engine to log
        Returns nothing
        Raises exception in order to abort the operation (e.g., if told to do so
        by the cloud engine)
        '''
        def set_if(d, k, v):
            '''set dict key to value only if value is not None'''
            if v is not None: d[k] = v

        # skip progress if no operation specified (the protocol requires identifying the operation)
        if not operation:
            return

        # prepare parameters (progress field must be present, even if None)
        param = dict(progress=progress, runtime=int(time.time() - ts_started))
        set_if(param, 'time_remain', time_remain)
        set_if(param, 'message'    , message)
        set_if(param, 'log'        , log)

        # send event (limited retries)
        rsp = request(operation, param, retries=1, backoff=False)
        #if rsp.xxx:
        #   raise CancelDriverException('message')
        return

    def run(self) -> None:        
        self._stop_flag = False
        self._init_signal_handlers()   

        # announce
        print('Saying HELLO.', end=' ') # TODO: Logs...
        self.delay()
        self.request(Event.HELLO, dict(agent=USER_AGENT))

        while not self._stop_flag:
            try:
                self.exec_command()
            except Exception as e:
                traceback.print_exc()

        try:
            self.request(Event.GOODBYE, dict(reason=self.stop_flag), retries=3, backoff=False)
        except Exception as e:
            print('Warning: failed to send GOODBYE: {}. Exiting anyway'.format(str(e)))

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
        print('\n*** Servo stop requested by signal "{}". Sending GOODBYE to Optune cloud service (may retry 3 times)'.format(sig_name))

        # send GOODBYE event (best effort)
        try:
            request('GOODBYE', dict(reason=sig_name), retries=3, backoff=False)
        except Exception as e:
            print('Warning: failed to send GOODBYE: {}. Exiting anyway'.format(str(e)))

        sys.exit(0) # abort now


    def _graceful_stop_handler(self, sig_num, unused_frame):
        """handle signal for graceful termination - simply set a flag to have the main loop exit after the current operation is completed"""
        self._stop_flag = "exit"

    def _graceful_restart_handler(self, sig_num, unused_frame):
        """handle signal for restart - simply set a flag to have the main loop exit and restart the process after the current operation is completed"""
        self._stop_flag = "restart"
    
    def run_driver(self, driver, app, req=None, describe=False, progress_cb: typing.Callable[..., None]=None):
        '''
        Execute external driver to perform adjustment or measurement - or just get a descriptor
        Parameters:
            driver : path to driver executable
            app    : application ID to pass to driver
            req    : request input data (descriptor) to submit to driver's stdin (dict)
            describe: bool. If true, requesting a descriptor, not adjust/measure. Req must be None
            progress_cb: callback function to report progress; if it raises exception, try to abort driver's operation
                    Callback takes zero or more of the following named parameters (send None or omit to skip):
                    - progress: int, 0-100%
                    - time_remain: int, seconds
                    - message: str, progress/stage message
                    - log: list[str], messages to be logged
        '''

        if args.verbose:
            print('DRIVER REQUEST:', driver, req)

        assert not(bool(describe) ^ (req is None)), 'Driver {}: unexpected invocation: describe={}, req={} (exactly one should be used)'.format(driver, describe, req)

        # test only FIXME@@@
        if progress_cb:
            progress_cb(progress=0, message='starting driver')

        # construct command line
        cmd = [driver]
        if describe:
            if isinstance(describe, bool):
                cmd.append('--describe')    #FIXME: candidate for unified driver option name
            else:
                assert isinstance(describe, str)
                cmd.append(describe)    #FIXME: remove hack when driver options are unified
        cmd.append(app)

        # prepare stdin in-memory file if a request is provided
        if req is not None:
            stdin = json.dumps(req).encode("UTF-8")   # input descriptor -> json -> bytes
        else:
            stdin = b''         # no stdin

        # execute driver, providing request and capturing stdout/stderr
        proc = subprocess.Popen(cmd, bufsize=0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        stderr = [] # collect all stderr here
        rsp = {"status": "nodata"} # in case driver outputs nothing at all
        ri = [proc.stdout, proc.stderr]
        wi = [proc.stdin]
        ei = [proc.stdin, proc.stdout, proc.stderr]
        eof_stdout = False
        eof_stderr = False #
        while True:
            if eof_stdout and eof_stderr:
                if proc.poll() is not None: # process exited and no more data
                    break
                try:
                    proc.wait(DRIVER_EXIT_TIMEOUT) # don't wait forever
                except subprocess.TimeoutExpired:
                    print("WARNING: killed stuck child process ({})".format(repr(cmd)), file=sys.stderr)
                    proc.kill()
                break
            r, w, e = select.select(ri, wi, ei, DRIVER_IO_TIMEOUT)
            if not r and not w and not e: # timed out
                proc.terminate()
                raise Exception("timed out waiting for child process ({}) output".format(repr(cmd)))
            for h in r:
                if h is proc.stderr:
                    l = h.read(4096)
                    if not l:
                        eof_stderr = True
                        ri.remove(proc.stderr) # remove from select list (once in EOF, it will always stay 'readable')
                        continue
                    stderr.append(l)
                else: # h is proc.stdout
                    l = h.readline()
                    if not l:
                        eof_stdout = True
                        ri.remove(proc.stdout)
                        continue
                    stdout_line = l.strip().decode("UTF-8") # there will always be a complete line, driver writes one line at a time
                    if args.verbose:
                        print('DRIVER STDOUT:', stdout_line)
                    if not stdout_line:
                        continue # ignore blank lines (shouldn't be output, though)
                    try:
                        stdout = json.loads(stdout_line)
                    except Exception as e:
                        proc.terminate()
                        raise DriverOutputDecodingError(f"Failed decoding JSON stdout from '{driver}' driver: {stdout_line}") from e
                    if "progress" in stdout:
                        if progress_cb:
                            progress_cb(progress=stdout["progress"], message=stdout.get("message", None)) # FIXME stage/stageprogress ignored
                    else: # should be the last line only (TODO: check only one non-progress line is output)
                        rsp = stdout
            if w:
                l = min(getattr(select, 'PIPE_BUF', 512), len(stdin)) # write with select.PIPE_BUF bytes or less should not block
                if not l: # done sending stdin
                    proc.stdin.close()
                    wi = []
                    ei = [proc.stdout, proc.stderr]
                else:
                    proc.stdin.write(stdin[:l])
                    stdin = stdin[l:]
            # if e:

        rc = proc.returncode
        if args.verbose or rc != 0:
            print('DRIVER RETURNED:\n---stderr------------------\n{}\n----------------------\n'.format(b"\n".join(stderr).decode("UTF-8")))  # use accumulated stderr

        # (nb: stderr is discarded TODO: consider capturing into annotation, if allowed)
        # err = (b"\n".join(stderr)).decode("UTF-8")
        if args.verbose:
            print('DRIVER RESPONSE:', rsp)
    #LK: already printed        if err:
    #            print('DRIVER STDERR:', '\n---\n', err, '\n---\n')

        if rc != 0: # error, add verbose info to returned data
            if not rsp.get("status"): # NOTE if driver didn't return any json, status will be "nodata". Preferably, it should print structured data even on failures, so errors can be reported neatly.
                rsp["status"] = "failed"
            m = rsp.get("message", "")
            # if config[report_stderr]:
            rs = os.environ.get("OPTUNE_VERBOSE_STDERR", "all") # FIXME: default setting?
            if rs == "all":
                rsp["message"] = m + "\nstderr:\n" + (b"\n".join(stderr)).decode("UTF-8")
            elif rs == "minimal": # 1st two lines only
                rsp["message"] = m + "\nstderr:\n" + (b"\n".join(stderr[0:2])).decode("UTF-8")
            # else don't send any bit of stderr

        return rsp

def exc_format(e):
    if type(e) is Exception:  # if it's just an Exception
        return str(e) # print only the message but not the type
    return "{}: {}".format(type(e).__name__, str(e)) # print the exception type and message
