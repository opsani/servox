import signal
import sys
import time
from enum import Enum
from logging import Logger
from typing import Any, Dict, List, Optional, Union

import backoff
import httpx
from devtools import pformat
from pydantic import BaseModel, Field, parse_obj_as

from servo.connector import USER_AGENT, Optimizer
from servo.servo import BaseServoConfiguration, Events, Servo
from servo.types import Control, Description, Measurement
from servo.utilities import SignalHandler


class APICommand(str, Enum):
    DESCRIBE = "DESCRIBE"
    MEASURE = "MEASURE"
    ADJUST = "ADJUST"
    SLEEP = "SLEEP"


class APIEvent(str, Enum):
    HELLO = "HELLO"
    GOODBYE = "GOODBYE"
    DESCRIPTION = "DESCRIPTION"
    WHATS_NEXT = "WHATS_NEXT"
    ADJUSTMENT = "ADJUSTMENT"
    MEASUREMENT = "MEASUREMENT"


class APIRequest(BaseModel):
    event: APIEvent
    param: Optional[Dict[str, Any]]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            APIEvent: lambda v: str(v),
        }


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


class CommandResponse(BaseModel):
    command: APICommand = Field(alias="cmd",)
    param: Optional[
        Union[MeasureParams, Dict[str, Any]]
    ]  # TODO: Switch to a union of supported types

    class Config:
        json_encoders = {
            APICommand: lambda v: str(v),
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
    base_url: str
    headers: Dict[str, str]
    _stop_flag: bool
    _signal_handler: SignalHandler

    def __init__(self, servo: Servo, *, interactive: bool = False, **kwargs) -> None:
        self.servo = servo
        self.interactive = interactive
        super().__init__()

    @property
    def optimizer(self) -> Optimizer:
        return self.servo.optimizer

    @property
    def configuration(self) -> BaseServoConfiguration:
        return self.servo.configuration

    @property
    def logger(self) -> Logger:
        return self.servo.logger

    def describe(self) -> Description:
        self.logger.info("Describing...")

        # TODO: This message dispatch should go through a driver for in-process vs. subprocess
        aggregate_description = Description.construct()
        results: List[EventResult] = self.servo.dispatch_event(Events.DESCRIBE)
        for result in results:
            description = result.value
            aggregate_description.components.extend(description.components)
            aggregate_description.metrics.extend(description.metrics)

        return aggregate_description

    def measure(self, param: MeasureParams) -> Measurement:
        self.logger.info(f"Measuring... [metrics={', '.join(param.metrics)}]")
        self.logger.trace(pformat(param))

        aggregate_measurement = Measurement.construct()
        results: List[EventResult] = self.servo.dispatch_event(
            Events.MEASURE, metrics=param.metrics, control=param.control
        )
        for result in results:
            measurement = result.value
            aggregate_measurement.readings.extend(measurement.readings)
            aggregate_measurement.annotations.update(measurement.annotations)

        return aggregate_measurement

    def adjust(self, param) -> dict:
        self.logger.info("Adjusting...")
        self.logger.trace(pformat(param))

        results: List[EventResult] = self.servo.dispatch_event(Events.ADJUST, param)
        for result in results:
            # TODO: Should be modeled
            adjustment = result.value
            status = adjustment.get("status", "undefined")

            if status == "ok":
                self.logger.info(f"{result.connector.name} - Adjustment completed")
                return adjustment
            else:
                raise ConnectorError(
                    'Adjustment driver failed with status "{}" and message:\n{}'.format(
                        status, str(adjustment.get("message", "undefined"))
                    ),
                    status=status,
                    reason=adjustment.get("reason", "undefined"),
                )

        # TODO: Model a response class
        return {}

    # --- Helpers -----------------------------------------------------------------

    def delay(self):
        if self.interactive:
            print("Press <Enter> to continue...", end="")
            sys.stdout.flush()
            sys.stdin.readline()
        elif self.delay:
            time.sleep(1.0)

    @backoff.on_exception(backoff.expo, (httpx.HTTPError), max_time=180, max_tries=12)
    def post_event(self, event: APIEvent, param) -> Union[CommandResponse, Status]:
        """
        Send request to cloud service. Retry if it fails to connect.
        """

        event_request = APIRequest(event=event, param=param)
        with self.servo.api_client() as client:
            try:
                response = client.post("servo", data=event_request.json())
                response.raise_for_status()
            except httpx.HTTPError as error:
                self.logger.exception(
                    f"HTTP error encountered while posting {event.value} event"
                )
                self.logger.trace(pformat(event_request))
                raise error

        return parse_obj_as(Union[CommandResponse, Status], response.json())

    def exec_command(self):
        cmd_response = self.post_event(APIEvent.WHATS_NEXT, None)
        self.logger.debug(f"What's Next? => {cmd_response.command}")
        self.logger.trace(pformat(cmd_response))

        try:
            if cmd_response.command == APICommand.DESCRIBE:
                description = self.describe()
                self.logger.info(
                    f"Described: {len(description.components)} components, {len(description.metrics)} metrics"
                )
                self.logger.trace(pformat(description))
                param = dict(descriptor=description.opsani_dict(), status="ok")
                self.post_event(APIEvent.DESCRIPTION, param)

            elif cmd_response.command == APICommand.MEASURE:
                measurement = self.measure(cmd_response.param)
                self.logger.info(
                    f"Measured: {len(measurement.readings)} readings, {len(measurement.annotations)} annotations"
                )
                self.logger.trace(pformat(measurement))
                param = measurement.opsani_dict()
                self.post_event(APIEvent.MEASUREMENT, param)

            elif cmd_response.command == APICommand.ADJUST:
                # # TODO: This needs to be modeled
                # oc"{'cmd': 'ADJUST', 'param': {'state': {'application': {'components': {'web': {'settings': {'cpu': {'value': 0.225}, 'mem': {'value': 0.1}}}}}}, 'control': {}}}"

                # TODO: Why do we do this nonsense??
                # create a new dict based on p['state'] (with its top level key
                # 'application') which also includes a top-level 'control' key, and
                # pass this to adjust()
                new_dict = cmd_response.param["state"].copy()
                new_dict["control"] = cmd_response.param.get("control", {})
                adjustment = self.adjust(new_dict)

                # TODO: What works like this and why?
                if (
                    "state" not in adjustment
                ):  # if driver didn't return state, assume it is what was requested
                    adjustment["state"] = cmd_response.param["state"]

                components_dict = adjustment["state"]["application"]["components"]
                components_count = len(components_dict)
                settings_count = sum(
                    len(components_dict[component]["settings"])
                    for component in components_dict
                )
                self.logger.info(
                    f"Adjusted: {components_count} components, {settings_count} settings"
                )

                self.post_event(APIEvent.ADJUSTMENT, adjustment)

            elif cmd_response.command == APICommand.SLEEP:
                if (
                    not self.interactive
                ):  # ignore sleep request when interactive - let user decide
                    # TODO: Model this
                    duration = int(cmd_response.param.get("duration", 120))
                    self.logger.info(f"Sleeping {duration} sec.")
                    time.sleep(duration)

            else:
                raise ValueError(f"Unknown command '{cmd_response.command.value}'")

        except Exception as error:
            self.logger.exception(f"{cmd_response.command} command failed!")
            param = dict(status="failed", message=_exc_format(error))
            sys.exit(2)
            self.post_event(_event_for_command(cmd_response.command), param)

    def run(self) -> None:
        self._stop_flag = False
        self._signal_handler = SignalHandler(
            stop_callback=self._stop_callback,
            restart_callback=self._restart_callback,
            terminate_callback=self._terminate_callback,
        )

        self.logger.info(
            f"Servo starting with {len(self.servo.connectors)} active connectors [{self.optimizer.id} @ {self.optimizer.base_url}]"
        )

        # announce
        self.logger.info("Saying HELLO.", end=" ")
        self.delay()
        self.post_event(APIEvent.HELLO, dict(agent=USER_AGENT))

        while not self._stop_flag:
            try:
                self.exec_command()
            except Exception:
                self.logger.exception("Exception encountered while executing command")

        try:
            self.post_event(APIEvent.GOODBYE, dict(reason=self.stop_flag))
        except Exception as e:
            self.logger.exception(
                f"Warning: failed to send GOODBYE: {e}. Exiting anyway"
            )

    def _stop_callback(self, sig_num: int) -> None:
        self._stop_flag = "exit"

    def _restart_callback(self, sig_num: int) -> None:
        self._stop_flag = "restart"

    def _terminate_callback(self, sig_num: int) -> None:
        # determine signal name (best effort)
        try:
            sig_name = signal.Signals(sig_num).name
        except ValueError:
            sig_name = f"signal #{sig_num}"

        # log signal
        self.logger.info(
            f'*** Servo stop requested by signal "{sig_name}". Sending GOODBYE'
        )

        # send GOODBYE event (best effort)
        try:
            self.post_event(APIEvent.GOODBYE, dict(reason=sig_name))
        except Exception as e:
            self.logger.exception(
                f"Warning: failed to send GOODBYE: {e}. Exiting anyway"
            )


def _event_for_command(command: APICommand) -> Optional[APIEvent]:
    if cmd_response.command == APICommand.DESCRIBE:
        return APIEvent.DESCRIPTION
    elif cmd_response.command == APICommand.MEASURE:
        return APIEvent.MEASUREMENT
    elif cmd_response.command == APICommand.ADJUST:
        return APIEvent.ADJUSTMENT
    else:
        return None


def _exc_format(e):
    if type(e) is Exception:  # if it's just an Exception
        return str(e)  # print only the message but not the type
    return "{type(e).__name__}: {e}"
