import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_type_hints, Union, Set, Tuple

import httpx
import typer
import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    FilePath,
    HttpUrl,
    ValidationError,
    constr,
    root_validator,
    validator,
)
import durationpy
import servo
from servo.connector import Connector, ConnectorCLI, ConnectorSettings, License, Maturity
import subprocess
import time
from threading import Timer
from datetime import datetime, timedelta
import copy
import logging
from loguru import logger
import sys
from devtools import pformat
from servo.metrics import Metric, Unit, Measurement, Numeric, Control, TimeSeries, Description

# logger.add({ "sink": sys.stdout, "colorize": True, "level": logging.DEBUG })

# TODO: This should really come down to `from servo import Connector, ConnectorSettings`

###
### Vegeta 

METRICS = [
    Metric('throughput', Unit.REQUESTS_PER_MINUTE),
    Metric('error_rate', Unit.PERCENTAGE),
    Metric('latency_total', Unit.MILLISECONDS),
    Metric('latency_mean', Unit.MILLISECONDS),
    Metric('latency_50th', Unit.MILLISECONDS),
    Metric('latency_90th', Unit.MILLISECONDS),
    Metric('latency_95th', Unit.MILLISECONDS),
    Metric('latency_99th', Unit.MILLISECONDS),
    Metric('latency_max', Unit.MILLISECONDS),
    Metric('latency_min', Unit.MILLISECONDS),
]

class TargetFormat(str, Enum):
    http = "http"
    json = "json"

    def __str__(self):
        return self.value

class Latencies(BaseModel):
    total: int
    mean: int
    p50: int = Field(alias='50th')
    p90: int = Field(alias='90th')
    p95: int = Field(alias='95th')
    p99: int = Field(alias='99th')
    max: int
    min: int

    @validator('*')
    def convert_nanoseconds_to_milliseconds(cls, latency):
        # Convert Nanonsecond -> Millisecond
        return (latency * 0.000001) if latency is not None else -1

class Bytes(BaseModel):
    total: int
    mean: float

class VegetaReport(BaseModel):
    latencies: Latencies
    bytes_in: Bytes
    bytes_out: Bytes
    earliest: datetime
    latest: datetime
    end: datetime
    duration: timedelta
    wait: timedelta
    requests: int
    rate: float
    throughput: float
    success: float
    error_rate: float = None
    status_codes: Dict[str, int]
    errors: List[str]

    @validator('throughput')
    def convert_throughput_to_rpm(cls, throughput):
        return throughput * 60
    
    @validator('error_rate', always=True, pre=True)
    def calculate_error_rate_from_success(cls, v, values):
        success_rate = values['success']
        return 100 - (success_rate * 100) # Fraction of success inverted into % of error

    def get(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        elif '.' in key:
            parent_key, child_key = key.split('.', 2)
            child = self.get(parent_key).dict(by_alias=True)
            return child[child_key]
        else:
            raise ValueError(f"unknown key '{key}'")

class VegetaSettings(ConnectorSettings):
    """
    Configuration of the Vegeta connector
    """

    rate: str = Field(
        description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.",
    )
    duration: str = Field(
        description="Specifies the amount of time to issue requests to the targets.",
    )
    format: TargetFormat = Field(
        "http",
        description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.",
    )
    target: Optional[str] = Field(
        description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin."
    )
    targets: Optional[FilePath] = Field(
        description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk."
    )
    connections: int = Field(
        10000,
        description="Specifies the maximum number of idle open connections per target host.",
    )
    workers: int = Field(
        10,
        description="Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.",
    )
    max_workers: int = Field(
        18446744073709551615,
        alias="max-workers",
        description="The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.",
        env="",
    )
    max_body: int = Field(
        -1,
        alias="max-body",
        description="Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.",
        env="",
    )
    http2: bool = Field(
        True,
        description="Specifies whether to enable HTTP/2 requests to servers which support it.",
    )
    keepalive: bool = Field(
        True,
        description="Specifies whether to reuse TCP connections between HTTP requests.",
    )
    insecure: bool = Field(
        False,
        description="Specifies whether to ignore invalid server TLS certificates.",
    )

    @root_validator()
    @classmethod
    def validate_target(cls, values):
        target, targets = values.get("target"), values.get("targets")
        if target is None and targets is None:
            raise ValueError("target or targets must be configured")

        if target is not None and targets is not None:
            raise ValueError("target and targets cannot both be configured")

        return values

    @root_validator()
    @classmethod
    def validate_target_format(cls, values):
        target, targets = values.get("target"), values.get("targets")

        # Validate JSON target formats
        if target is not None and values.get("format") == TargetFormat.json:
            try:
                json.loads(target)
            except Exception as e:
                raise ValueError("the target is not valid JSON") from e

        if targets is not None and values.get("format") == TargetFormat.json:
            try:
                json.load(open(targets))
            except Exception as e:
                raise ValueError("the targets file is not valid JSON") from e

        # TODO: Add validation of JSON with JSON Schema (https://github.com/tsenart/vegeta/blob/master/lib/target.schema.json)
        # and HTTP format
        return values

    @validator("rate")
    @classmethod
    def validate_rate(cls, v):
        assert isinstance(
            v, (int, str)
        ), "rate must be an integer or a rate descriptor string"

        # Integer rates
        if isinstance(v, int) or v.isnumeric():
            return str(v)

        # Check for hits/interval
        components = v.split("/")
        assert len(components) == 2, "rate strings are of the form hits/interval"

        hits = components[0]
        duration = components[1]
        assert hits.isnumeric(), "rate must have an integer hits component"

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(duration)
        except Exception as e:
            raise ValueError(f"Invalid duration '{duration}' in rate '{v}'") from e

        return v

    @validator("duration")
    @classmethod
    def validate_duration(cls, v):
        assert isinstance(
            v, (int, str)
        ), "duration must be an integer or a duration descriptor string"

        if v == "0" or v == 0:
            return v

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(v)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    class Config:
        json_encoders = {TargetFormat: lambda t: t.value()}

## TODO: Model all this crap
# DEFAULT_DURATION = 120
# DEFAULT_WARMUP = 0
# DEFAULT_DELAY = 0
REPORTING_INTERVAL = 2



@servo.connector.metadata(
    description="Vegeta load testing connector",
    version="0.5.0",
    homepage="https://github.com/opsani/vegeta-connector",
    license=License.APACHE2,
    maturity=Maturity.STABLE,
)
class VegetaConnector(Connector):
    settings: VegetaSettings

    time_series_metrics = {}
    _vegeta_reports: List[VegetaReport] = []
    warmup_until_timestamp: datetime = None
    proc: Any

    def cli(self) -> ConnectorCLI:
        """Returns a Typer CLI for interacting with this connector"""
        cli = ConnectorCLI(self, help="Load generation with Vegeta")

        @cli.command()
        def loadgen():
            """
            Run an adhoc load generation
            """
            self.measure()

        return cli
    
    def print_progress(self, str, *args, **kwargs):
        # debug(str)
        logger.debug(str)
        # debug(str, *args, **kwargs)
    
    def debug(self, str, *args, **kwargs):
        # debug(str, *args, **kwargs)
        logger.debug(str)
    
    def format_metric(self, metric) -> str:
        return f"{metric['value']:.2f}{metric['unit']}"

    def format_matrics(self, metrics: dict) -> str:
        throughput = self.format_metric(metrics['throughput'])
        error_rate = self.format_metric(metrics['error_rate'])
        latency_50th = self.format_metric(metrics['latency_50th'])
        latency_90th = self.format_metric(metrics['latency_90th'])
        latency_95th = self.format_metric(metrics['latency_95th'])
        latency_99th = self.format_metric(metrics['latency_99th'])
        return f'Vegeta attacking "{self.settings.target}" @ {self.settings.rate}: ~{throughput} ({error_rate} errors) [latencies: 50th={latency_50th}, 90th={latency_90th}, 95th={latency_95th}, 99th={latency_99th}]'


    # TODO: Add a decorator of some kind
    def describe(self) -> Description:
        return Description(metrics=METRICS, components=[])
    
    def measure(self, *, metrics: List[str] = None, control: Control = Control()) -> Measurement:
        # Handle delay (if any)
        # TODO: Make the delay/warm-up reusable...
        if control.delay > 0:
            self.progress = 0
            self.print_progress(f'DELAY: sleeping {control.delay} seconds')
            time.sleep(control.delay)
        self.warmup_until_timestamp = datetime.now() + timedelta(seconds=control.warmup)

        # Run the load test
        number_of_urls = 1 if self.settings.target else _number_of_lines_in_file(self.settings.targets)
        summary = f"Loading {number_of_urls} URL(s) for {self.settings.duration} (delay of {control.delay}, warmup of {control.warmup}) at a rate of {self.settings.rate}"
        self.print_progress(summary)
        exit_code, command = self._run_vegeta()
        self.print_progress(f"Producing time series metrics from {len(self._vegeta_reports)} measurements")
        measurements = self._time_series_measurements_from_vegeta_reports() if self._vegeta_reports else {}        
        response = Measurement(readings=measurements, annotations={
            'load_profile': summary,
        })
        self.print_progress(f"Reporting time series metrics {pformat(measurements)} and annotations {pformat(response.annotations)}")
        return response

    def _run_vegeta(self):
        prog_coefficient = 1.0
        prog_start = 0.0
        exit_code = None

        # construct and run Vegeta command
        # TODO: need a utility for stringifying lists/dicts for subprocess fun
        vegeta_attack_args = [
            'vegeta', 'attack',
            '-rate', self.settings.rate, 
            '-duration', self.settings.duration, 
            '-targets', self.settings.targets if self.settings.targets else 'stdin',
            '-format', str(self.settings.format),
            '-connections', str(self.settings.connections),
            '-workers', str(self.settings.workers),
            '-max-workers', str(self.settings.max_workers),
            '-http2', str(self.settings.http2),
            '-keepalive', str(self.settings.keepalive),
            '-insecure', str(self.settings.insecure),
            '-max-body', str(self.settings.max_body),
        ]

        vegeta_report_args = [
            'vegeta', 'report', 
            '-type', 'json',
            '-every', str(REPORTING_INTERVAL) + 's'
        ]

        echo_args = ['echo', f"{self.settings.target}"]
        echo_cmd = f'echo "{self.settings.target}" | ' if self.settings.target else ''
        vegeta_cmd = echo_cmd + ' '.join(vegeta_attack_args) + ' | ' + ' '.join(vegeta_report_args)
        self.debug("Running Vegeta:", vegeta_cmd)
        self.print_progress(f"Vegeta started: {vegeta_cmd}")

        # If we are loading a single target, we need to connect an echo proc into Vegeta stdin
        echo_proc_stdout = subprocess.Popen(echo_args, stdout=subprocess.PIPE, encoding="utf8").stdout if self.settings.target else None
        vegeta_attack_proc = subprocess.Popen(vegeta_attack_args, stdin=echo_proc_stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')

        # Pipe the output from the attack process into the reporting process
        self.proc = subprocess.Popen(vegeta_report_args, stdin=vegeta_attack_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')

        # start progress for time limited vegeta command: update every 5 seconds -
        # it is printed by default every 30 seconds
        duration_in_seconds = _seconds_from_duration_str(self.settings.duration) # TODO: Replace with library, move to stdlib
        started_at = time.time()
        # timer = repeatingTimer(REPORTING_INTERVAL, self._update_timed_progress, started_at, duration_in_seconds,
        #                     prog_start, prog_coefficient)
        # timer.start()

        # loop and poll our process pipe to gather report data        
        try:
            # compile a regex to strip the ANSI escape sequences from the report output (clear screen)
            ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
            while True:
                output = self.proc.stdout.readline()
                if output:                    
                    json_report = ansi_escape.sub('', output)
                    debug(json.loads(json_report))
                    vegeta_report = VegetaReport(**json.loads(json_report))
                    debug(vegeta_report)                    
                    # metrics = self._metrics_from_vegeta_report(vegeta_report)
                    if datetime.now() > self.warmup_until_timestamp:
                        # timestamp = int(time.time())
                        # self.time_series_metrics[timestamp] = metrics
                        self._vegeta_reports.append(vegeta_report)
                        # self.print_progress(f"Vegeta metrics aggregated: {metrics}")
                        # debug(self.format_matrics(metrics))
                    else:
                        self.print_progress(f"Vegeta metrics excluded (warmup in effect): {metrics}")
                if self.proc.poll() is not None:
                    # Child process has exited, stop polling
                    break
            exit_code = self.proc.returncode
            if exit_code != 0:
                error = self.proc.stderr.readline()
                self.debug("Vegeta error:", error)
            self.debug("Vegeta exited with exit code:", exit_code)
            self.print_progress(f"Vegeta exited with exit code: {exit_code}")
        finally:
            pass
            # timer.cancel()
        return exit_code, vegeta_cmd

    # helper:  take the time series metrics gathered during the attack and map them into OCO format
    def _time_series_measurements_from_vegeta_reports(self):
        time_series_measurements = []

        for metric in METRICS:
            if metric.name in ('throughput', 'error_rate',):
                key = metric.name
            elif metric.name.startswith('latency_'):
                key = 'latencies' + '.' + metric.name.replace('latency_', '')
            else:
                raise NameError(f'Unexpected metric name "{metric.name}"')
            
            values: List[Tuple(datetime, Numeric)] = []
            for report in self._vegeta_reports:
                values.append((report.end, report.get(key),))
            
            time_series_measurements.append(TimeSeries(metric=metric, values=values))
        
        debug(time_series_measurements)
        sys.exit(2)

        # Initialize values storage
        # for metric_name, data in metrics.items():
        #     data['values'] = [ { 'id': str(int(time.time())), 'data': [] } ]

        # # Fill the values with arrays of [timestamp, value] sampled from the reports
        # for timestamp, report in self.time_series_metrics.items():
        #     for metric_name, data in report.items():
        #         value = data['value']
        #         metrics[metric_name]['values'][0]['data'].append([timestamp, value])
        
        return metrics

    # helper:  update timer based progress
    # TODO: Moves into superclass
    def _update_timed_progress(self, t_start, t_limit, prog_start, prog_coefficient):
        # When duration is 0, Vegeta is attacking forever
        if t_limit:
            prog = min(100.0, 100.0 * (time.time() - t_start) / t_limit)
            self.progress = min(100, int((prog_coefficient * prog) + prog_start))


def _number_of_lines_in_file(filename):
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            count += 1
    return count

# Parses a Golang duration string into seconds
# TODO: This may be replacable with the library
# TODO: Factor into parent class/utility library
def _seconds_from_duration_str(duration_value):
    if isinstance(duration_value, (int, float)):
        return int(duration_value)
    elif duration_value.strip() == '0':
        return 0
    hours, minutes, seconds = 0, 0, 0
    for component in re.findall('\d+[hms]', duration_value):
        time = component[:-1]
        unit = component[-1]
        if unit == 'h': hours = int(time)
        if unit == 'm': minutes = int(time)
        if unit == 's': seconds = int(time)
    total_seconds = (hours * 60 * 60) + (minutes * 60) + seconds
    return total_seconds