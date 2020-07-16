import sys
import os
import time
import yaml
import servo
from servo.connector import Connector, BaseConfiguration, License, Maturity, event, metadata, on_event
from servo.types import *
import durationpy
from pydantic import BaseModel, Extra, validator, HttpUrl, AnyHttpUrl
from typing import List, Tuple, Optional
from datetime import timedelta
import httpx
from devtools import pformat

DEFAULT_BASE_URL = "http://prometheus:9090"
API_PATH = "/api/v1"


class PrometheusMetric(Metric):
    query: str
    # period: timedelta
    period: int = 0

    # @classmethod
    # @validator("period", pre=True, always=True)
    # def validate_period(cls, value) -> timedelta:
    #     if isinstance(value, str):
    #         return durationpy.from_str(value)
    #     return 0


class PrometheusConfiguration(BaseConfiguration):
    base_url: AnyHttpUrl = DEFAULT_BASE_URL
    metrics: List[PrometheusMetric]

    @classmethod
    def generate(cls, **kwargs) -> 'PrometheusConfiguration':
        return cls(
            description="Update the base_url and metrics to match your Prometheus configuration",
            metrics=[
                PrometheusMetric('throughput', Unit.REQUESTS_PER_SECOND, query="rate(http_requests_total[1s])[3m]", period=0),
                PrometheusMetric('error_rate', Unit.PERCENTAGE, query="rate(errors)", period=0),
            ],
            **kwargs
        )

    @classmethod
    @validator("base_url")    
    def coerce_duration(cls, base_url) -> str:
        return base_url.rstrip("/")
    
    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}/"


@metadata(
    description="Prometheus Connector for Opsani",
    version="1.5.0",
    homepage="https://github.com/opsani/prometheus-connector",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class PrometheusConnector(Connector):
    config: PrometheusConfiguration

    @on_event()
    def check(self) -> CheckResult: # TODO: Turn this into a list
        start, end = time.time() - 600, time.time()
        for metric in self.config.metrics:
            m_values = self._query_prom(metric, start, end)
            self.logger.debug(
                "Initial value for metric %s: %s" % (metric.query, m_values)
            )

        return CheckResult(name="Check Prometheus", success=True, comment="!!!!!! All checks passed successfully.")
        # TODO: For check: run each query and look at results
        # TODO: This should become an awway with results of pass, fail, warn

    @on_event()
    def describe(self) -> Description:
        return Description(metrics=self.config.metrics)
        # TODO: Should this sample the current values?
    
    @on_event()
    def metrics(self) -> List[Metric]:
        return self.config.metrics
    
    @on_event()
    def measure(self, *, metrics: List[str] = None, control: Control = Control()) -> Measurement:
        
        # TODO: This becomes a pre-filter on the event -- probably a "before_event *" to bind to anything
        # execute pre_cmd, if any
        # pre_cmd = cfg.get('pre_cmd')  
        # if pre_cmd:
        #     self._run_command(pre_cmd, pre=True)

        

        # TODO: This warmup becomes a filter also
        # TODO: Before filters need to get the same input as the main action
        # try:
        #     warmup = int(self.input_data["control"]["warmup"])
        # except:
        #     warmup = 0

        # try:
        #     duration = int(self.input_data["control"]["duration"])
        # except:
        #     raise Exception('Control configuration is missing "duration"')

        # delay = int(self.input_data["control"].get("delay", 0))

        # TODO: This whole thing becomes a before filter
        start = time.time() + control.warmup
        # sleep
        debug(control)
        t_sleep = control.warmup + control.duration + control.delay
        self.logger.debug("Sleeping for %d seconds (%d warmup + %d duration)"
                   % (t_sleep, control.warmup, control.duration)
                   )
        time.sleep(t_sleep)

        metrics = self.gather_metrics(metrics, start, start + control.duration)
        annotations = {
            # 'prometheus_base_url': base_url,
        }

        return (metrics, annotations)
    
    def _get_metric_named(self, name: str) -> Optional[PrometheusMetric]:
        for metric in self.config.metrics:
            if metric.name == name:
                return metric
        
        raise ValueError(f"Unknown metric named '{name}' -- aborting")

    def gather_metrics(self, metric_names: List[str], start, end):
        metrics = {}
        for metric_name in metric_names:
            metric = _get_metric_named(metric_name) # TODO: This can be tighter
            m_values = self._query_prom(metric, start, end,)
            self.logger.debug(
                "Initial value for metric %s: %s" % (metric.query, m_values)
            )

            metrics.update(
                {metric_name: {"values": m_values, "annotation": metric.query, }}
            )
        # TODO: This is gonna return time series
        debug(metrics)
        return metrics
            
    def _query_prom(self, metric: PrometheusMetric, start: float, end: float) -> List[dict]:
        debug(start, end)
        # TODO" Tigthen this up...
        url = "%squery_range?query=%s&start=%s&end=%s&step=%i" % (
            self.config.api_url,
            metric.query,
            start,
            end,
            60 # TODO: metric.period
        )

        self.logger.info(f"Getting url: {url}")
        with self.api_client() as client:
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                self.logger.exception(
                    f"HTTP error encountered during GET {url}"
                )
                raise error

        data = response.json()
        self.logger.info(f"Got response data: {data}")

        # TODO: Turn this assert into an exception
        assert ("status" in data and data["status"] == "success"), "Prometheus server did not return status success"

        # TODO: This is total insanity that needs t be modeled before we all go crazy
        # TODO: Marshall this data into a Pydantic model that it makes sense...

        insts = []
        for i in data["data"]["result"]:
            metric = i["metric"].copy()
            if "__name__" in metric:
                del metric["__name__"]
            metric_id = "   ".join(
                map(lambda m: ":".join(m), sorted(metric.items(), key=lambda m: m[0]))
            )
            values = []
            for item in i["values"]:
                try:
                    if "." in item[1]:
                        d = float(item[1])
                    elif item[1] == 'NaN':
                        continue
                    else:
                        d = int(item[1])
                except ValueError:
                    continue
                values.append((item[0], d))
            insts.append(dict(id=metric_id, data=list(values)))

        debug(insts)
        return insts or []
    