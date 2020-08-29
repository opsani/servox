from __future__ import annotations
import asyncio
from functools import reduce
from datetime import datetime, timedelta
from typing import List, Optional

import httpx
import httpcore._exceptions
from pydantic import BaseModel, AnyHttpUrl, validator

from servo import (
    BaseConfiguration,
    BaseConnector,
    Control,
    Check,
    Description,
    Duration,
    Filter,
    HaltOnFailed,
    License,
    Maturity,
    Measurement,
    Metric,
    Unit,
    metadata,
    on_event,
    TimeSeries,
    DurationProgress,
)
from servo.checks import create_checks_from_iterable
from servo.utilities import join_to_series


DEFAULT_BASE_URL = "http://prometheus:9090"
API_PATH = "/api/v1"


class PrometheusMetric(Metric):
    query: str
    step: Duration = "1m"

    def __check__(self) -> Check:
        return Check(
            name=f"Check {self.name}",
            description=f"Run Prometheus query \"{self.query}\""
        )


class PrometheusConfiguration(BaseConfiguration):
    base_url: AnyHttpUrl = DEFAULT_BASE_URL
    metrics: List[PrometheusMetric]

    @classmethod
    def generate(cls, **kwargs) -> "PrometheusConfiguration":
        return cls(
            description="Update the base_url and metrics to match your Prometheus configuration",
            metrics=[
                PrometheusMetric(
                    "throughput",
                    Unit.REQUESTS_PER_SECOND,
                    query="rate(http_requests_total[1s])[3m]",
                    step="1m",
                ),
                PrometheusMetric(
                    "error_rate", Unit.PERCENTAGE, query="rate(errors)", step="1m"
                ),
            ],
            **kwargs,
        )
    
    @validator("base_url", allow_reuse=True)
    @classmethod
    def rstrip_base_url(cls, base_url):
        return base_url.rstrip("/")

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"


class PrometheusRequest(BaseModel):
    base_url: AnyHttpUrl
    metric: PrometheusMetric
    start: datetime
    end: datetime
    
    @property
    def query(self) -> str:
        return self.metric.query
    
    @property
    def step(self) -> Duration:
        return self.metric.step 

    @property
    def url(self) -> str:
        return "".join(self.base_url.rstrip("/") + 
            "/query_range" +
            f"?query={self.query}" +
            f"&start={self.start.timestamp()}" +
            f"&end={self.end.timestamp()}" +
            f"&step={self.metric.step}"
        )

@metadata(
    description="Prometheus Connector for Opsani",
    version="1.5.0",
    homepage="https://github.com/opsani/prometheus-connector",
    license=License.APACHE2,
    maturity=Maturity.EXPERIMENTAL,
)
class PrometheusConnector(BaseConnector):
    config: PrometheusConfiguration

    @on_event()
    async def check(self, 
        filter_: Optional[Filter] = None, 
        halt_on: HaltOnFailed = HaltOnFailed.requirement
    ) -> List[Check]:
        start, end = datetime.now() - timedelta(minutes=10), datetime.now()        
        async def check_query(metric: PrometheusMetric) -> str:
            result = await self._query_prom(metric, start, end)
            return f"returned {len(result)} TimeSeries readings"

        # wrap all queries into checks and verify that they work
        PrometheusChecks = create_checks_from_iterable(check_query, self.config.metrics)
        return await PrometheusChecks.run(self.config, filter_, halt_on=halt_on)

    @on_event()
    def describe(self) -> Description:
        return Description(metrics=self.config.metrics)

    @property
    @on_event()
    def metrics(self) -> List[Metric]:
        return self.config.metrics

    @on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: Control = Control()
    ) -> Measurement:
        if metrics:
            metrics__ = list(filter(lambda m: m.name in metrics, self.metrics))
        else:
            metrics__ = self.metrics
        measuring_names = list(map(lambda m: m.name, metrics__))
        self.logger.info(f"Starting measurement of {len(metrics__)} metrics: {join_to_series(measuring_names)}")

        start = datetime.now() + control.warmup
        end = start + control.duration

        sleep_duration = Duration(control.warmup + control.duration)
        self.logger.info(
            f"Waiting {sleep_duration} during metrics collection ({control.warmup} warmup + {control.duration} duration)..."
        )

        progress = DurationProgress(sleep_duration)
        notifier = lambda p: self.logger.info(p.annotate(f"waiting {sleep_duration} during metrics collection...", False), progress=p.progress)
        await progress.watch(notifier)
        self.logger.info(f"Done waiting {sleep_duration} for metrics collection, resuming optimization.")

        # Capture the measurements
        self.logger.info(f"Querying Prometheus for {len(metrics__)} metrics...")
        readings = await asyncio.gather(
            *list(map(lambda m: self._query_prom(m, start, end), metrics__))
        )
        all_readings = reduce(lambda x, y: x+y, readings) if readings else []
        measurement = Measurement(readings=all_readings)
        return measurement

    async def _query_prom(
        self, metric: PrometheusMetric, start: datetime, end: datetime
    ) -> List[TimeSeries]:
        prometheus_request = PrometheusRequest(base_url=self.config.api_url, metric=metric, start=start, end=end)
        
        self.logger.trace(f"Querying Prometheus (`{metric.query}`): {prometheus_request.url}")
        async with self.api_client() as client:
            try:
                response = await client.get(prometheus_request.url)
                response.raise_for_status()
            except (httpx.HTTPError, httpcore._exceptions.ReadTimeout, httpcore._exceptions.ConnectError) as error:                
                self.logger.trace(f"HTTP error encountered during GET {prometheus_request.url}: {error}")
                raise

        data = response.json()
        self.logger.trace(f"Got response data for metric {metric}: {data}")

        if "status" not in data or data["status"] != "success":
            return []
    
        readings = []
        for result_dict in data["data"]["result"]:
            m_ = result_dict["metric"].copy()
            # NOTE: Unpack "metrics" subdict and pack into a string
            if "__name__" in m_:
                del m_["__name__"]
            instance = m_.get("instance")
            job = m_.get("job")
            annotation = " ".join(map(lambda m: "=".join(m), sorted(m_.items(), key=lambda m: m[0])))
            readings.append(
                TimeSeries(
                    metric=metric,
                    annotation=annotation,
                    values=result_dict["values"],
                    id=f"{{instance={instance},job={job}}}",
                    metadata=dict(instance=instance, job=job)
                )
            )
        return readings
