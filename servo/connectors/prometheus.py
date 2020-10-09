import asyncio
import datetime
import functools
from typing import Dict, Iterable, List, Optional, Tuple

import httpcore._exceptions
import httpx
import pydantic

import servo

DEFAULT_BASE_URL = "http://prometheus:9090"
API_PATH = "/api/v1"


class PrometheusMetric(servo.Metric):
    """PrometheusMetric objects describe metrics that can be measured by querying
    Prometheus.
    """

    query: str
    """A PromQL query that returns the value of the target metric.

    For details on PromQL, see the [Prometheus
    Querying](https://prometheus.io/docs/prometheus/latest/querying/basics/)
    documentation.
    """

    step: servo.types.Duration = "1m"
    """The resolution of the query.

    The step resolution determines the number of data points captured across a
    query range.
    """

    def __check__(self) -> servo.Check:
        return servo.Check(
            name=f"Check {self.name}",
            description=f'Run Prometheus query "{self.query}"',
        )


class PrometheusTarget(pydantic.BaseModel):
    """PrometheusTarget objects describe targets that are scraped by Prometheus jobs."""

    labels: Optional[Dict[str, str]]


class PrometheusConfiguration(servo.BaseConfiguration):
    """PrometheusConfiguration objects describe how PrometheusConnector objects
    capture measurements from the Prometheus metrics server.
    """

    base_url: pydantic.AnyHttpUrl = DEFAULT_BASE_URL
    """The base URL for accessing the Prometheus metrics API.

    The URL must point to the root of the Prometheus deployment. Resource paths
    are computed as necessary for API requests.
    """

    metrics: List[PrometheusMetric]
    """The metrics to measure from Prometheus.

    Metrics must include a valid query.
    """

    targets: Optional[List[PrometheusTarget]]
    """An optional set of Prometheus target descriptors that are expected to be
    scraped by the Prometheus instance being queried.
    """

    @classmethod
    def generate(cls, **kwargs) -> "PrometheusConfiguration":
        """Generates a default configuration for capturing measurements from the
        Prometheus metrics server.

        Returns:
            A default configuration for PrometheusConnector objects.
        """
        return cls(
            description="Update the base_url and metrics to match your Prometheus configuration",
            metrics=[
                PrometheusMetric(
                    "throughput",
                    servo.Unit.REQUESTS_PER_SECOND,
                    query="rate(http_requests_total[5m])",
                    step="1m",
                ),
                PrometheusMetric(
                    "error_rate",
                    servo.Unit.PERCENTAGE,
                    query="rate(errors[5m])",
                    step="1m",
                ),
            ],
            **kwargs,
        )

    @pydantic.validator("base_url")
    @classmethod
    def rstrip_base_url(cls, base_url):
        return base_url.rstrip("/")

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"


class PrometheusRequest(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    metric: PrometheusMetric
    start: datetime.datetime
    end: datetime.datetime

    @property
    def query(self) -> str:
        return self.metric.query

    @property
    def step(self) -> servo.Duration:
        return self.metric.step

    @property
    def url(self) -> str:
        return "".join(
            self.base_url.rstrip("/")
            + "/query_range"
            + f"?query={self.query}"
            + f"&start={self.start.timestamp()}"
            + f"&end={self.end.timestamp()}"
            + f"&step={self.metric.step}"
        )


class PrometheusChecks(servo.BaseChecks):
    """PrometheusChecks objects check the state of a PrometheusConfiguration to
    determine if it is ready for use in an optimization run.
    """

    config: PrometheusConfiguration

    @servo.require('Connect to "{self.config.base_url}"')
    async def check_base_url(self) -> None:
        """Checks that the Prometheus base URL is valid and reachable."""
        async with httpx.AsyncClient(base_url=self.config.api_url) as client:
            response = await client.get("targets")
            response.raise_for_status()

    @servo.multicheck('Run query "{item.query}"')
    async def check_queries(self) -> Tuple[Iterable, servo.CheckHandler]:
        """Checks that all metrics have valid, well-formed PromQL queries."""

        async def query_for_metric(metric: PrometheusMetric) -> str:
            start, end = (
                datetime.datetime.now() - datetime.timedelta(minutes=10),
                datetime.datetime.now(),
            )
            prometheus_request = PrometheusRequest(
                base_url=self.config.api_url, metric=metric, start=start, end=end
            )

            self.logger.trace(
                f"Querying Prometheus (`{metric.query}`): {prometheus_request.url}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(prometheus_request.url)
                response.raise_for_status()
                result = response.json()
                return f"returned {len(result)} results"

        return self.config.metrics, query_for_metric

    @servo.check("Active targets")
    async def check_targets(self) -> str:
        """Checks that all targets are being scraped by Prometheus and report as
        healthy.
        """
        async with httpx.AsyncClient(base_url=self.config.base_url) as client:
            response = await client.get("/api/v1/targets")
            response.raise_for_status()
            result = response.json()

        target_count = len(result["data"]["activeTargets"])
        assert target_count > 0
        return f"found {target_count} targets"


@servo.metadata(
    description="Prometheus Connector for Opsani",
    version="1.5.0",
    homepage="https://github.com/opsani/prometheus-connector",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.STABLE,
)
class PrometheusConnector(servo.BaseConnector):
    """PrometheusConnector objects enable servo assemblies to capture
    measurements from the [Prometheus](https://prometheus.io/) metrics server.
    """

    config: PrometheusConfiguration

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter] = None,
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.CRITICAL,
    ) -> List[servo.Check]:
        """Checks that the configuration is valid and the connector can capture
        measurements from Prometheus.

        Checks are implemented in the PrometheusChecks class.

        Args:
            matching (Optional[Filter], optional): A filter for limiting the
                checks that are run. Defaults to None.
            halt_on (Severity, optional): When to halt running checks.
                Defaults to Severity.critical.

        Returns:
            List[Check]: A list of check objects that report the outcomes of the
                checks that were run.
        """
        return await PrometheusChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    @servo.on_event()
    def describe(self) -> servo.Description:
        """Describes the current state of Metrics measured by querying Prometheus.

        Returns:
            Description: An object describing the current state of metrics
                queried from Prometheus.
        """
        return servo.Description(metrics=self.config.metrics)

    @servo.on_event()
    def metrics(self) -> List[servo.Metric]:
        """Returns the list of Metrics measured through Prometheus queries.

        Returns:
            List[Metric]: The list of metrics to be queried.
        """
        return self.config.metrics

    @servo.on_event()
    async def measure(
        self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        """Queries Prometheus for metrics as time series values and returns a
        Measurement object that aggregates the readings for processing by the
        optimizer.

        Args:
            metrics (List[str], optional): A list of the metric names to measure.
                When None, all configured metrics are measured. Defaults to None.
            control (Control, optional): A control descriptor that describes how
                the measurement is to be captured. Defaults to Control().

        Returns:
            Measurement: An object that aggregates the state of the metrics
            queried from Prometheus.
        """
        if metrics:
            metrics__ = list(filter(lambda m: m.name in metrics, self.metrics()))
        else:
            metrics__ = self.metrics()
        measuring_names = list(map(lambda m: m.name, metrics__))
        self.logger.info(
            f"Starting measurement of {len(metrics__)} metrics: {servo.utilities.join_to_series(measuring_names)}"
        )

        start = datetime.datetime.now() + control.warmup
        end = start + control.duration

        sleep_duration = servo.Duration(control.warmup + control.duration)
        self.logger.info(
            f"Waiting {sleep_duration} during metrics collection ({control.warmup} warmup + {control.duration} duration)..."
        )

        progress = servo.DurationProgress(sleep_duration)
        notifier = lambda p: self.logger.info(
            p.annotate(f"waiting {sleep_duration} during metrics collection...", False),
            progress=p.progress,
        )
        await progress.watch(notifier)
        self.logger.info(
            f"Done waiting {sleep_duration} for metrics collection, resuming optimization."
        )

        # Capture the measurements
        self.logger.info(f"Querying Prometheus for {len(metrics__)} metrics...")
        readings = await asyncio.gather(
            *list(map(lambda m: self._query_prom(m, start, end), metrics__))
        )
        all_readings = (
            functools.reduce(lambda x, y: x + y, readings) if readings else []
        )
        measurement = servo.Measurement(readings=all_readings)
        return measurement

    async def _query_prom(
        self, metric: PrometheusMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        prometheus_request = PrometheusRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        self.logger.trace(
            f"Querying Prometheus (`{metric.query}`): {prometheus_request.url}"
        )
        async with self.api_client() as client:
            try:
                response = await client.get(prometheus_request.url)
                response.raise_for_status()
            except (
                httpx.HTTPError,
                httpcore._exceptions.ReadTimeout,
                httpcore._exceptions.ConnectError,
            ) as error:
                self.logger.trace(
                    f"HTTP error encountered during GET {prometheus_request.url}: {error}"
                )
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
            annotation = " ".join(
                map(lambda m: "=".join(m), sorted(m_.items(), key=lambda m: m[0]))
            )
            readings.append(
                servo.TimeSeries(
                    metric=metric,
                    annotation=annotation,
                    values=result_dict["values"],
                    id=f"{{instance={instance},job={job}}}",
                    metadata=dict(instance=instance, job=job),
                )
            )
        return readings
