import asyncio
import datetime
import enum
import functools
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpcore._exceptions
import httpx
import pydantic
import pytz

import servo

DEFAULT_BASE_URL = "http://prometheus:9090"
API_PATH = "/api/v1"


class Absent(str, enum.Enum):
    """An enumeration of behaviors for handling absent metrics.

    Absent metrics are metrics that do not exist in Prometheus at query time.
    This may indicate that the metric has never been reported or that the Prometheus
    instance has limited state (e.g., Opsani Dev utilizes a transient Prometheus sidecar).

    The behaviors are:
        * ignore: Silently ignore the absent metric and continue processing.
        * zero: Return a vector of zero for the absent metric.
        * warn: Log a warning message when an absent metric is encountered.
        * fail: Raise a runtime exception when an absent metric is encountered.
    """
    ignore = "ignore"
    zero = "zero"
    warn = "warn"
    fail = "fail"

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

    absent: Absent = Absent.ignore
    """How to behave when the metric is absent from Prometheus."""

    @property
    def query_escaped(self) -> str:
        return re.sub(r"\{(.*?)\}", r"{{\1}}", self.query)

    def __check__(self) -> servo.Check:
        return servo.Check(
            name=f"Check {self.name}",
            description=f'Run Prometheus query "{self.query}"',
        )


class PrometheusTarget(pydantic.BaseModel):
    """PrometheusTarget objects describe targets that are scraped by Prometheus jobs."""
    pool: str
    url: str
    global_url: str
    health: str # TODO: This should be an enum but I dunno what the values could be
    labels: Optional[Dict[str, str]]
    discovered_labels: Optional[Dict[str, str]]
    last_scraped_at: Optional[datetime.datetime]
    last_scrape_duration: Optional[servo.Duration]
    last_error: Optional[str]
    
    @pydantic.root_validator(pre=True)
    def _map_from_prometheus_json(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pool": values["scrapePool"],
            "url": values["scrapeUrl"],
            "global_url": values["globalUrl"],
            "health": values["health"],
            "labels": values["labels"],
            "discovered_labels": values["discoveredLabels"],
            "last_scraped_at": values["lastScrape"],
            "last_scrape_duration": values["lastScrapeDuration"],
            "last_error": values["lastError"] if values["lastError"] else None
        }


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
        """Generate a default configuration for capturing measurements from the
        Prometheus metrics server.

        Returns:
            A default configuration for PrometheusConnector objects.
        """
        return cls(
            **{**dict(
                description="Update the base_url and metrics to match your Prometheus configuration",
                metrics=[
                    PrometheusMetric(
                        "throughput",
                        servo.Unit.REQUESTS_PER_SECOND,
                        query="rate(http_requests_total[5m])",
                        absent=Absent.zero,
                        step="1m",
                    ),
                    PrometheusMetric(
                        "error_rate",
                        servo.Unit.PERCENTAGE,
                        query="rate(errors[5m])",
                        absent=Absent.zero,
                        step="1m",
                    ),
                ],
            ), **kwargs}
        )

    @pydantic.validator("base_url")
    @classmethod
    def rstrip_base_url(cls, base_url):
        return base_url.rstrip("/")

    @property
    def api_url(self) -> str:
        return f"{self.base_url}{API_PATH}"

class BaseQuery(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    metric: PrometheusMetric


class InstantQuery(BaseQuery):
    @property
    def query(self) -> str:
        return self.metric.query

    @property
    def url(self) -> str:
        return "".join(
            self.base_url.rstrip("/")
            + "/query"
            + f"?query={self.query}"
        )


class RangeQuery(BaseQuery):
    start: datetime.datetime
    end: datetime.datetime

    @property
    def query(self) -> str:
        if self.metric.absent == Absent.zero:
            return self.metric.query + " or on() vector(0)"
        else:
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

class ResultType(str, enum.Enum):
    """Types of results that can be returned for Prometheus Queries.

    See https://prometheus.io/docs/prometheus/latest/querying/api/#expression-query-result-formats
    """
    matrix = "matrix"
    vector = "vector"
    scalar = "scalar"
    string = "string"

class QueryResult(pydantic.BaseModel):
    query: BaseQuery
    status: str
    type: ResultType
    metric: Optional[dict] # TODO: dunno here...
    values: Optional[List[Tuple[datetime.datetime, float]]]
    data: Any

    @property
    def metric(self) -> None:
        return self.query.metric

    @pydantic.root_validator(pre=True)
    def _map_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        result = values["data"]["result"][0] if values["data"]["result"] else {}
        return {
            "query": values["query"],
            "status": values["status"],
            "type": values["data"]["resultType"],
            "metric": result.get("metric", None),
            "values": result.get("values", None),
            "data": values["data"]
        }
    
    @pydantic.validator("values")
    @classmethod
    def _sort_values(cls, values: Optional[List[Tuple[datetime.datetime, float]]]) -> Optional[List[Tuple[datetime.datetime, float]]]:        
        return sorted(values, key=lambda x: x[0]) if values else None


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

    @servo.multicheck('Run query "{item.query_escaped}"')
    async def check_queries(self) -> Tuple[Iterable, servo.CheckHandler]:
        """Checks that all metrics have valid, well-formed PromQL queries."""

        async def query_for_metric(metric: PrometheusMetric) -> str:
            start, end = (
                datetime.datetime.now() - datetime.timedelta(minutes=10),
                datetime.datetime.now(),
            )
            prometheus_request = RangeQuery(
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
        """Check that all targets are being scraped by Prometheus and report as healthy."""

        async with httpx.AsyncClient(base_url=self.config.base_url) as client:
            response = await client.get("/api/v1/targets")
            response.raise_for_status()
            result = response.json()

        target_count = len(result["data"]["activeTargets"])
        assert target_count > 0, "no targets are being scraped by Prometheus"
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
            f"Measuring {len(metrics__)} metrics: {servo.utilities.join_to_series(measuring_names)}"
        )
        
        # TODO: Rationalize these given the streaming metrics support
        start = datetime.datetime.now() + control.warmup
        end = start + control.duration
        measurement_duration = servo.Duration(control.warmup + control.duration)
        # TODO: Push target metrics into config and support nore than 1
        target_metric = next(filter(lambda m: m.name == "throughput", metrics__))
        active_reading: Optional[Tuple[datetime.datetime, float]] = None
        
        async def check_metrics(progress: servo.EventProgress) -> None:
            nonlocal active_reading
            self.logger.info(
                progress.annotate(f"measuring Prometheus metrics for up to {progress.timeout} (eager reporting when stable for {progress.settlement})...", False),
                progress=progress.progress,
            )
            if progress.timed_out:
                self.logger.info(f"measurement duration of {measurement_duration} elapsed: reporting metrics")
                progress.complete()
                return
            else:
                # NOTE: We need throughput to do anything meaningful. Generalize?
                throughput_readings = await self._query_prometheus(target_metric, start, end)
                if throughput_readings:
                    latest_reading = throughput_readings[0].last()                    
                    self.logger.trace(f"Prometheus returned reading for the `{target_metric.name}` metric: {latest_reading}")
                    if latest_reading[1] > 0:
                        if active_reading is None:
                            active_reading = latest_reading
                            self.logger.success(progress.annotate(f"read `{target_metric.name}` metric value of {round(active_reading[1])}{target_metric.unit}, awaiting {progress.settlement} before reporting"))
                            progress.trigger()
                        elif latest_reading[1] != active_reading[1]:
                            previous_reading = active_reading
                            active_reading = latest_reading
                            delta_str = _chart_delta(previous_reading[1], active_reading[1], target_metric.unit)
                            if progress.settling:
                                self.logger.success(progress.annotate(f"read updated `{target_metric.name}` metric value of {round(active_reading[1])}{target_metric.unit} ({delta_str}) during settlement, resetting to capture more data"))
                                progress.reset()
                            else:
                                # TODO: Should this just complete? How would we get here...
                                self.logger.success(progress.annotate(f"read updated `{target_metric.name}` metric value of {round(active_reading[1])}{target_metric.unit} ({delta_str}), awaiting {progress.settlement} before reporting"))
                                progress.trigger()
                        else:
                            self.logger.debug(f"metric `{target_metric.name}` has not changed value, ignoring (reading={active_reading}, num_readings={len(throughput_readings[0].values)})")
                    else:                        
                        if active_reading:
                            # NOTE: If we had a value and fall back to zero it could be a burst
                            if not progress.settling:
                                servo.logger.warning(f"Prometheus returned zero value for the `{target_metric.name}` metric after returning a non-zero value. Could be a burst: {active_reading}")
                                progress.trigger()
                            else:
                                # NOTE: We are waiting out settlement
                                servo.logger.warning(f"zero value metric under settlement with {progress.settlement_remaining} remaining")
                        else:
                            servo.logger.debug(f"Prometheus returned zero value for the `{target_metric.name}` metric")
                else:
                    if active_reading:
                        servo.logger.warning(progress.annotate(f"Prometheus returned no readings for the `{target_metric.name}` metric"))
                    else:
                        # NOTE: generally only happens on initialization and we don't care
                        servo.logger.trace(progress.annotate(f"Prometheus returned no readings for the `{target_metric.name}` metric"))
            
            if not progress.completed and not progress.timed_out:
                servo.logger.debug(f"sleeping for {target_metric.step} to allow metrics to aggregate")
                await asyncio.sleep(target_metric.step.total_seconds())
            
        # TODO: The settlement time is totally arbitrary. Configure? Push up to the server under control field?
        progress = servo.EventProgress(timeout=measurement_duration, settlement=servo.Duration("1m"))
        await progress.watch(check_metrics)
        debug(progress, "progress is ", progress.progress)
        # TODO: need a repr that reflects how we exited

        # Capture the measurements
        self.logger.info(f"Querying Prometheus for {len(metrics__)} metrics...")
        readings = await asyncio.gather(
            *list(map(lambda m: self._query_prometheus(m, start, end), metrics__))
        )
        all_readings = (
            functools.reduce(lambda x, y: x + y, readings) if readings else []
        )
        measurement = servo.Measurement(readings=all_readings)
        return measurement

    async def targets(self) -> List:
        """Return a list of targets being scraped by Prometheus."""
        async with httpx.AsyncClient(base_url=self.config.base_url) as client:
            response = await client.get("/api/v1/targets")
            response.raise_for_status()            
            return pydantic.parse_obj_as(List[PrometheusTarget], response.json()['data']['activeTargets'])

    async def _query_prometheus(
        self, metric: PrometheusMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        async def _query(request: BaseQuery) -> QueryResult:
            self.logger.trace(
                f"Querying Prometheus (`{request.metric.query}`): {request.url}"
            )
            async with self.api_client() as client:
                try:
                    response = await client.get(request.url)
                    response.raise_for_status()
                    return QueryResult(query=request, **response.json())
                except (
                    httpx.HTTPError,
                    httpcore._exceptions.ReadTimeout,
                    httpcore._exceptions.ConnectError,
                ) as error:
                    self.logger.trace(
                        f"HTTP error encountered during GET {request.url}: {error}"
                    )
                    raise

        request = RangeQuery(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )
        result = await _query(request)
        self.logger.trace(f"Got response data type {result.__class__} for metric {metric}: {result}")

        if result.status != "success":
            # TODO: Prolly need to raise or error here?
            return []

        readings = []
        # TODO: check and handle the resultType
        if result.values == []:
            if metric.absent == Absent.ignore:
                pass
            elif metric.absent == Absent.zero:
                # Handled in RangeQuery
                pass
            else:
                # Check for an absent metric
                absent_metric = metric.copy()
                absent_metric.query = f"absent({metric.query})"
                absent_query = InstantQuery(
                    base_url=self.config.api_url, metric=absent_metric
                )
                absent_result = await _query(absent_query)
                self.logger.debug(f"Absent metric introspection returned {absent_metric}: {absent_result}")

                # TODO: this is brittle...
                # [{'metric': {}, 'value': [1607078958.682, '1']}]
                absent = int(absent_result.value[0]['value'][1]) == 1
                if absent:
                    if metric.absent == Absent.warn:
                        self.logger.warning(
                            f"Found absent metric for query (`{absent_metric.query}`): {absent_query.url}"
                        )
                    elif metric.absent == Absent.fail:
                        raise RuntimeError(f"Found absent metric for query (`{absent_metric.query}`): {absent_query.url}")
                    else:
                        raise ValueError(f"unknown metric absent value: {metric.absent}")

        else:
            # debug(result)
            # debug("HOLY SHIT REUSLT METRIC", result.metric)
            # TODO: This should not be necessary...
            for result_dict in result.data["result"]:
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
                        values=result.values,
                        id=f"{{instance={instance},job={job}}}",
                        metadata=dict(instance=instance, job=job),
                    )
                )

        return readings


app = servo.cli.ConnectorCLI(PrometheusConnector, help="Metrics from Prometheus")

@app.command()
def targets(
    context: servo.cli.Context,
):
    """Display the targets being scraped."""
    targets = servo.cli.run_async(context.connector.targets())
    headers = ["POOL", "HEALTH", "URL", "LABELS", "LAST SCRAPED", "ERROR"]
    table = []
    for target in targets:
        labels = sorted(
            list(
                map(
                    lambda l: f"{l[0]}={l[1]}",
                    target.labels.items(),
                )
            )
        )
        table.append(
            [
                target.pool,
                target.health,
                f"{target.url} ({target.global_url})" if target.url != target.global_url else target.url,
                "\n".join(labels),
                f"{target.last_scraped_at:%Y-%m-%d %H:%M:%S} ({servo.cli.timeago(target.last_scraped_at, pytz.utc.localize(datetime.datetime.now()))} in {target.last_scrape_duration})" if target.last_scraped_at else "-",
                target.last_error or "-",
            ]
        )
    
    servo.cli.print_table(table, headers)

def _delta(a, b):
    if (a == b):
        return 0
    elif (a < 0) and (b < 0) or (a > 0) and (b > 0):
        if (a < b):
            return (abs(abs(a) - abs(b)))
        else:
            return -(abs(abs(a) - abs(b)))
    else:
        return math.copysign((abs(a) + abs(b)),b)

def _chart_delta(a, b, unit) -> str:
    delta = _delta(round(a), round(b))
    if delta == 0:
        return "♭"
    elif delta < 0:
        return f"📉{delta}{unit}"
    else:
        return f"📈+{delta}{unit}"
