import abc
import asyncio
import datetime
import enum
import functools
import math
import operator
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
    pool: str = pydantic.Field(..., alias='scrapePool')
    url: str = pydantic.Field(..., alias='scrapeUrl')
    global_url: str = pydantic.Field(..., alias='globalUrl')
    health: str
    labels: Optional[Dict[str, str]]
    discovered_labels: Optional[Dict[str, str]] = pydantic.Field(..., alias='discoveredLabels')
    last_scraped_at: Optional[datetime.datetime] = pydantic.Field(..., alias='lastScrape')
    last_scrape_duration: Optional[servo.Duration] = pydantic.Field(..., alias='lastScrapeDuration')
    last_error: Optional[str] = pydantic.Field(..., alias='lastError')


class BaseQuery(pydantic.BaseModel, abc.ABC):
    """BaseQuery models common behaviors across Prometheus query types."""
    metric: PrometheusMetric
    timeout: Optional[servo.Duration]

    @property
    def query(self) -> str:
        """Return the PromQL query."""
        if self.metric.absent == Absent.zero:
            return self.metric.query + " or on() vector(0)"
        else:
            return self.metric.query

    @abc.abstractmethod
    def url(self) -> str:
        """Return the relative URL for executing the query."""
        ...

class InstantQuery(BaseQuery):
    """Instant queries return a vector result reading metrics at a moment in time."""
    time: Optional[datetime.datetime]

    @property
    def url(self) -> str:
        """Return the relative URL for executing the query."""
        return "".join(
            "/query"
            + f"?query={self.query}"
            + (f"&time={self.time.timestamp()}" if self.time else "")
            + (f"&timeout={self.timeout}" if self.timeout else "")
        )


class RangeQuery(BaseQuery):
    """Range queries return a matrix result of a time series of metrics across time."""
    start: datetime.datetime
    end: datetime.datetime

    @pydantic.validator("end")
    @classmethod
    def _validate_range(cls, end, values) -> dict:
        assert end > values["start"], "start time must be earlier than end time"
        return end

    @property
    def step(self) -> servo.Duration:
        return self.metric.step

    @property
    def url(self) -> str:
        """Return the relative URL for executing the query."""
        return "".join(
            "/query_range"
            + f"?query={self.query}"
            + f"&start={self.start.timestamp()}"
            + f"&end={self.end.timestamp()}"
            + f"&step={self.metric.step}"
            + (f"&timeout={self.timeout}" if self.timeout else "")
        )

class ResultType(str, enum.Enum):
    """Types of results that can be returned for Prometheus Queries.

    See https://prometheus.io/docs/prometheus/latest/querying/api/#expression-query-result-formats
    """
    matrix = "matrix"
    vector = "vector"
    scalar = "scalar"
    string = "string"


Scalar = Tuple[datetime.datetime, float]
String = Tuple[datetime.datetime, str]


class BaseVector(abc.ABC, pydantic.BaseModel):
    """Abstract base class for Prometheus vector types.

    Attributes:
        metric: The Prometheus metric name and labels.
    """
    metric: Dict[str, str]

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __iter__(self) -> Scalar:
        ...


class InstantVector(BaseVector):
    """InstantVector objects model the value of a metric captured at a moment in time.
    [
        {
            'metric': {},
            'value': [
                1607989427.782,
                '19.8',
            ],
        },
    ]
    """
    value: Scalar

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Scalar:
        return iter((self.value, ))


class RangeVector(BaseVector):
    """A collection of values of a metric captured over a time range.
    [
        {
            "metric": { "<label_name>": "<label_value>", ... },
            "values": [ [ <unix_time>, "<sample_value>" ], ... ]
        },
    ]
    """
    values: List[Scalar]

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Scalar:
        return iter(self.values)


class Status(str, enum.Enum):
    """Prometheus API query response statuses.

    See https://prometheus.io/docs/prometheus/latest/querying/api/#format-overview
    """
    success = "success"
    error = "error"


class Error(pydantic.BaseModel):
    type: str = pydantic.Field(..., alias='errorType')
    message: str = pydantic.Field(..., alias='error')


class Data(pydantic.BaseModel):
    type: ResultType = pydantic.Field(..., alias='resultType')
    result: Union[List[InstantVector], List[RangeVector], Scalar, String]

    def __len__(self) -> int:
        if self.is_vector:
            return len(self.result)
        elif self.is_value:
            return 1
        else:
            raise TypeError(f"unknown data type '{self.type}'")

    def __iter__(self):
        if self.is_vector:
            return iter(self.result)
        elif self.is_value:
            return iter((self.result, ))
        else:
            raise TypeError(f"unknown data type '{self.type}'")

    @property
    def is_vector(self) -> bool:
        return self.type in (servo.connectors.prometheus.ResultType.vector, servo.connectors.prometheus.ResultType.matrix)

    @property
    def is_value(self) -> bool:
        return self.type in (servo.connectors.prometheus.ResultType.scalar, servo.connectors.prometheus.ResultType.string)

class Response(pydantic.BaseModel):
    """Response objects model a PromQL query response returned from the Prometheus API."""
    query: BaseQuery
    status: Status
    error: Optional[Error]
    warnings: Optional[List[str]]
    data: Optional[Data]

    @pydantic.root_validator(pre=True)
    def _parse_error(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if error := dict(filter(lambda item: item[0].startswith("error"), values.items())):
            values["error"] = error
        return values

    def raise_for_error(self) -> None:
        """Raise an error if the query was not successful."""
        if self.status == Status.error:
            raise RuntimeError(f"Prometheus query request failed with error '{self.error.type}': {self.error.messge}")

    def results(self) -> Optional[List[servo.Reading]]:
        """Return DataPoint or TimeSeries representations of the query results.

        Vector responses are mapped to
        """
        if self.status == Status.error:
            return None
        elif not self.data:
            return []

        results_ = []
        for result in self.data:
            if self.data.is_vector:
                results_.append(
                    self._time_series_from_vector(result)
                )
            elif self.data.is_value:
                results_.append(
                    servo.DataPoint(self.query.metric, **result)
                )
            else:
                raise TypeError(f"unknown Result type '{result.__class__.name}' encountered")

        return results_

    def _time_series_from_vector(self, vector: BaseVector) -> servo.TimeSeries:
        instance = vector.metric.get("instance")
        job = vector.metric.get("job")
        annotation = " ".join(
            map(lambda m: "=".join(m), sorted(vector.metric.items(), key=operator.itemgetter(0)))
        )
        return servo.TimeSeries(
            self.query.metric,
            list(map(lambda v: servo.DataPoint(self.query.metric, *v), iter(vector))),
            id=f"{{instance={instance},job={job}}}",
            annotation=annotation,
        )

def _rstrip_slash(cls, base_url):
    return base_url.rstrip("/")

class Client(pydantic.BaseModel):
    """Client objects interact with the Prometheus API.

    The client supports instant and range queries and retrieving the targets.
    Requests and responses are serialized through an object model to make working
    with Prometheus fast and ergonomic.

    Args:
        base_url: The base URL for connecting to Prometheus.

    Attributes:
        base_url: The base URL for connecting to Prometheus.
    """
    base_url: pydantic.AnyHttpUrl
    _normalize_base_url = pydantic.validator('base_url', allow_reuse=True)(_rstrip_slash)

    @property
    def api_url(self) -> str:
        """Return the full URL for accessing the Prometheus API."""
        return f"{self.base_url}{API_PATH}"

    async def get_query(self, query: BaseQuery) -> Response:
        """Run a query and return the response."""
        servo.logger.trace(
            f"Querying Prometheus (`{query.metric.query}`): {query.url}"
        )
        async with httpx.AsyncClient(base_url=self.api_url) as client:
            try:
                response = await client.get(query.url)
                response.raise_for_status()
                return Response(query=query, **response.json())
            except (
                httpx.HTTPError,
                httpx.ReadTimeout,
                httpx.ConnectError,
            ) as error:
                self.logger.trace(
                    f"HTTP error encountered during GET {query.url}: {error}"
                )
                raise

    async def get_targets(self) -> List[PrometheusTarget]:
        """Return a list of targets being scraped by Prometheus."""
        async with httpx.AsyncClient(base_url=self.api_url) as client:
            response = await client.get("/targets")
            response.raise_for_status()
            return pydantic.parse_obj_as(List[PrometheusTarget], response.json()['data']['activeTargets'])

    # get_range_query, get_metric_query
    async def get_instant_query(
        promql: str,
        *,
        time: Optional[datetime.datetime] = None,
        timeout: Optional[servo.Duration] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        step: Optional[str] = None,
    ):
        """Run an adhoc query and return the results."""
        ...

    async def check_for_absent_metric(self, metric: PrometheusMetric) -> bool:
        # Determine if the metric is actually absent or just returned an empty result set
        absent_metric = metric.copy()
        absent_metric.query = f"absent({metric.query})"
        response = await self.get_query(InstantQuery(metric=absent_metric))
        servo.logger.debug(f"Absent metric introspection returned {absent_metric}: {response}")
        if response.data:
            if response.data.type != servo.connectors.prometheus.ResultType.vector:
                raise TypeError(f"expected a vector result but found {response.data.type}")
            if len(response.data) != 1:
                raise ValueError(f"expected a single result vector but found {len(response.data)}")
            result = next(iter(response.data))
            return int(result.value[1]) == 1

        else:
            servo.logger.info(f"Metric '{absent_metric.name}' is present in Prometheus but returned an empty result set (query='{absent_metric.query}')")
            return False

class PrometheusConfiguration(servo.BaseConfiguration):
    """PrometheusConfiguration objects describe how PrometheusConnector objects
    capture measurements from the Prometheus metrics server.
    """

    base_url: pydantic.AnyHttpUrl = DEFAULT_BASE_URL
    _normalize_base_url = pydantic.validator('base_url', allow_reuse=True)(_rstrip_slash)
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
                        servo.Unit.requests_per_second,
                        query="rate(http_requests_total[5m])",
                        absent=Absent.zero,
                        step="1m",
                    ),
                    PrometheusMetric(
                        "error_rate",
                        servo.Unit.percentage,
                        query="rate(errors[5m])",
                        absent=Absent.zero,
                        step="1m",
                    ),
                ],
            ), **kwargs}
        )


class PrometheusChecks(servo.BaseChecks):
    """PrometheusChecks objects check the state of a PrometheusConfiguration to
    determine if it is ready for use in an optimization run.
    """
    config: PrometheusConfiguration

    @property
    def _client(self) -> Client:
        return Client(base_url=self.config.base_url)

    @servo.require('Connect to "{self.config.base_url}"')
    async def check_base_url(self) -> None:
        """Checks that the Prometheus base URL is valid and reachable."""
        await self._client.get_targets()

    @servo.multicheck('Run query "{item.query_escaped}"')
    async def check_queries(self) -> Tuple[Iterable, servo.CheckHandler]:
        """Checks that all metrics have valid, well-formed PromQL queries."""

        async def query_for_metric(metric: PrometheusMetric) -> str:
            start, end = (
                datetime.datetime.now() - datetime.timedelta(minutes=10),
                datetime.datetime.now(),
            )

            self.logger.trace(
                f"Querying Prometheus (`{metric.query}`)"
            )
            response = await self._client.get_query(
                RangeQuery(metric=metric, start=start, end=end)
            )
            return f"returned {len(response.data)} results"

        return self.config.metrics, query_for_metric

    @servo.check("Active targets")
    async def check_targets(self) -> str:
        """Check that all targets are being scraped by Prometheus and report as healthy."""
        targets = await self._client.get_targets()
        assert len(targets) > 0, "no targets are being scraped by Prometheus"
        return f"found {len(targets)} targets"


@servo.metadata(
    description="Prometheus Connector for Opsani",
    version="1.5.0",
    homepage="https://github.com/opsani/prometheus-connector",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
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
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.critical,
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

        # TODO: Wrap this into a partial?
        active_data_point: Optional[servo.DataPoint] = None
        async def check_metrics(progress: servo.EventProgress) -> None:
            nonlocal active_data_point
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
                    data_point = throughput_readings[0][-1]
                    self.logger.trace(f"Prometheus returned reading for the `{target_metric.name}` metric: {data_point}")
                    if data_point.value > 0:
                        if active_data_point is None:
                            active_data_point = data_point
                            self.logger.success(progress.annotate(f"read `{target_metric.name}` metric value of {round(active_data_point.value)}{target_metric.unit}, awaiting {progress.settlement} before reporting"))
                            progress.trigger()
                        elif data_point.value != active_data_point.value:
                            previous_reading = active_data_point
                            active_data_point = data_point
                            delta_str = _chart_delta(previous_reading.value, active_data_point.value, target_metric.unit)
                            if progress.settling:
                                self.logger.success(progress.annotate(f"read updated `{target_metric.name}` metric value of {round(active_data_point[1])}{target_metric.unit} ({delta_str}) during settlement, resetting to capture more data"))
                                progress.reset()
                            else:
                                # TODO: Should this just complete? How would we get here...
                                self.logger.success(progress.annotate(f"read updated `{target_metric.name}` metric value of {round(active_data_point[1])}{target_metric.unit} ({delta_str}), awaiting {progress.settlement} before reporting"))
                                progress.trigger()
                        else:
                            self.logger.debug(f"metric `{target_metric.name}` has not changed value, ignoring (reading={active_data_point}, num_readings={len(throughput_readings[0]._data_points)})")
                    else:
                        if active_data_point:
                            # NOTE: If we had a value and fall back to zero it could be a burst
                            if not progress.settling:
                                servo.logger.warning(f"metric `{target_metric.name}` has fallen to zero from {active_data_point[1]}: may indicate a bursty traffic pattern. Will report eagerly if metric remains zero after {progress.settlement}")
                                progress.trigger()
                            else:
                                # NOTE: We are waiting out settlement
                                servo.logger.warning(f"metric `{target_metric.name}` has fallen to zero. Will report eagerly if metric remains zero in {progress.settlement_remaining}")
                        else:
                            servo.logger.debug(f"Prometheus returned zero value for the `{target_metric.name}` metric")
                else:
                    if active_data_point:
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

        # Capture the measurements
        self.logger.info(f"Querying Prometheus for {len(metrics__)} metrics...")
        readings = await asyncio.gather(
            *list(map(lambda m: self._query_prometheus(m, start, end), metrics__))
        )
        debug("GATHERED READINGS: ", readings)
        all_readings = (
            functools.reduce(lambda x, y: x + y, readings) if readings else []
        )
        measurement = servo.Measurement(readings=all_readings)
        return measurement

    async def targets(self) -> List:
        """Return a list of targets being scraped by Prometheus."""
        client = Client(base_url=self.config.base_url)
        return await client.get_targets()

    async def _query_prometheus(
        self, metric: PrometheusMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        client = Client(base_url=self.config.base_url)
        response = await client.get_query(
            RangeQuery(metric=metric, start=start, end=end)
        )
        self.logger.trace(f"Got response data type {response.__class__} for metric {metric}: {response}")
        response.raise_for_error()

        if response.data:
            debug("data is there")
            return response.results()
        else:
            # Handle absent metric cases
            if metric.absent in {Absent.ignore, Absent.zero}:
                # NOTE: metric zeroing is handled at the query level
                pass
            else:
                absent = await client.check_for_absent_metric(metric)
                if metric.absent == Absent.warn:
                    servo.logger.warning(
                        f"Found absent metric for query (`{metric.query}`)"
                    )
                elif metric.absent == Absent.fail:
                    servo.logger.error(f"Required metric '{metric.name}' is absent from Prometheus (query='{metric.query}')")
                    raise RuntimeError(f"Required metric '{metric.name}' is absent from Prometheus")
                else:
                    raise ValueError(f"unknown metric absent value: {metric.absent}")

            return []

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
