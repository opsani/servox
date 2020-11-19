import asyncio
import datetime
import functools
import re
import enum
from typing import Dict, Iterable, List, Optional, Tuple

import httpcore._exceptions
import httpx
import pydantic

import servo

DEFAULT_BASE_URL = "http://wavefront.com:2878"
API_PATH = "/api/v2/"


class Granularity(str, enum.Enum):
    """Granularity determines the resolution of the query, via the number of
    data points captured across a query range. A query's granularity is completely
    independent from any range durations specified in the WQL expression it evaluates.
    """

    second = "s"
    minute = "m"
    hour = "h"
    day = "d"


class Summarization(str, enum.Enum):
    """The summarization strategy to use when bucketing points together.
    Available values: MEAN, MEDIAN, MIN, MAX, SUM, COUNT, LAST, FIRST.
    """

    mean = "MEAN"
    median = "MEDIAN"
    min = "MIN"
    max = "MAX"
    sum = "SUM"
    count = "COUNT"
    last = "LAST"
    first = "FIRST"


class WavefrontMetric(servo.Metric):
    """WavefrontMetric objects describe metrics that can be measured by querying
    Wavefront.
    """

    query: str
    """A WQL query that returns the value of the target metric.

    For details on Wavefront, see the [Wavefront
    Querying](https://docs.Wavefront.com/query_language_getting_started.html)
    documentation.
    """

    granularity: Granularity = Granularity.minute
    """The granular resolution of the query, independent from any range durations specified
    in the WQL expression it evaluates.

    Available values: s, m, h, d.
    """

    summarized_by: Summarization = Summarization.last
    """Summarization strategy to use when bucketing points together.

    Available values: MEAN, MEDIAN, MIN, MAX, SUM, COUNT, LAST, FIRST.
    """

    @property
    def query_escaped(self) -> str:
        # Not used as such, leaving for now in prom style
        return re.sub(r"\{(.*?)\}", r"{{\1}}", self.query)

    def __check__(self) -> servo.Check:
        return servo.Check(
            name=f"Check {self.name}",
            description=f'Run Wavefront query "{self.query}"',
        )


class WavefrontConfiguration(servo.BaseConfiguration):
    """WavefrontConfiguration objects describe how WavefrontConnector objects
    capture measurements from the Wavefront metrics server.
    """

    api_key: pydantic.SecretStr
    """The API key for accessing the Wavefront metrics API."""

    base_url: pydantic.AnyHttpUrl = DEFAULT_BASE_URL
    """The base URL for accessing the Wavefront metrics API.

    The URL must point to the root of the Wavefront deployment. Resource paths
    are computed as necessary for API requests.
    """

    metrics: List[WavefrontMetric]
    """The metrics to measure from Wavefront.

    Metrics must include a valid query.
    """

    @classmethod
    def generate(cls, **kwargs) -> "WavefrontConfiguration":
        """Generates a default configuration for capturing measurements from the
        Wavefront metrics server.

        Returns:
            A default configuration for WavefrontConnector objects.
        """
        return cls(
            description="Update the api_key, base_url and metrics to match your Wavefront configuration",
            api_key='replace-me',
            metrics=[
                WavefrontMetric(
                    "throughput",
                    servo.Unit.REQUESTS_PER_MINUTE,
                    query="avg(ts(appdynamics.apm.overall.calls_per_min, env=foo and app=my-app))",
                    granularity=Granularity.minute,
                    summarized_by=Summarization.last,
                ),
                WavefrontMetric(
                    "error_rate",
                    servo.Unit.COUNT,
                    query="avg(ts(appdynamics.apm.transactions.errors_per_min, env=foo and app=my-app))",
                    granularity=Granularity.minute,
                    summarized_by=Summarization.last,
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


class WavefrontRequest(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    metric: WavefrontMetric
    start: datetime.datetime
    end: datetime.datetime

    @property
    def query(self) -> str:
        return self.metric.query

    @property
    def granularity(self) -> str:
        return self.metric.granularity

    @property
    def summarization(self) -> str:
        return self.metric.summarized_by

    @property
    def url(self) -> str:
        return "".join(
            self.base_url.rstrip("/")
            + "/chart/api"
            + f"?q={self.query}"
            + f"&s={self.start.timestamp()}"
            + f"&e={self.end.timestamp()}"
            + f"&g={self.metric.granularity}"
            + f"&summarization={self.metric.summarized_by}"
            + f"&strict=True"  # Should remain as non-configurable, else query will return points outside window
        )


class WavefrontChecks(servo.BaseChecks):
    """WavefrontChecks objects check the state of a WavefrontConfiguration to
    determine if it is ready for use in an optimization run.
    """

    config: WavefrontConfiguration

    @servo.multicheck('Run query "{item.query_escaped}"')
    async def check_queries(self) -> Tuple[Iterable, servo.CheckHandler]:
        """Checks that all metrics have valid, well-formed WQL queries."""

        async def query_for_metric(metric: WavefrontMetric) -> str:
            start, end = (
                datetime.datetime.now() - datetime.timedelta(minutes=10),
                datetime.datetime.now(),
            )
            wavefront_request = WavefrontRequest(
                base_url=self.config.api_url, metric=metric, start=start, end=end
            )

            self.logger.trace(
                f"Querying Wavefront (`{metric.query}`): {wavefront_request.url}"
            )
            async with httpx.AsyncClient(
                base_url=wavefront_request.url,
                headers={'Authorization': f'Bearer {self.config.api_key}'},
            ) as client:
                try:
                    response = await client.get(wavefront_request.url)
                    response.raise_for_status()
                    result = response.json()
                    return f"returned {len(result['timeseries'])} results"
                except (httpx.HTTPError, httpcore._exceptions.ReadTimeout, httpcore._exceptions.ConnectError) as error:
                    self.logger.trace(f"HTTP error encountered during GET {wavefront_request.url}: {error}")
                    raise

        return self.config.metrics, query_for_metric


@servo.metadata(
    description="Wavefront Connector for Opsani",
    version="1.0.0",
    homepage="https://github.com/opsani/servox",
    license=servo.License.APACHE2,
    maturity=servo.Maturity.STABLE,
)
class WavefrontConnector(servo.BaseConnector):
    """WavefrontConnector objects enable servo assemblies to capture
    measurements from the [Wavefront](https://Wavefront.io/) metrics server.
    """

    config: WavefrontConfiguration

    @servo.on_event()
    async def check(
        self,
        matching: Optional[servo.CheckFilter] = None,
        halt_on: Optional[servo.ErrorSeverity] = servo.ErrorSeverity.CRITICAL,
    ) -> List[servo.Check]:
        """Checks that the configuration is valid and the connector can capture
        measurements from Wavefront.

        Checks are implemented in the WavefrontChecks class.

        Args:
            matching (Optional[Filter], optional): A filter for limiting the
                checks that are run. Defaults to None.
            halt_on (Severity, optional): When to halt running checks.
                Defaults to Severity.critical.

        Returns:
            List[Check]: A list of check objects that report the outcomes of the
                checks that were run.
        """
        return await WavefrontChecks.run(
            self.config, matching=matching, halt_on=halt_on
        )

    @servo.on_event()
    def describe(self) -> servo.Description:
        """Describes the current state of Metrics measured by querying Wavefront.

        Returns:
            Description: An object describing the current state of metrics
                queried from Wavefront.
        """
        return servo.Description(metrics=self.config.metrics)

    @servo.on_event()
    def metrics(self) -> List[servo.Metric]:
        """Returns the list of Metrics measured through Wavefront queries.

        Returns:
            List[Metric]: The list of metrics to be queried.
        """
        return self.config.metrics

    @servo.on_event()
    async def measure(
            self, *, metrics: List[str] = None, control: servo.Control = servo.Control()
    ) -> servo.Measurement:
        """Queries Wavefront for metrics as time series values and returns a
        Measurement object that aggregates the readings for processing by the
        optimizer.

        Args:
            metrics (List[str], optional): A list of the metric names to measure.
                When None, all configured metrics are measured. Defaults to None.
            control (Control, optional): A control descriptor that describes how
                the measurement is to be captured. Defaults to Control().

        Returns:
            Measurement: An object that aggregates the state of the metrics
            queried from Wavefront.
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

        def notifier(p):
            return self.logger.info(
                p.annotate(f"waiting {sleep_duration} during metrics collection...", False),
                progress=p.progress,
            )

        await progress.watch(notifier)
        self.logger.info(
            f"Done waiting {sleep_duration} for metrics collection, resuming optimization."
        )

        # Capture the measurements
        self.logger.info(f"Querying Wavefront for {len(metrics__)} metrics...")
        readings = await asyncio.gather(
            *list(map(lambda m: self._query_wf(m, start, end), metrics__))
        )
        all_readings = (
            functools.reduce(lambda x, y: x + y, readings) if readings else []
        )
        measurement = servo.Measurement(readings=all_readings)
        return measurement

    async def _query_wf(
            self, metric: WavefrontMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        wavefront_request = WavefrontRequest(
            base_url=self.config.api_url, metric=metric, start=start, end=end
        )

        self.logger.trace(
            f"Querying Wavefront (`{metric.query}`): {wavefront_request.url}"
        )
        async with httpx.AsyncClient(
                base_url=wavefront_request.url,
                headers={'Authorization': f'Bearer {self.config.api_key}'},
        ) as client:
            try:
                response = await client.get(wavefront_request.url)
                response.raise_for_status()
            except (httpx.HTTPError, httpcore._exceptions.ReadTimeout, httpcore._exceptions.ConnectError) as error:
                self.logger.trace(f"HTTP error encountered during GET {wavefront_request.url}: {error}")
                raise

        data = response.json()
        self.logger.trace(f"Got response data for metric {metric}: {data}")

        readings = []

        for result_dict in data["timeseries"]:
            t_ = result_dict["tags"].copy()  # Unpack "tags" subdict and pack into a string
            instance = t_.get("nodename")  # e.g. 'ip-10-131-115-160.us-west-2.compute.internal'
            job = t_.get("type")  # e.g. 'node'
            annotation = " ".join(
                map(lambda m: "=".join(m), sorted(t_.items(), key=lambda m: m[0]))
            )

            if result_dict["data"]:
                readings.append(
                    servo.TimeSeries(
                        metric=metric,
                        annotation=annotation,
                        values=result_dict["data"],
                        id=f"{{instance={instance},job={job}}}",
                        metadata=dict(instance=instance, job=job),
                    )
                )
        return readings
