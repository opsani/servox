import asyncio
import datetime
import functools
import math
import operator
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type, Union

import pydantic
import pytz

import servo

from .configuration import PrometheusConfiguration
from .checks import PrometheusChecks
from .models import *
from .client import Client

DEFAULT_BASE_URL = "http://prometheus:9090"
API_PATH = "/api/v1"
CHANNEL = 'metrics.prometheus'

@servo.metadata(
    description="Prometheus Connector for Opsani",
    version="1.5.0",
    homepage="https://github.com/opsani/prometheus-connector",
    license=servo.License.apache2,
    maturity=servo.Maturity.stable,
)
class PrometheusConnector(servo.BaseConnector):
    """A servo connector that captures measurements from Prometheus.

    ### Attributes:
        config: The configuration of the connector instance.
    """
    config: PrometheusConfiguration

    @servo.on_event()
    async def startup(self) -> None:
        # Continuously publish a stream of metrics broadcasting every N seconds
        streaming_interval = self.config.streaming_interval
        if streaming_interval is not None:
            logger = servo.logger.bind(component=f"{self.name} -> {CHANNEL}")
            logger.info(f"Streaming Prometheus metrics every {streaming_interval}")

            @self.publish(CHANNEL, every=streaming_interval)
            async def _publish_metrics(publisher: servo.pubsub.Publisher) -> None:
                report = []
                client = Client(base_url=self.config.base_url)
                responses = await asyncio.gather(
                    *list(map(client.query, self.config.metrics)),
                    return_exceptions=True
                )
                for response in responses:
                    if isinstance(response, Exception):
                        logger.error(f"failed querying Prometheus for metrics: {response}")
                        continue

                    if response.data:
                        # NOTE: Instant queries return a single vector
                        timestamp, value = response.data[0].value
                        report.append((response.metric.name, timestamp.isoformat(), value))

                await publisher(servo.pubsub.Message(json=report))
                logger.debug(f"Published {len(report)} metrics.")

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
        """Returns the list of metrics measured by querying Prometheus."""
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

        # TODO: Rationalize these given the streaming metrics support
        start = datetime.datetime.now() + control.warmup
        end = start + control.duration
        measurement_duration = servo.Duration(control.warmup + control.duration)
        self.logger.info(
            f"Measuring {len(metrics__)} metrics for {measurement_duration}: {servo.utilities.join_to_series(measuring_names)}"
        )

        # Handle eager metrics
        eager_metrics = list(filter(lambda m: m.eager, metrics__))
        eager_settlement = max(eager_metrics, key=operator.attrgetter("eager")).eager if eager_metrics else None
        eager_observer = EagerMetricObserver(base_url=self.config.base_url, metrics=eager_metrics, start=start, end=end)
        if eager_metrics:
            servo.logger.info(f"Observing values of {len(eager_metrics)} eager metrics: measurement will return after {eager_settlement} of stability")
        else:
            servo.logger.debug(f"No eager metrics found: measurement will proceed for full duration of {measurement_duration}")

        # Allow the maximum settlement time of eager metrics to elapse before eager return (None runs full duration)
        progress = servo.EventProgress(timeout=measurement_duration, settlement=eager_settlement)
        await progress.watch(eager_observer.observe)

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

    async def targets(self) -> List[TargetsResponse]:
        """Return the targets discovered by Prometheus."""
        client = Client(base_url=self.config.base_url)
        response = await client.list_targets()
        return response

    async def _query_prometheus(
        self, metric: PrometheusMetric, start: datetime, end: datetime
    ) -> List[servo.TimeSeries]:
        client = Client(base_url=self.config.base_url)
        response = await client.query_range(metric, start, end)
        self.logger.trace(f"Got response data type {response.__class__} for metric {metric}: {response}")
        response.raise_for_error()

        if response.data:
            return response.results()
        else:
            # Handle absent metric cases
            if metric.absent in {AbsentMetricPolicy.ignore, AbsentMetricPolicy.zero}:
                # NOTE: metric zeroing is handled at the query level
                pass
            else:
                if await client.check_is_metric_absent(metric):
                    if metric.absent == AbsentMetricPolicy.warn:
                        servo.logger.warning(
                            f"Found absent metric for query (`{metric.query}`)"
                        )
                    elif metric.absent == AbsentMetricPolicy.fail:
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
    for target in targets.active:
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


class EagerMetricObserver(pydantic.BaseModel):
    base_url: pydantic.AnyHttpUrl
    metrics: List[servo.Metric]
    start: datetime.datetime
    end: datetime.datetime
    data_points: Dict[servo.Metric, servo.DataPoint] = {}

    async def observe(self, progress: servo.EventProgress) -> None:
        if not self.metrics:
            # bail if there are no eager metrics to observe
            servo.logger.info(
                progress.annotate(f"measuring Prometheus metrics for {progress.timeout}", False),
                progress=progress.progress,
            )
            return

        servo.logger.info(
            progress.annotate(f"measuring Prometheus metrics for up to {progress.timeout} (eager reporting when stable for {progress.settlement})...", False),
            progress=progress.progress,
        )
        if progress.timed_out:
            servo.logger.info(f"measurement duration of {progress.timeout} elapsed: reporting metrics")
            progress.complete()
            return
        else:
            for metric in self.metrics:
                active_data_point = self.data_points.get(metric)
                readings = await self._query_prometheus(metric)
                if readings:
                    data_point = readings[0][-1]
                    servo.logger.trace(f"Prometheus returned reading for the `{metric.name}` metric: {data_point}")
                    if data_point.value > 0:
                        if active_data_point is None:
                            active_data_point = data_point
                            servo.logger.success(progress.annotate(f"read `{metric.name}` metric value of {round(active_data_point.value)}{metric.unit}, awaiting {progress.settlement} before reporting"))
                            progress.trigger()
                        elif data_point.value != active_data_point.value:
                            previous_reading = active_data_point
                            active_data_point = data_point
                            delta_str = _chart_delta(previous_reading.value, active_data_point.value, metric.unit)
                            if progress.settling:
                                servo.logger.success(progress.annotate(f"read updated `{metric.name}` metric value of {round(active_data_point[1])}{metric.unit} ({delta_str}) during settlement, resetting to capture more data"))
                                progress.reset()
                            else:
                                # TODO: Should this just complete? How would we get here...
                                servo.logger.success(progress.annotate(f"read updated `{metric.name}` metric value of {round(active_data_point[1])}{metric.unit} ({delta_str}), awaiting {progress.settlement} before reporting"))
                                progress.trigger()
                        else:
                            servo.logger.debug(f"metric `{metric.name}` has not changed value, ignoring (reading={active_data_point}, num_readings={len(readings[0].data_points)})")
                    else:
                        if active_data_point:
                            # NOTE: If we had a value and fall back to zero it could be a burst
                            if not progress.settling:
                                servo.logger.warning(f"metric `{metric.name}` has fallen to zero from {active_data_point[1]}: may indicate a bursty traffic pattern. Will report eagerly if metric remains zero after {progress.settlement}")
                                progress.trigger()
                            else:
                                # NOTE: We are waiting out settlement
                                servo.logger.warning(f"metric `{metric.name}` has fallen to zero. Will report eagerly if metric remains zero in {progress.settlement_remaining}")
                        else:
                            servo.logger.debug(f"Prometheus returned zero value for the `{metric.name}` metric")
                else:
                    if active_data_point:
                        servo.logger.warning(progress.annotate(f"Prometheus returned no readings for the `{metric.name}` metric"))
                    else:
                        # NOTE: generally only happens on initialization and we don't care
                        servo.logger.trace(progress.annotate(f"Prometheus returned no readings for the `{metric.name}` metric"))

            if not progress.completed and not progress.timed_out:
                max_step_metric = max(self.metrics, key=operator.attrgetter("step"), default=None)
                servo.logger.debug(f"sleeping for {max_step_metric.step} to allow metrics to aggregate")
                await asyncio.sleep(max_step_metric.step.total_seconds())

    async def _query_prometheus(
        self, metric: PrometheusMetric
    ) -> List[servo.TimeSeries]:
        # TODO: Duplicating controller functionality. Refactor. Likely becomes boundary for Promethean library
        client = Client(base_url=self.base_url)
        response = await client.query_range(metric, self.start, self.end)
        servo.logger.trace(f"Got response data type {response.__class__} for metric {metric}: {response}")
        response.raise_for_error()

        if response.data:
            return response.results()
        else:
            # Handle absent metric cases
            if metric.absent in {AbsentMetricPolicy.ignore, AbsentMetricPolicy.zero}:
                # NOTE: metric zeroing is handled at the query level
                pass
            else:
                if await client.check_is_metric_absent(metric):
                    if metric.absent == AbsentMetricPolicy.warn:
                        servo.logger.warning(
                            f"Found absent metric for query (`{metric.query}`)"
                        )
                    elif metric.absent == AbsentMetricPolicy.fail:
                        servo.logger.error(f"Required metric '{metric.name}' is absent from Prometheus (query='{metric.query}')")
                        raise RuntimeError(f"Required metric '{metric.name}' is absent from Prometheus")
                    else:
                        raise ValueError(f"unknown metric absent value: {metric.absent}")

            return []
