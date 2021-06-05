import servo
import datetime
from . import PrometheusConfiguration, Client, PrometheusMetric
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type, Union

class PrometheusChecks(servo.BaseChecks):
    """A collection of checks verifying the correctness and health of a Prometheus connector configuration.

    ### Attributes:
        config: The connector configuration being checked.
    """
    config: PrometheusConfiguration

    @property
    def _client(self) -> Client:
        return Client(base_url=self.config.base_url)

    @servo.require('Connect to "{self.config.base_url}"')
    async def check_base_url(self) -> None:
        """Checks that the Prometheus base URL is valid and reachable."""
        await self._client.list_targets()

    @servo.multicheck('Run query "{item.escaped_query}"')
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
            response = await self._client.query_range(metric, start, end)
            return f"returned {len(response.data)} results"

        return self.config.metrics, query_for_metric

    @servo.check("Active targets")
    async def check_targets(self) -> str:
        """Check that all targets are being scraped by Prometheus and report as healthy."""
        targets = await self._client.list_targets()
        assert len(targets.active) > 0, "no targets are being scraped by Prometheus"
        return f"found {len(targets.active)} targets"
