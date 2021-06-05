import servo
import pydantic
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Type, Union
from .models import *

DEFAULT_BASE_URL = "http://prometheus:9090"

def _rstrip_slash(cls, base_url):
    return base_url.rstrip("/")

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

    streaming_interval: Optional[servo.Duration] = None

    metrics: List[PrometheusMetric]
    """The metrics to measure from Prometheus.

    Metrics must include a valid query.
    """

    targets: Optional[List[ActiveTarget]]
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
                        absent=AbsentMetricPolicy.ignore,
                        step="1m",
                    ),
                    PrometheusMetric(
                        "error_rate",
                        servo.Unit.percentage,
                        query="rate(errors[5m])",
                        absent=AbsentMetricPolicy.ignore,
                        step="1m",
                    ),
                ],
            ), **kwargs}
        )
