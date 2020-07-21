from datetime import timedelta

from pydantic import ValidationError

from servo.connectors.prometheus import PrometheusConfiguration, PrometheusMetric
from servo.types import *


class TestPrometheusMetric:
    def test_accepts_period_as_duration(self):
        metric = PrometheusMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE,
            query="throughput",
            period="45m",
        )
        assert metric.period == timedelta(seconds=2700)  # 45 mins

    def test_accepts_period_as_integer_of_seconds(self):
        metric = PrometheusMetric(
            name="test", unit=Unit.REQUESTS_PER_MINUTE, query="throughput", period=180,
        )
        assert metric.period
        debug(metric.period)
        assert metric.period == timedelta(seconds=180)

    # Query
    def test_query_required(self):
        try:
            PrometheusMetric(
                name="throughput", unit=Unit.REQUESTS_PER_MINUTE, query=None
            )
        except ValidationError as error:
            assert {
                "loc": ("query",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    # NOTE: Floating point values may come back as strings?
    def test_conversion_of_floats_from_strings(self):
        pass

    # item[1] == 'NaN':
    def test_handling_nan_values(self):
        pass


class TestPrometheusConfiguration:
    # URL
    def test_url_required(self):
        try:
            PrometheusConfiguration(base_url=None)
        except ValidationError as error:
            assert {
                "loc": ("base_url",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    def test_base_url_is_rstripped(self):
        config = PrometheusConfiguration(
            base_url="http://prometheus.io/some/path/", metrics=[]
        )
        assert config.base_url == "http://prometheus.io/some/path/"

    def test_supports_localhost_url(self):
        config = PrometheusConfiguration(base_url="http://localhost:9090", metrics=[])
        assert config.base_url == "http://localhost:9090"

    def test_supports_cluster_url(self):
        config = PrometheusConfiguration(
            base_url="http://prometheus.default.svc.cluster.local:9090", metrics=[]
        )
        assert config.base_url == "http://prometheus.default.svc.cluster.local:9090"

    def test_rejects_invalid_url(self):
        try:
            PrometheusConfiguration(base_url="gopher://this-is-invalid")
        except ValidationError as error:
            assert {
                "loc": ("base_url",),
                "msg": "URL scheme not permitted",
                "type": "value_error.url.scheme",
                "ctx": {"allowed_schemes": {"http", "https",},},
            } in error.errors()

    def test_api_url(self):
        config = PrometheusConfiguration(
            base_url="http://prometheus.default.svc.cluster.local:9090", metrics=[]
        )
        assert (
            config.api_url == "http://prometheus.default.svc.cluster.local:9090/api/v1/"
        )

    # Metrics
    def test_metrics_required(self):
        try:
            PrometheusConfiguration(metrics=None)
        except ValidationError as error:
            assert {
                "loc": ("metrics",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    # Generation
    def test_generate_default_config(self):
        config = PrometheusConfiguration.generate()
        assert config.yaml() == (
            "base_url: http://prometheus:9090\n"
            "description: Update the base_url and metrics to match your Prometheus configuration\n"
            "metrics:\n"
            "- name: throughput\n"
            "  period: 1m\n"
            "  query: rate(http_requests_total[1s])[3m]\n"
            "  unit: rps\n"
            "- name: error_rate\n"
            "  period: 1m\n"
            "  query: rate(errors)\n"
            "  unit: '%'\n"
        )


# TODO: Add support for before and after filters that enable warmup and settlement
# TODO: Progress logging
# TODO: Async/Cancellation -> Dispatch events async
# TODO: What is the right name for the `period` attribute?
# TODO: Reporting interval...


class TestPrometheusConnector:
    def test_describe(self):
        pass

    def test_measure(self):
        pass

    def test_check_fails_with_invalid_query(self):
        pass

    def test_check_fails_if_unreachable(self):
        pass

    def test_check_fails_with_invalid_query(self):
        pass


class TestPrometheusCLI:
    def test_metrics(self):
        pass

    def test_measure(self):
        pass

    def test_check(self):
        pass


# TODO: Annotate with pydantic and run prometheus locally for smoke tests
class TestPrometheusIntegration:
    pass
