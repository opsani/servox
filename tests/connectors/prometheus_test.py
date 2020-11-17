import datetime
import re

import httpx
import pytest
import respx
from freezegun import freeze_time
from pydantic import ValidationError

from servo.connectors.prometheus import PrometheusChecks, PrometheusConfiguration, PrometheusMetric, PrometheusRequest
from servo.types import *


class TestPrometheusMetric:
    def test_accepts_step_as_duration(self):
        metric = PrometheusMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE,
            query="throughput",
            step="45m",
        )
        assert metric.step == datetime.timedelta(seconds=2700)  # 45 mins

    def test_accepts_step_as_integer_of_seconds(self):
        metric = PrometheusMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE,
            query="throughput",
            step=180,
        )
        assert metric.step
        assert metric.step == datetime.timedelta(seconds=180)

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
        assert config.base_url == "http://prometheus.io/some/path"

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
                "ctx": {
                    "allowed_schemes": {
                        "http",
                        "https",
                    },
                },
            } in error.errors()

    def test_api_url(self):
        config = PrometheusConfiguration(
            base_url="http://prometheus.default.svc.cluster.local:9090", metrics=[]
        )
        assert (
            config.api_url == "http://prometheus.default.svc.cluster.local:9090/api/v1"
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
            "description: Update the base_url and metrics to match your Prometheus configuration\n"
            "base_url: http://prometheus:9090\n"
            "metrics:\n"
            "- name: throughput\n"
            "  unit: rps\n"
            "  query: rate(http_requests_total[5m])\n"
            "  step: 1m\n"
            "- name: error_rate\n"
            "  unit: '%'\n"
            "  query: rate(errors[5m])\n"
            "  step: 1m\n"
            "targets: null\n"
        )


class TestPrometheusRequest:
    @freeze_time("2020-01-01")
    def test_url(self):
        request = PrometheusRequest(
            base_url="http://prometheus.default.svc.cluster.local:9090/api/v1/",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=PrometheusMetric(
                "go_memstats_heap_inuse_bytes",
                Unit.BYTES,
                query="go_memstats_heap_inuse_bytes",
            ),
        )
        assert (
            request.url
            == "http://prometheus.default.svc.cluster.local:9090/api/v1/query_range?query=go_memstats_heap_inuse_bytes&start=1577836800.0&end=1577966400.0&step=1m"
        )

    @freeze_time("2020-01-01")
    def test_other_url(self):
        request = PrometheusRequest(
            base_url="http://localhost:9090/api/v1/",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=PrometheusMetric(
                "go_memstats_heap_inuse_bytes",
                Unit.BYTES,
                query="go_memstats_heap_inuse_bytes",
            ),
        )
        assert (
            request.url
            == "http://localhost:9090/api/v1/query_range?query=go_memstats_heap_inuse_bytes&start=1577836800.0&end=1577966400.0&step=1m"
        )


class TestPrometheusConnector:
    async def test_describe(self) -> None:
        pass

    async def test_measure(self) -> None:
        pass

    async def test_metrics(self) -> None:
        # TODO: This was broken because of the @property
        # TODO: should I figure out how to allow properties as event handlers?
        # TODO: Add mocks to stub out
        pass

    async def test_check(self) -> None:
        pass


# @pytest.mark.integration
# class TestPrometheusIntegration:
#     async def test_check_targets(self) -> None:
#         config = PrometheusConfiguration.generate(base_url="http://localhost:9090")
#         optimizer = servo.Optimizer("test.com/foo", token="12345")
#         debug(config, optimizer)
#         connector = PrometheusConnector(config=config, optimizer=optimizer)
#         checks = await connector.check()
#         debug(checks)

# @pytest.fixture
def envoy_sidecars() -> dict:
    return {
        "status": "success",
        "data": {
            "activeTargets": [
                {
                    "discoveredLabels": {
                        "__address__": "192.168.95.123:9901",
                        "__meta_kubernetes_namespace": "default",
                        "__meta_kubernetes_pod_annotation_kubectl_kubernetes_io_restartedAt": "2020-08-31T04:10:38-07:00",
                        "__meta_kubernetes_pod_annotation_kubernetes_io_psp": "eks.privileged",
                        "__meta_kubernetes_pod_annotation_prometheus_opsani_com_path": "/stats/prometheus",
                        "__meta_kubernetes_pod_annotation_prometheus_opsani_com_port": "9901",
                        "__meta_kubernetes_pod_annotation_prometheus_opsani_com_scrape": "true",
                        "__meta_kubernetes_pod_annotationpresent_kubectl_kubernetes_io_restartedAt": "true",
                        "__meta_kubernetes_pod_annotationpresent_kubernetes_io_psp": "true",
                        "__meta_kubernetes_pod_annotationpresent_prometheus_opsani_com_path": "true",
                        "__meta_kubernetes_pod_annotationpresent_prometheus_opsani_com_port": "true",
                        "__meta_kubernetes_pod_annotationpresent_prometheus_opsani_com_scrape": "true",
                        "__meta_kubernetes_pod_container_init": "false",
                        "__meta_kubernetes_pod_container_name": "envoy",
                        "__meta_kubernetes_pod_container_port_name": "metrics",
                        "__meta_kubernetes_pod_container_port_number": "9901",
                        "__meta_kubernetes_pod_container_port_protocol": "TCP",
                        "__meta_kubernetes_pod_controller_kind": "ReplicaSet",
                        "__meta_kubernetes_pod_controller_name": "web-6f756468f6",
                        "__meta_kubernetes_pod_host_ip": "192.168.92.91",
                        "__meta_kubernetes_pod_ip": "192.168.95.123",
                        "__meta_kubernetes_pod_label_app": "web",
                        "__meta_kubernetes_pod_label_pod_template_hash": "6f756468f6",
                        "__meta_kubernetes_pod_labelpresent_app": "true",
                        "__meta_kubernetes_pod_labelpresent_pod_template_hash": "true",
                        "__meta_kubernetes_pod_name": "web-6f756468f6-w96f2",
                        "__meta_kubernetes_pod_node_name": "ip-192-168-92-91.us-east-2.compute.internal",
                        "__meta_kubernetes_pod_phase": "Running",
                        "__meta_kubernetes_pod_ready": "true",
                        "__meta_kubernetes_pod_uid": "c80a750c-773b-4c27-abe0-45d53a782781",
                        "__metrics_path__": "/metrics",
                        "__scheme__": "http",
                        "job": "opsani-envoy-sidecars",
                    },
                    "labels": {
                        "app": "web",
                        "instance": "192.168.95.123:9901",
                        "job": "opsani-envoy-sidecars",
                        "pod_template_hash": "6f756468f6",
                    },
                    "scrapePool": "opsani-envoy-sidecars",
                    "scrapeUrl": "http://192.168.95.123:9901/stats/prometheus",
                    "globalUrl": "http://192.168.95.123:9901/stats/prometheus",
                    "lastError": "",
                    "lastScrape": "2020-09-09T10:04:02.662498189Z",
                    "lastScrapeDuration": 0.013974479,
                    "health": "up",
                }
            ]
        },
    }


class TestPrometheusChecks:
    @pytest.fixture
    def metric(self) -> PrometheusMetric:
        return PrometheusMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE,
            query="throughput",
            step="45m",
        )

    @pytest.fixture
    def go_memstats_gc_sys_bytes(self) -> dict:
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "__name__": "go_memstats_gc_sys_bytes",
                            "instance": "localhost:9090",
                            "job": "prometheus",
                        },
                        "values": [
                            [1595142421.024, "3594504"],
                            [1595142481.024, "3594504"],
                        ],
                    }
                ],
            },
        }

    @pytest.fixture
    def mocked_api(self, go_memstats_gc_sys_bytes):
        with respx.mock(
            base_url="http://localhost:9090", assert_all_called=False
        ) as respx_mock:
            respx_mock.get("/api/v1/targets", alias="targets", content=[])

            # re.compile(r"/api/v1/query_range/\w+")
            respx_mock.get(
                re.compile(r"/api/v1/query_range.+"),
                alias="query",
                content=go_memstats_gc_sys_bytes,
            )
            # respx_mock.get("/api/v1/query_range", alias="query", content=go_memstats_gc_sys_bytes)
            yield respx_mock

    @pytest.fixture
    def checks(self, metric) -> PrometheusChecks:
        config = PrometheusConfiguration(
            base_url="http://localhost:9090", metrics=[metric]
        )
        return PrometheusChecks(config=config)

    async def test_check_base_url(self, mocked_api, checks) -> None:
        request = mocked_api["targets"]
        check = await checks.check_base_url()
        assert request.called
        assert check
        assert check.name == 'Connect to "http://localhost:9090"'
        assert check.id == "check_base_url"
        assert check.critical
        assert check.success
        assert check.message is None

    async def test_check_base_url_failing(self, checks) -> None:
        with respx.mock(base_url="http://localhost:9090") as respx_mock:
            request = respx_mock.get("/api/v1/targets", status_code=503)
            check = await checks.check_base_url()
            assert request.called
            assert check
            assert check.name == 'Connect to "http://localhost:9090"'
            assert check.id == "check_base_url"
            assert check.critical
            assert not check.success
            assert check.message is not None
            assert isinstance(check.exception, httpx.HTTPStatusError)

    @respx.mock
    async def test_check_queries(self, mocked_api, checks) -> None:
        request = mocked_api["query"]
        multichecks = await checks._expand_multichecks()
        check = await multichecks[0]()
        assert request.called
        assert check
        assert check.name == 'Run query "throughput"'
        assert check.id == "check_queries_item_0"
        assert not check.critical
        assert check.success
        assert check.message == "returned 2 results"

    @pytest.mark.parametrize(
        "targets, success, message",
        [
            (
                {"status": "success", "data": {"activeTargets": []}},
                False,
                "caught exception: no targets are being scraped by Prometheus",
            ),
            (envoy_sidecars(), True, "found 1 targets"),
        ],
    )
    @respx.mock
    async def test_check_targets(self, checks, targets, success, message) -> str:
        with respx.mock(base_url="http://localhost:9090") as respx_mock:
            request = respx_mock.get("/api/v1/targets", content=targets)
            check = await checks.check_targets()
            assert request.called
            assert check
            assert check.name == "Active targets"
            assert check.id == "check_targets"
            assert not check.critical
            assert check.success == success
            assert check.message == message
