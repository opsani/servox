import datetime
import json
import pathlib
import re
from typing import AsyncIterator

import freezegun
import httpx
import pydantic
import pytest
import respx
import typer
from servo.connectors import prometheus

import servo.utilities
from servo.connectors.prometheus import (
    PrometheusChecks,
    PrometheusConfiguration,
    PrometheusConnector,
    PrometheusMetric,
    RangeQuery,
)
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
        except pydantic.ValidationError as error:
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
        except pydantic.ValidationError as error:
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
        except pydantic.ValidationError as error:
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
        except pydantic.ValidationError as error:
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
            "  absent: zero\n"
            "- name: error_rate\n"
            "  unit: '%'\n"
            "  query: rate(errors[5m])\n"
            "  step: 1m\n"
            "  absent: zero\n"
            "targets: null\n"
        )
    
    def test_generate_override_metrics(self):
        PrometheusConfiguration.generate(
            metrics=[
                PrometheusMetric(
                    "throughput",
                    servo.Unit.REQUESTS_PER_SECOND,
                    query="sum(rate(envoy_cluster_upstream_rq_total[1m]))",
                    absent=servo.connectors.prometheus.Absent.zero,
                    step="1m",
                ),
                PrometheusMetric(
                    "error_rate",
                    servo.Unit.PERCENTAGE,
                    query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[1m]))",
                    absent=servo.connectors.prometheus.Absent.zero,
                    step="1m",
                ),
            ],                
        )


class TestPrometheusRequest:
    @freezegun.freeze_time("2020-01-01")
    def test_url(self):
        request = RangeQuery(
            base_url="http://prometheus.default.svc.cluster.local:9090/api/v1/",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=PrometheusMetric(
                "go_memstats_heap_inuse_bytes",
                Unit.BYTES,
                query="go_memstats_heap_inuse_bytes",
            ),
        )
        assert request.base_url, "request base_url should not be nil"
        assert request.url, "request URL should not be nil"
        assert (
            request.url
            == "http://prometheus.default.svc.cluster.local:9090/api/v1/query_range?query=go_memstats_heap_inuse_bytes&start=1577836800.0&end=1577966400.0&step=1m"
        )

    @freezegun.freeze_time("2020-01-01")
    def test_other_url(self):
        request = RangeQuery(
            base_url="http://localhost:9090/api/v1/",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=PrometheusMetric(
                "go_memstats_heap_inuse_bytes",
                Unit.BYTES,
                query="go_memstats_heap_inuse_bytes",
            ),
        )
        
        assert request.url
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
            respx_mock.get(
                "/api/v1/targets",
                name="targets"
            ).mock(return_value=httpx.Response(200, json=[]))

            respx_mock.get(
                re.compile(r"/api/v1/query_range.+"),
                name="query",
            ).mock(return_value=httpx.Response(200, json=go_memstats_gc_sys_bytes))
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
            request = respx_mock.get("/api/v1/targets").mock(return_value=httpx.Response(status_code=503))
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
                "caught exception (AssertionError): no targets are being scraped by Prometheus",
            ),
            (envoy_sidecars(), True, "found 1 targets"),
        ],
    )
    @respx.mock
    async def test_check_targets(self, checks, targets, success, message) -> str:
        with respx.mock(base_url="http://localhost:9090") as respx_mock:
            request = respx_mock.get("/api/v1/targets").mock(httpx.Response(200, json=targets))
            check = await checks.check_targets()
            assert request.called
            assert check
            assert check.name == "Active targets"
            assert check.id == "check_targets"
            assert not check.critical
            assert check.success == success
            assert check.message == message


###
# Integration tests...
# Look at targets
# CLI on targets
# Targets with init container
# Querying for data that is null
# Querying for data that is partially null


@pytest.mark.integration
@pytest.mark.applymanifests(
    "../manifests",
    files=[
        "prometheus.yaml",
    ]
)
@pytest.mark.clusterrolebinding('cluster-admin')
class TestPrometheusIntegration:
    def optimizer(self) -> servo.Optimizer:
        return servo.Optimizer(
            id="dev.opsani.com/blake-ignite",
            token="bfcf94a6e302222eed3c73a5594badcfd53fef4b6d6a703ed32604",
        )

    async def test_check_targets(
        self,
        optimizer: servo.Optimizer,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    ) -> None:
        # Deploy Prometheus and take a look at the targets it starts scraping
        async with kube_port_forward("deploy/prometheus", 9090) as url:
            config = PrometheusConfiguration.generate(base_url=url)
            connector = PrometheusConnector(config=config, optimizer=optimizer)
            targets = await asyncio.wait_for(
                asyncio.gather(connector.targets()),
                timeout=10
            )
            debug(targets)
    
    async def test_target_discovery(self) -> None:
        # Deploy fiber-http with annotations and Prometheus will start scraping it
        ...

    @pytest.mark.clusterrolebinding('cluster-admin')
    @pytest.mark.applymanifests(
        "../manifests",
        files=[
            "fiber-http-opsani-dev.yaml",
            "k6.yaml"
        ],
    )
    async def test_reactive_measurement(
        self,
        optimizer: servo.Optimizer,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    ) -> None:
        # NOTE: What we are going to do here is deploy Prometheus, fiber-http, and k6 on a k8s cluster,
        # port forward so we can talk to them, and then spark up the connector and it will adapt to 
        # changes in traffic. If it holds steady for 1 minute, it will early report. This supports bursty traffic.
        # Measurements are taken based on the `step` of the target metric (currently hardwired to `throughput`).
        servo.logging.set_level("DEBUG")
        # FIXME: Absent.zero mode needs to URL escape the query string
        async with kube_port_forward("deploy/prometheus", 9090) as url:
            config = PrometheusConfiguration.generate(
                base_url=url,
                metrics=[
                    PrometheusMetric(
                        "throughput",
                        servo.Unit.REQUESTS_PER_SECOND,
                        query="sum(rate(envoy_cluster_upstream_rq_total[15s]))",
                        # absent=servo.connectors.prometheus.Absent.zero,
                        step="15s",
                    ),
                    PrometheusMetric(
                        "error_rate",
                        servo.Unit.PERCENTAGE,
                        query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[1m]))",
                        # absent=servo.connectors.prometheus.Absent.zero,
                        step="1m",
                    ),
                ],                
            )
            connector = PrometheusConnector(config=config, optimizer=optimizer)
            measurement = await asyncio.wait_for(
                connector.measure(control=servo.Control(duration="3m")),
                timeout=240 # NOTE: Always make timeout exceed control duration
            )
            debug(measurement)
    
    # TODO: Test no traffic -- no k6, timeout at the end and return an empty measure set
    @pytest.mark.clusterrolebinding('cluster-admin')
    @pytest.mark.applymanifests(
        "../manifests",
        files=[
            "fiber-http-opsani-dev.yaml",
        ],
    )
    async def test_no_traffic(
        self,
        optimizer: servo.Optimizer,
        kube,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    ) -> None:
        # NOTE: What we are going to do here is deploy Prometheus, fiber-http, and k6 on a k8s cluster,
        # port forward so we can talk to them, and then spark up the connector and it will adapt to 
        # changes in traffic. If it holds steady for 1 minute, it will early report. This supports bursty traffic.
        # Measurements are taken based on the `step` of the target metric (currently hardwired to `throughput`).
        servo.logging.set_level("DEBUG")
        kube.wait_for_registered(timeout=30)
        
        # FIXME: Absent.zero mode needs to URL escape the query string
        # async with kube_port_forward(
        #     {
        #         "deploy/prometheus": 9090,
        #         "service/fiber-http": 80
        #     }
        # ) as urls:  # TODO: urls should be a dict        
        async with kube_port_forward("deploy/prometheus", 9090) as prometheus_url:
            async with kube_port_forward("service/fiber-http", 80) as fiber_url:
                config = PrometheusConfiguration.generate(
                    base_url=prometheus_url,
                    metrics=[
                        PrometheusMetric(
                            "throughput",
                            servo.Unit.REQUESTS_PER_SECOND,
                            query="sum(rate(envoy_cluster_upstream_rq_total[15s]))",
                            # absent=servo.connectors.prometheus.Absent.zero,
                            step="15s",
                        ),
                        PrometheusMetric(
                            "error_rate",
                            servo.Unit.PERCENTAGE,
                            query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[1m]))",
                            # absent=servo.connectors.prometheus.Absent.zero,
                            step="10s",
                        ),
                    ],                
                )
                connector = PrometheusConnector(config=config, optimizer=optimizer)
                measurement = await asyncio.wait_for(
                    connector.measure(control=servo.Control(duration="1m")),
                    timeout=70 # NOTE: Always make timeout exceed control duration
                )
                assert measurement
                debug(measurement)
                # FIXME: This should have error_rate also
    
    
    # TODO: Test burst -- no k6, pump requests to fiber directly
    @pytest.mark.clusterrolebinding('cluster-admin')
    @pytest.mark.applymanifests(
        "../manifests",
        files=[
            "fiber-http-opsani-dev.yaml",
        ],
    )
    async def test_bursty_traffic(
        self,
        optimizer: servo.Optimizer,
        event_loop: asyncio.AbstractEventLoop,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],        
    ) -> None:
        # NOTE: What we are going to do here is deploy Prometheus and fiber-http on a k8s cluster,        
        # port forward so we can talk to them, and then spark up the connector and it will adapt to 
        # changes in traffic. 
        # 
        # In this scenario, we will let the connector begin collecting metrics with zero traffic
        # and then manually burst it with traffic via httpx, wait for the metrics to begin flowing,
        # then suspend the traffic and let it fall back to zero. If all goes well, the connector will
        # detect this change and enter into the 1 minute settlement time, early return, and report a
        # set of readings that includes the traffic burst, the zero readings on either side of the
        # burst, and will early return once the metrics stabilize without waiting for the full
        # measurement duration as prescribed by the control structure (13 minutes).
        servo.logging.set_level("DEBUG")
        async with kube_port_forward("deploy/prometheus", 9090) as prometheus_url:
            async with kube_port_forward("service/fiber-http", 80) as fiber_url:
                config = PrometheusConfiguration.generate(
                    base_url=prometheus_url,
                    metrics=[
                        PrometheusMetric(
                            "throughput",
                            servo.Unit.REQUESTS_PER_SECOND,
                            query="sum(rate(envoy_cluster_upstream_rq_total[15s]))",
                            step="15s",
                        ),
                        PrometheusMetric(
                            "error_rate",
                            servo.Unit.PERCENTAGE,
                            query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[1m]))",
                            step="1m",
                        ),
                    ],
                )
                
                async def burst_traffic() -> None:
                    burst_until = datetime.datetime.now() + datetime.timedelta(seconds=45)                    
                    async with httpx.AsyncClient(base_url=fiber_url) as client:
                        servo.logger.info(f"Bursting traffic to {fiber_url} for 45 seconds...")
                        count = 0
                        while datetime.datetime.now() < burst_until:
                            response = await client.get("/")
                            response.raise_for_status()
                            count += 1
                        servo.logger.success(f"Bursted {count} requests to {fiber_url} over 45 seconds.")
                
                connector = PrometheusConnector(config=config, optimizer=optimizer)
                event_loop.call_later(
                    15, 
                    asyncio.create_task, 
                    burst_traffic()
                )
                measurement = await asyncio.wait_for(
                    connector.measure(control=servo.Control(duration="13m")),
                    timeout=300 # NOTE: if we haven't returned in 5 minutes all is lost
                )
                assert measurement
                debug("Finished testing burst traffic scenario: ", measurement)
                # TODO: Check the readings on both sides

    @pytest.mark.applymanifests(
        "../manifests",
        files=[
            "fiber-http-opsani-dev.yaml",

        ],
    )
    async def test_load_testing(
        self,
        optimizer: servo.Optimizer,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    ) -> None:
        # Deploy fiber-http with annotations and Prometheus will start scraping it
        async with kube_port_forward("deploy/prometheus", 9090) as url:
            config = PrometheusConfiguration.generate(base_url=url)
            connector = PrometheusConnector(config=config, optimizer=optimizer)
            metrics = await asyncio.wait_for(
                asyncio.gather(connector.measure(control=servo.Control(duration="5s"))),
                timeout=30
            )
            debug(metrics)


def empty_targets_response() -> Dict[str, Any]:
    return json.load("{'status': 'success', 'data': {'activeTargets': [], 'droppedTargets': []}}")

class TestCLI:
    class TestTargets:
        async def test_no_active_connectors(self) -> None:
            # TODO: Put config into tmpdir without connector
            ...

        @pytest.fixture
        def metric(self) -> PrometheusMetric:
            return PrometheusMetric(
                name="test",
                unit=Unit.REQUESTS_PER_MINUTE,
                query="throughput",
                step="45m",
            )

        @pytest.fixture
        def config(self, metric: PrometheusMetric) -> PrometheusConfiguration:
            return PrometheusConfiguration(
                base_url="http://localhost:9090", metrics=[metric]
            )

        @pytest.fixture
        def connector(self, config: PrometheusConfiguration) -> PrometheusConnector:
            return PrometheusConnector(config=config)

        # TODO: This needs to be an integration test
        def test_one_active_connector(self, optimizer_env: None, connector: PrometheusConnector, config: PrometheusConfiguration, servo_cli: servo.cli.ServoCLI, cli_runner: typer.testing.CliRunner, tmp_path: pathlib.Path) -> None:
            with respx.mock(base_url="http://localhost:9090") as respx_mock:
                targets = envoy_sidecars()
                request = respx_mock.get("/api/v1/targets").mock(httpx.Response(200, json=targets))
                
                config_file = tmp_path / "servo.yaml"
                import tests.helpers # TODO: Turn into fixtures!
                tests.helpers.write_config_yaml({"prometheus": config}, config_file)
                
                result = cli_runner.invoke(servo_cli, "prometheus targets")
                assert result.exit_code == 0, f"expected exit status 0, got {result.exit_code}: stdout={result.stdout}, stderr={result.stderr}"
                assert request.called
                assert "opsani-envoy-sidecars  up        http://192.168.95.123:9901/stats/prometheus" in result.stdout
            

        async def test_multiple_active_connector(self) -> None:
            # TODO: Put config into tmpdir with two connectors, invoke both, invoke each one
            ...

    class TestQuery:
        ...
# CLI TESTS:
# Test without active target
# Test with multiple targets
# Tests with specific target
# TODO: Add query CLI


class TestConnector:
    @pytest.fixture
    def metric(self) -> PrometheusMetric:
        return PrometheusMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE,
            query="throughput",
            step="45m",
        )

    @pytest.fixture
    def config(self, metric: PrometheusMetric) -> PrometheusConfiguration:
        return PrometheusConfiguration(
            base_url="http://localhost:9090", metrics=[metric]
        )

    @pytest.fixture
    def connector(self, config: PrometheusConfiguration) -> PrometheusConnector:
        return PrometheusConnector(config=config)
        
    # TODO: Wrap into parsing logic
    async def test_one_active_connector(self, connector: PrometheusConnector) -> None:
        with respx.mock(base_url="http://localhost:9090") as respx_mock:
            targets = envoy_sidecars()            
            request = respx_mock.get("/api/v1/targets").mock(return_value=httpx.Response(200, json=targets))
            targets = await connector.targets()
            assert request.called
            assert len(targets) == 1
            assert targets[0].pool == 'opsani-envoy-sidecars'
            assert targets[0].url == 'http://192.168.95.123:9901/stats/prometheus'
            assert targets[0].health == 'up'
            
            import timeago
            import pytz

            utc_now = pytz.utc.localize(datetime.datetime.utcnow())
            debug(datetime.datetime.now(), datetime.datetime.utcnow(), utc_now)
            debug(timeago.format(targets[0].last_scraped_at, utc_now))


class TestInstantVector:
    @pytest.fixture
    def vector(self) -> servo.connectors.prometheus.InstantVector:
        return pydantic.parse_obj_as(
            servo.connectors.prometheus.InstantVector, 
            {
                'metric': {},
                'value': [
                    1607989427.782,
                    '19.8',
                ],
            }
        )
        
    def test_parse(self, vector) -> None:
        assert vector
        assert vector.metric == {}
        assert vector.value == (
            datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc),
            19.8
        )
    
    def test_len(self, vector) -> None:
        assert vector
        assert len(vector) == 1
        
    def test_iterate(self, vector) -> None:
        assert vector
        for sample in vector:
            assert sample == (
                datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc),
                19.8
            )

class TestRangeVector:
    @pytest.fixture
    def vector(self) -> servo.connectors.prometheus.RangeVector:
        return pydantic.parse_obj_as(
            servo.connectors.prometheus.RangeVector, 
            {
                "metric": {
                    "__name__": "go_memstats_gc_sys_bytes",
                    "instance": "localhost:9090",
                    "job": "prometheus",
                },
                "values": [
                    [1595142421.024, "3594504"],
                    [1595142481.024, "3594504"],
                    [1595152585.055, "31337"],
                ],
            }
        )
        
    def test_parse(self, vector) -> None:
        assert vector
        assert vector.metric == {
            "__name__": "go_memstats_gc_sys_bytes",
            "instance": "localhost:9090",
            "job": "prometheus",
        }
        assert vector.values == [
            (datetime.datetime(2020, 7, 19, 7, 7, 1, 24000, tzinfo=datetime.timezone.utc), 3594504.0),
            (datetime.datetime(2020, 7, 19, 7, 8, 1, 24000, tzinfo=datetime.timezone.utc), 3594504.0),
            (datetime.datetime(2020, 7, 19, 9, 56, 25, 55000, tzinfo=datetime.timezone.utc), 31337.0),
        ]
    
    def test_len(self, vector) -> None:
        assert vector
        assert len(vector) == 3
        
    def test_iterate(self, vector) -> None:
        assert vector
        expected = [
            (datetime.datetime(2020, 7, 19, 7, 7, 1, 24000, tzinfo=datetime.timezone.utc), 3594504.0),
            (datetime.datetime(2020, 7, 19, 7, 8, 1, 24000, tzinfo=datetime.timezone.utc), 3594504.0),
            (datetime.datetime(2020, 7, 19, 9, 56, 25, 55000, tzinfo=datetime.timezone.utc), 31337.0),
        ]
        for sample in vector:
            assert sample == expected.pop(0)
        assert not expected
        
class TestResultPrimitives:    
    def test_scalar(self) -> None:
        output = pydantic.parse_obj_as(servo.connectors.prometheus.Scalar, [1607989427.782, '1234'])
        assert output
        assert output == (
            datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc),
            1234.0
        )
    
    def test_string(self) -> None:
        output = pydantic.parse_obj_as(servo.connectors.prometheus.String, [1607989427.782, 'whatever'])
        assert output
        assert output == (
            datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc),
            'whatever'
        )
    
    def test_scalar_parses_as_string(self) -> None:
        output = pydantic.parse_obj_as(servo.connectors.prometheus.String, [1607989427.782, '1234.56'])
        assert output
        assert output == (
            datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc),
            '1234.56'
        )
    
    def test_string_does_not_parse_as_scalar(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="value is not a valid float"):
            pydantic.parse_obj_as(servo.connectors.prometheus.Scalar, [1607989427.782, 'thug_life'])


class TestData:
    class TestVector:
        @pytest.fixture
        def obj(self):
            return {
                "resultType" : "vector",
                "result" : [
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "prometheus",
                            "instance" : "localhost:9090"
                        },
                        "value": [ 1435781451.781, "1" ]
                    },
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "node",
                            "instance" : "localhost:9100"
                        },
                        "value" : [ 1435781451.781, "0" ]
                    }
                ]
            }
            
        def test_parse(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            assert data.type == servo.connectors.prometheus.ResultType.vector
            assert len(data) == 2
        
        def test_iterate(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            for vector in data:
                assert isinstance(vector, servo.connectors.prometheus.InstantVector)
                assert vector.metric["__name__"] == "up"
                assert vector.value[0] == datetime.datetime(2015, 7, 1, 20, 10, 51, 781000, tzinfo=datetime.timezone.utc)
    
    class TestMatrix:
        @pytest.fixture
        def obj(self):
            return {
                "resultType" : "matrix",
                "result" : [
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "prometheus",
                            "instance" : "localhost:9090"
                        },
                        "values" : [
                            [ 1435781430.781, "1" ],
                            [ 1435781445.781, "1" ],
                            [ 1435781460.781, "1" ]
                        ]
                    },
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "node",
                            "instance" : "localhost:9091"
                        },
                        "values" : [
                            [ 1435781430.781, "0" ],
                            [ 1435781445.781, "0" ],
                            [ 1435781460.781, "1" ]
                        ]
                    }
                ]
            }
            
        def test_parse(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            assert data.type == servo.connectors.prometheus.ResultType.matrix
            assert len(data) == 2
        
        def test_iterate(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            
            values = [
                [(datetime.datetime(2015, 7, 1, 20, 10, 30, 781000, tzinfo=datetime.timezone.utc), 1.0,),
                 (datetime.datetime(2015, 7, 1, 20, 10, 45, 781000, tzinfo=datetime.timezone.utc), 1.0,),
                 (datetime.datetime(2015, 7, 1, 20, 11, 0, 781000, tzinfo=datetime.timezone.utc), 1.0,),], 
                [(datetime.datetime(2015, 7, 1, 20, 10, 30, 781000, tzinfo=datetime.timezone.utc), 0.0,),
                 (datetime.datetime(2015, 7, 1, 20, 10, 45, 781000, tzinfo=datetime.timezone.utc), 0.0,),
                 (datetime.datetime(2015, 7, 1, 20, 11, 0, 781000, tzinfo=datetime.timezone.utc), 1.0,),]
            ]
            for vector in data:
                assert isinstance(vector, servo.connectors.prometheus.RangeVector)
                assert vector.metric["__name__"] == "up"
                assert vector.values == values.pop(0)
            
            assert not values

    class TestScalar:
        @pytest.fixture
        def obj(self):
            return {
                "resultType" : "scalar",
                "result" : [1435781460.781, "1"]
            }
            
        def test_parse(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            assert data.type == servo.connectors.prometheus.ResultType.scalar
            assert len(data) == 1
        
        def test_iterate(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            for scalar in data:
                assert scalar[0] == datetime.datetime(2015, 7, 1, 20, 11, 0, 781000, tzinfo=datetime.timezone.utc)
                assert scalar[1] == 1.0
    
    class TestString:
        @pytest.fixture
        def obj(self):
            return {
                "resultType" : "string",
                "result" : [1607989427.782, 'thug_life']
            }
            
        def test_parse(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            assert data.type == servo.connectors.prometheus.ResultType.string
            assert len(data) == 1
        
        def test_iterate(self, obj) -> None:
            data = servo.connectors.prometheus.Data.parse_obj(obj)
            assert data
            for string in data:
                assert string[0] == datetime.datetime(2020, 12, 14, 23, 43, 47, 782000, tzinfo=datetime.timezone.utc)
                assert string[1] == "thug_life"


@pytest.fixture
def config() -> PrometheusConfiguration:
    return PrometheusConfiguration(
        base_url="http://prometheus.io/some/path/", metrics=[]
    )
        
class TestResponse:
    def test_parsing_error(self, config) -> None:
        obj = {
            "status": "error",
            "errorType": "failure", 
            "error": "couldn't hang",
            "data": None,
        }
        metric = servo.connectors.prometheus.PrometheusMetric(
            "tuning_request_rate",
            servo.types.Unit.REQUESTS_PER_SECOND,
            query='rate(envoy_cluster_upstream_rq_total{opsani_role="tuning"}[10s])',
            step="10s"
        )        
        query = servo.connectors.prometheus.InstantQuery(
            base_url=config.base_url,
            metric=metric
        )
        response = servo.connectors.prometheus.Response.parse_obj(dict(query=query, **obj))
        assert response.status == "error"
        assert response.error == { "type": "failure", "message": "couldn't hang" }
    
    def test_parsing_vector_result(self, query) -> None:
        obj = {
            "status" : "success",
            "data" : {
                "resultType" : "vector",
                "result" : [
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "prometheus",
                            "instance" : "localhost:9090"
                        },
                        "value": [ 1435781451.781, "1" ]
                    },
                    {
                        "metric" : {
                            "__name__" : "up",
                            "job" : "node",
                            "instance" : "localhost:9100"
                        },
                        "value" : [ 1435781451.781, "0" ]
                    }
                ]
            }
        }
        response = servo.connectors.prometheus.Response.parse_obj(dict(query=query, **obj))
        assert response.status == "success"
        assert response.error is None
        assert response.warnings is None

    
    def test_parsing_matrix_result(self, config, query) -> None:
        obj = {
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
        if response := servo.connectors.prometheus.Response.parse_obj(dict(query=query, **obj)):
            assert response.status == "success"
            assert response.error is None
            assert response.warnings is None

class TestError:
    @pytest.fixture
    def data(self) -> Dict[str, str]:
        return { "errorType": "failure", "error": "couldn't hang" }

    def test_parsing(self, data) -> None:
        error = servo.connectors.prometheus.Error.parse_obj(data)
        assert error
        assert error.type == "failure"
        assert error.message == "couldn't hang"
    
    def test_cannot_parse_empty(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            servo.connectors.prometheus.Error.parse_obj({})





@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.applymanifests(
    "../manifests",
    files=[
        "fiber-http-opsani-dev.yaml",
        "prometheus.yaml"
    ],
)
@pytest.mark.integration
async def test_kubetest(
    optimizer: servo.Optimizer,
    kube,
    kube_port_forward: Callable[[str, int], AsyncIterator[str]],
) -> None:
    # NOTE: What we are going to do here is deploy Prometheus, fiber-http, and k6 on a k8s cluster,
    # port forward so we can talk to them, and then spark up the connector and it will adapt to 
    # changes in traffic. If it holds steady for 1 minute, it will early report. This supports bursty traffic.
    # Measurements are taken based on the `step` of the target metric (currently hardwired to `throughput`).
    servo.logging.set_level("DEBUG")
    kube.wait_for_registered(timeout=30)
    
    async with kube_port_forward("deploy/prometheus", 9090) as prometheus_url:
        async with kube_port_forward("service/fiber-http", 80) as fiber_url:
            config = PrometheusConfiguration.generate(
                base_url=prometheus_url,
                metrics=[
                    PrometheusMetric(
                        "throughput",
                        servo.Unit.REQUESTS_PER_SECOND,
                        query="sum(rate(envoy_cluster_upstream_rq_total[15s]))",
                        # absent=servo.connectors.prometheus.Absent.zero,
                        step="15s",
                    ),
                    PrometheusMetric(
                        "error_rate",
                        servo.Unit.PERCENTAGE,
                        query="sum(rate(envoy_cluster_upstream_rq_xx{opsani_role!=\"tuning\", envoy_response_code_class=~\"4|5\"}[1m]))",
                        # absent=servo.connectors.prometheus.Absent.zero,
                        step="10s",
                    ),
                ],                
            )
            connector = PrometheusConnector(config=config, optimizer=optimizer)
            measurement = await asyncio.wait_for(
                connector.measure(control=servo.Control(duration="1m")),
                timeout=70 # NOTE: Always make timeout exceed control duration
            )
            assert measurement
            # FIXME: This should have error_rate also
    
    
@pytest.fixture
def query(config):
    metric = servo.connectors.prometheus.PrometheusMetric(
        "tuning_request_rate",
        servo.types.Unit.REQUESTS_PER_SECOND,
        query='rate(envoy_cluster_upstream_rq_total{opsani_role="tuning"}[10s])',
        step="10s"
    )        
    return servo.connectors.prometheus.InstantQuery(
        base_url=config.base_url,
        metric=metric
    )
    