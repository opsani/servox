import datetime
import re

import httpx
import pytest
import respx
from freezegun import freeze_time
from pydantic import ValidationError

from servo.connectors.wavefront import WavefrontChecks, WavefrontConfiguration, WavefrontMetric, WavefrontRequest
from servo.types import *


class TestWavefrontMetric():
    def test_accepts_granularity_as_alpha(self):
        metric = WavefrontMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE_WF,
            query='rate(ts("heapster.node.network.tx", cluster="idps-preprod-west2.cluster.k8s.local"))',
            granularity="m",
        )
        assert metric.granularity.isalpha()

    def test_accepts_summarization_as_alpha(self):
        metric = WavefrontMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE_WF,
            query='rate(ts("heapster.node.network.tx", cluster="idps-preprod-west2.cluster.k8s.local"))',
            summarization="LAST",
        )
        assert metric.summarization.isalpha()

    # Query
    def test_query_required(self):
        try:
            WavefrontMetric(
                name="throughput", unit=Unit.REQUESTS_PER_MINUTE, query=None
            )
        except ValidationError as error:
            assert {
                "loc": ("query",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()


class TestWavefrontConfiguration:
    def test_url_required(self):
        try:
            WavefrontConfiguration(base_url=None)
        except ValidationError as error:
            assert {
                "loc": ("base_url",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    def test_base_url_is_rstripped(self):
        config = WavefrontConfiguration(
            base_url="http://wavefront.com/some/path/", metrics=[]
        )
        assert config.base_url == "http://wavefront.com/some/path"

    def test_supports_localhost_url(self):
        config = WavefrontConfiguration(base_url="http://localhost:2878", metrics=[])
        assert config.base_url == "http://localhost:2878"

    def test_supports_cluster_url(self):
        config = WavefrontConfiguration(
            base_url="http://wavefront.com:2878", metrics=[]
        )
        assert config.base_url == "http://wavefront.com:2878"

    def test_rejects_invalid_url(self):
        try:
            WavefrontConfiguration(base_url="gopher://this-is-invalid")
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
        config = WavefrontConfiguration(
            base_url="http://wavefront.com:2878", metrics=[]
        )
        assert (
            config.api_url == "http://wavefront.com:2878/api/v2"
        )

    # Metrics
    def test_metrics_required(self):
        try:
            WavefrontConfiguration(metrics=None)
        except ValidationError as error:
            assert {
                "loc": ("metrics",),
                "msg": "none is not an allowed value",
                "type": "type_error.none.not_allowed",
            } in error.errors()

    # Generation
    def test_generate_default_config(self):
        config = WavefrontConfiguration.generate()
        assert config.yaml() == (
            "base_url: http://wavefront:2878\n"
            "description: Update the base_url and metrics to match your Wavefront configuration\n"
            "metrics:\n"
            "- name: throughput\n"
            "  query: avg(ts(appdynamics.apm.overall.calls_per_min, env=foo and app=my-app))\n"
            "  granularity: m\n"
            "  summarization: LAST\n"
            "  unit: request/m\n"
            "- name: error_rate\n"
            "  query: avg(ts(appdynamics.apm.transactions.errors_per_min, env=foo and app=my-app))\n"
            "  granularity: m\n"
            "  summarization: LAST\n"
            "  unit: errors/m\n"
            # "targets: null\n" (?)
        )


class TestWavefrontRequest:
    @freeze_time("2020-01-01")
    def test_url(self):
        request = WavefrontRequest(
            base_url="http://wavefront.com:2878/api/v2/",
            start=datetime.datetime.now(),
            end=datetime.datetime.now() + Duration("36h"),
            metric=WavefrontMetric(
                "throughput",
                servo.Unit.REQUESTS_PER_MINUTE_WF,
                query='rate(ts("heapster.node.network.tx", cluster="idps-preprod-west2.cluster.k8s.local"))',
                granularity="m",
                summarization="LAST"
            ),
        )
        assert (
            request.url
            == 'http://wavefront.com:2878/api/v2/chart/api?q=rate(ts("heapster.node.network.tx", cluster="idps-preprod-west2.cluster.k8s.local"))&s=1577836800&e=1577966400&g=m&summarization=LAST'
        )

# @pytest.mark.integration
# class TestWavefrontIntegration:
#     async def test_check_targets(self) -> None:
#         config = WavefrontConfiguration.generate(base_url="http://localhost:2878")
#         optimizer = servo.Optimizer("test.com/foo", token="12345")
#         debug(config, optimizer)
#         connector = WavefrontConnector(config=config, optimizer=optimizer)
#         checks = await connector.check()
#         debug(checks)


class TestWavefrontChecks:
    @pytest.fixture
    def metric(self) -> WavefrontMetric:
        return WavefrontMetric(
            name="test",
            unit=Unit.REQUESTS_PER_MINUTE_WF,
            query='rate(ts("heapster.node.network.tx", cluster="idps-preprod-west2.cluster.k8s.local"))',
            granularity="m",
        )

    @pytest.fixture
    def heapster_node_network_tx(self) -> dict:
        return {
            'granularity': 60,
            'name': 'rate(ts("heapster.node.network.tx", '
            'cluster="idps-preprod-west2.cluster.k8s.local"))',
            'query': 'rate(ts("heapster.node.network.tx", '
            'cluster="idps-preprod-west2.cluster.k8s.local"))',
            'stats': {'buffer_keys': 154,
                      'cached_compacted_keys': 0,
                      'compacted_keys': 24,
                      'compacted_points': 12440,
                      'cpu_ns': 36609718,
                      'distributions': 0,
                      'dropped_distributions': 0,
                      'dropped_edges': 0,
                      'dropped_metrics': 0,
                      'dropped_spans': 0,
                      'edges': 0,
                      'keys': 168,
                      'latency': 11,
                      'metrics': 12584,
                      'points': 12584,
                      'queries': 108,
                      'query_tasks': 0,
                      's3_keys': 0,
                      'skipped_compacted_keys': 22,
                      'spans': 0,
                      'summaries': 12584},
            'timeseries': [
                {'data': [[1604626020, 68441.23333333334],
                          [1604626080, 75125.6],
                          [1604626140, 59805.666666666664]],
                 'host': 'ip-10-131-115-108.us-west-2.compute.internal',
                 'label': 'heapster.node.network.tx',
                 'tags': {'cluster': 'idps-preprod-west2.cluster.k8s.local',
                          'label.beta.kubernetes.io/arch': 'amd64',
                          'label.beta.kubernetes.io/instance-type': 'm5.2xlarge',
                          'label.beta.kubernetes.io/os': 'linux',
                          'label.failure-domain.beta.kubernetes.io/region': 'us-west-2',
                          'label.failure-domain.beta.kubernetes.io/zone': 'us-west-2b',
                          'label.kops.k8s.io/instancegroup': 'iks-system',
                          'label.kubernetes.io/arch': 'amd64',
                          'label.kubernetes.io/hostname': 'ip-10-131-115-108.us-west-2.compute.internal',
                          'label.kubernetes.io/os': 'linux',
                          'label.kubernetes.io/role': 'node',
                          'label.node.kubernetes.io/instance-type': 'm5.2xlarge',
                          'label.topology.kubernetes.io/region': 'us-west-2',
                          'label.topology.kubernetes.io/zone': 'us-west-2b',
                          'nodename': 'ip-10-131-115-108.us-west-2.compute.internal',
                                      'type': 'node'}},
                {'data': [[1604626020, 33849.583333333336],
                          [1604626080, 48680.51666666667],
                          [1604626140, 34244.1]],
                 'host': 'ip-10-131-115-88.us-west-2.compute.internal',
                 'label': 'heapster.node.network.tx',
                 'tags': {'cluster': 'idps-preprod-west2.cluster.k8s.local',
                          'label.beta.kubernetes.io/arch': 'amd64',
                          'label.beta.kubernetes.io/instance-type': 'm5.2xlarge',
                          'label.beta.kubernetes.io/os': 'linux',
                          'label.failure-domain.beta.kubernetes.io/region': 'us-west-2',
                          'label.failure-domain.beta.kubernetes.io/zone': 'us-west-2b',
                          'label.kops.k8s.io/instancegroup': 'iks-system',
                          'label.kubernetes.io/arch': 'amd64',
                          'label.kubernetes.io/hostname': 'ip-10-131-115-88.us-west-2.compute.internal',
                          'label.kubernetes.io/os': 'linux',
                          'label.kubernetes.io/role': 'node',
                          'label.node.kubernetes.io/instance-type': 'm5.2xlarge',
                          'label.topology.kubernetes.io/region': 'us-west-2',
                          'label.topology.kubernetes.io/zone': 'us-west-2b',
                          'nodename': 'ip-10-131-115-88.us-west-2.compute.internal',
                                      'type': 'node'}}],
            'traceDimensions': []
        }

    @ pytest.fixture
    def mocked_api(self, heapster_node_network_tx):
        with respx.mock(
            base_url="http://localhost:2878", assert_all_called=False
        ) as respx_mock:
            respx_mock.get(
                re.compile(r"/api/v2/chart/api.+"),
                alias="query",
                content=heapster_node_network_tx,
            )
            yield respx_mock

    @ pytest.fixture
    def checks(self, metric) -> WavefrontChecks:
        config = WavefrontConfiguration(
            base_url="http://localhost:2878", metrics=[metric]
        )
        return WavefrontChecks(config=config)

    async def test_check_base_url(self, mocked_api, checks) -> None:
        request = mocked_api["targets"]
        check = await checks.check_base_url()
        assert request.called
        assert check
        assert check.name == 'Connect to "http://localhost:2878"'
        assert check.id == "check_base_url"
        assert check.critical
        assert check.success
        assert check.message is None

    async def test_check_base_url_failing(self, checks) -> None:
        with respx.mock(base_url="http://localhost:2878") as respx_mock:
            request = respx_mock.get("/api/v2/foobar", status_code=503)
            check = await checks.check_base_url()
            assert request.called
            assert check
            assert check.name == 'Connect to "http://localhost:2878"'
            assert check.id == "check_base_url"
            assert check.critical
            assert not check.success
            assert check.message is not None
            assert isinstance(check.exception, httpx.HTTPStatusError)

    @ respx.mock
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
