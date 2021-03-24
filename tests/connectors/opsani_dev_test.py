import asyncio
import contextlib
import datetime
import functools
import os
import pathlib
import re
from typing import (
    Any,
    AsyncContextManager,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

import devtools
import httpx
import kubernetes_asyncio
import pydantic
import pytest
import pytz
import respx

import servo
import servo.cli
import servo.connectors.kubernetes
import servo.connectors.opsani_dev
import servo.connectors.prometheus


@pytest.fixture
def config(kube) -> servo.connectors.opsani_dev.OpsaniDevConfiguration:
    return servo.connectors.opsani_dev.OpsaniDevConfiguration(
        namespace=kube.namespace,
        deployment="fiber-http",
        container="fiber-http",
        service="fiber-http",
        cpu=servo.connectors.kubernetes.CPU(min="125m", max="4000m", step="125m"),
        memory=servo.connectors.kubernetes.Memory(min="128 MiB", max="4.0 GiB", step="128 MiB"),
        __optimizer__=servo.configuration.Optimizer(id="test.com/foo", token="12345")
    )


@pytest.fixture
def checks(config: servo.connectors.opsani_dev.OpsaniDevConfiguration) -> servo.connectors.opsani_dev.OpsaniDevChecks:
    return servo.connectors.opsani_dev.OpsaniDevChecks(config=config)

class TestConfig:
    def test_generate(self) -> None:
        config = servo.connectors.opsani_dev.OpsaniDevConfiguration.generate()
        assert list(config.dict().keys()) == ['description', 'namespace', 'deployment', 'container', 'service', 'port', 'cpu', 'memory', 'prometheus_base_url']

    def test_generate_yaml(self) -> None:
        config = servo.connectors.opsani_dev.OpsaniDevConfiguration.generate()
        assert config.yaml(exclude_unset=True) == (
            "namespace: default\n"
            "deployment: app-deployment\n"
            "container: main\n"
            "service: app\n"
            "cpu:\n"
            "  min: 250m\n"
            "  max: '4'\n"
            "memory:\n"
            "  min: 256.0MiB\n"
            "  max: 4.0GiB\n"
        )

    def test_assign_optimizer(self) -> None:
        config = servo.connectors.opsani_dev.OpsaniDevConfiguration.generate()
        config.__optimizer__ = None


@pytest.mark.applymanifests(
    "opsani_dev",
    files=[
        "deployment.yaml",
        "service.yaml",
        "prometheus.yaml",
    ],
)
@pytest.mark.integration
@pytest.mark.usefixtures("kubeconfig", "kubernetes_asyncio_config")
class TestIntegration:
    class TestChecksOriginalState:
        @pytest.fixture(autouse=True)
        async def load_manifests(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, kubeconfig
        ) -> None:
            kube.wait_for_registered()
            checks.config.namespace = kube.namespace

            # Fake out the servo metadata in the environment
            # These env vars are set by our manifests
            pods = kube.get_pods(labels={ "app.kubernetes.io/name": "servo"})
            assert pods, "servo is not deployed"
            try:
                os.environ['POD_NAME'] = list(pods.keys())[0]
                os.environ["POD_NAMESPACE"] = kube.namespace

                yield

            finally:
                os.environ.pop('POD_NAME', None)
                os.environ.pop('POD_NAMESPACE', None)


        @pytest.mark.parametrize(
            "resource", ["namespace", "deployment", "container", "service"]
        )
        async def test_resource_exists(
            self, resource: str, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_kubernetes_{resource}")
            assert result.success

        async def test_target_container_resources_within_limits(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, config: servo.connectors.opsani_dev.OpsaniDevConfiguration
        ) -> None:
            config.cpu.min = "125m"
            config.cpu.max = "2000m"
            config.memory.min = "128MiB"
            config.memory.max = "4GiB"
            result = await checks.run_one(id=f"check_target_container_resources_within_limits")
            assert result.success, f"Expected success but got: {result}"

        async def test_target_container_resources_outside_of_limits(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, config: servo.connectors.opsani_dev.OpsaniDevConfiguration
        ) -> None:
            config.cpu.min = "4000m"
            config.cpu.max = "5000m"
            config.memory.min = "2GiB"
            config.memory.min = "4GiB"
            result = await checks.run_one(id=f"check_target_container_resources_within_limits")
            assert result.exception

        async def test_service_routes_traffic_to_deployment(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_service_routes_traffic_to_deployment")
            assert result.success, f"Failed with message: {result.message}"

        async def test_prometheus_sidecar_exists(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_prometheus_sidecar_exists")
            assert result.success

        async def test_prometheus_sidecar_is_ready(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_prometheus_sidecar_is_ready")
            assert result.success

        async def test_check_prometheus_restart_count(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_prometheus_restart_count")
            assert result.success

        async def test_check_prometheus_container_port(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            result = await checks.run_one(id=f"check_prometheus_container_port")
            assert result.success

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
                base_url=servo.connectors.opsani_dev.PROMETHEUS_SIDECAR_BASE_URL,
                assert_all_called=False
            ) as respx_mock:
                respx_mock.get(
                    "/ap/v1/targets",
                    name="targets"
                ).mock(return_value=httpx.Response(200, json=[]))

                respx_mock.get(
                    re.compile(r"/ap/v1/query_range.+"),
                    name="query",
                ).mock(return_value=httpx.Response(200, json=go_memstats_gc_sys_bytes))
                yield respx_mock

        async def test_check_prometheus_is_accessible(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            with respx.mock(base_url=servo.connectors.opsani_dev.PROMETHEUS_SIDECAR_BASE_URL) as respx_mock:
                request = respx_mock.get("/api/v1/targets").mock(return_value=httpx.Response(status_code=503))
                check = await checks.run_one(id=f"check_prometheus_is_accessible")
                assert request.called
                assert check
                assert check.name == 'Prometheus is accessible'
                assert check.id == "check_prometheus_is_accessible"
                assert check.critical
                assert not check.success
                assert check.message is not None
                assert isinstance(check.exception, httpx.HTTPStatusError)

@pytest.mark.applymanifests(
    "opsani_dev",
    files=[
        "deployment.yaml",
        "service.yaml",
        "prometheus.yaml",
    ],
)
@pytest.mark.integration
@pytest.mark.usefixtures("kubeconfig", "kubernetes_asyncio_config")
@pytest.mark.clusterrolebinding('cluster-admin')
class TestServiceMultiport:
    @pytest.fixture
    async def multiport_service(self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks) -> None:
        kube.wait_for_registered()
        service = await servo.connectors.kubernetes.Service.read(checks.config.service, checks.config.namespace)
        assert service
        assert len(service.ports) == 1
        assert service.find_port('http')

        # Add a port
        port = kubernetes_asyncio.client.V1ServicePort(name="elite", port=31337)
        service.obj.spec.ports.append(port)

        await service.patch()

        assert len(service.ports) == 2
        assert service.find_port('http')
        assert service.find_port('elite')

        return service

    async def test_requires_port_config_when_multiple_exist(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, multiport_service
    ) -> None:
        result = await checks.run_one(id=f"check_kubernetes_service_port")
        assert not result.success
        assert result.exception
        assert result.message == 'caught exception (ValueError): service defines more than one port: a `port` (name or number) must be specified in the configuration'

    async def test_resolve_port_by_name(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, multiport_service
    ) -> None:
        checks.config.port = 'elite'
        result = await checks.run_one(id=f"check_kubernetes_service_port")
        assert result.success
        assert result.message == 'Service Port: elite 31337:31337/TCP'

    async def test_resolve_port_by_number(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, multiport_service
    ) -> None:
        checks.config.port = 80
        result = await checks.run_one(id=f"check_kubernetes_service_port")
        assert result.success
        assert result.message == 'Service Port: http 80:8480/TCP'

    async def test_cannot_resolve_port_by_name(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, multiport_service
    ) -> None:
        kube.wait_for_registered()
        pass
        # checks.config.port = 'invalid'
        # result = await checks.run_one(id=f"check_kubernetes_service_port")
        # assert not result.success
        # assert result.message == 'caught exception (LookupError): could not find a port named: invalid'

    async def test_cannot_resolve_port_by_number(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks, multiport_service
    ) -> None:
        checks.config.port = 187
        result = await checks.run_one(id=f"check_kubernetes_service_port")
        assert not result.success
        assert result.message == 'caught exception (LookupError): could not find a port numbered: 187'
    # Errors:
    # Permissions, (Namespace, Deployment, Container, Service -> doesn't Exist, Cant Read), ports don't match

    # Warnings:
    # 9980 port conflict

    # init containers are ignored

    # TODO: Test deployment, pod with init container, test nginx not match,
    # TODO: check namespace affinity only scrapes in current namespace

    # async def test_permissions(self) -> None:
    #     ...

    # async def test_cannot_read_namespace(self) -> None:
    #     ...

    # async def test_cannot_read_deployment(self) -> None:
    #     ...

    # async def test_cannot_find_container(self) -> None:
    #     ...

    # async def test_resource_requirements(self) -> None:
    #     ...

    # async def test_deployment_ready(self) -> None:
    #     ...
    class TestInstall:
        @pytest.fixture(autouse=True)
        async def load_manifests(
            self, kube, kubeconfig, kubernetes_asyncio_config, checks: servo.connectors.opsani_dev.OpsaniDevChecks
        ) -> None:
            kube.wait_for_registered()
            checks.config.namespace = kube.namespace

            # Fake out the servo metadata in the environment
            # These env vars are set by our manifests
            deployment = kube.get_deployments()["servo"]
            pod = deployment.get_pods()[0]
            os.environ['POD_NAME'] = pod.name
            os.environ["POD_NAMESPACE"] = kube.namespace

        async def test_process(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks,
            kube_port_forward: Callable[[str, int], AsyncContextManager[str]],
            load_generator: Callable[[], 'LoadGenerator'],
        ) -> None:
            # Deploy fiber-http with annotations and Prometheus will start scraping it
            envoy_proxy_port = servo.connectors.opsani_dev.ENVOY_SIDECAR_DEFAULT_PORT
            async with kube_port_forward("deploy/servo", 9090) as prometheus_base_url:
                # Connect the checks to our port forward interface
                checks.config.prometheus_base_url = prometheus_base_url

                deployment = await servo.connectors.kubernetes.Deployment.read(checks.config.deployment, checks.config.namespace)
                assert deployment, f"failed loading deployment '{checks.config.deployment}' in namespace '{checks.config.namespace}'"

                prometheus_config = servo.connectors.prometheus.PrometheusConfiguration.generate(base_url=prometheus_base_url)
                prometheus_connector = servo.connectors.prometheus.PrometheusConnector(config=prometheus_config)

                ## Step 1
                servo.logger.critical("Step 1 - Annotate the Deployment PodSpec")
                async with assert_check_raises_in_context(
                    servo.checks.CheckError,
                    match="deployment 'fiber-http' is missing annotations"
                ) as assertion:
                    assertion.set(checks.run_one(id=f"check_deployment_annotations"))

                # Add a subset of the required annotations to catch partial setup cases
                async with change_to_resource(deployment):
                    await add_annotations_to_podspec_of_deployment(deployment,
                        {
                            "prometheus.opsani.com/path": "/stats/prometheus",
                            "prometheus.opsani.com/port": "9901",
                            "servo.opsani.com/optimizer": checks.config.optimizer.id,
                        }
                    )
                await assert_check_raises(
                    checks.run_one(id=f"check_deployment_annotations"),
                    servo.checks.CheckError,
                    re.escape("deployment 'fiber-http' is missing annotations: prometheus.opsani.com/scheme, prometheus.opsani.com/scrape")
                )

                # Fill in the missing annotations
                async with change_to_resource(deployment):
                    await add_annotations_to_podspec_of_deployment(deployment,
                        {
                            "prometheus.opsani.com/scrape": "true",
                            "prometheus.opsani.com/scheme": "http",
                        }
                    )
                await assert_check(checks.run_one(id=f"check_deployment_annotations"))

                # Step 2: Verify the labels are set on the Deployment pod spec
                servo.logger.critical("Step 2 - Label the Deployment PodSpec")
                await assert_check_raises(
                    checks.run_one(id=f"check_deployment_labels"),
                    servo.checks.CheckError,
                    re.escape("deployment 'fiber-http' is missing labels: servo.opsani.com/optimizer=test.com_foo, sidecar.opsani.com/type=envoy")
                )

                async with change_to_resource(deployment):
                    await add_labels_to_podspec_of_deployment(deployment,
                        {
                            "sidecar.opsani.com/type": "envoy",
                            "servo.opsani.com/optimizer": servo.connectors.kubernetes.labelize(checks.config.optimizer.id),
                        }
                    )
                await assert_check(checks.run_one(id=f"check_deployment_labels"))

                # Step 3
                servo.logger.critical("Step 3 - Inject Envoy sidecar container")
                await assert_check_raises(
                    checks.run_one(id=f"check_deployment_envoy_sidecars"),
                    servo.checks.CheckError,
                    re.escape("deployment 'fiber-http' pod template spec does not include envoy sidecar container ('opsani-envoy')")
                )

                # servo.logging.set_level("DEBUG")
                async with change_to_resource(deployment):
                    servo.logger.info(f"injecting Envoy sidecar to Deployment {deployment.name} PodSpec")
                    await deployment.inject_sidecar('opsani-envoy', 'opsani/envoy-proxy:latest', service="fiber-http")

                await wait_for_check_to_pass(functools.partial(checks.run_one, id=f"check_deployment_envoy_sidecars"))
                await wait_for_check_to_pass(functools.partial(checks.run_one, id=f"check_pod_envoy_sidecars"))

                # Step 4
                servo.logger.critical("Step 4 - Check that Prometheus is discovering and scraping annotated Pods")
                servo.logger.info("waiting for Prometheus to scrape our Pods")

                async def wait_for_targets_to_be_scraped() -> List[servo.connectors.prometheus.ActiveTarget]:
                    servo.logger.info(f"Waiting for Prometheus scrape Pod targets...")
                    # NOTE: Prometheus is on a 5s scrape interval
                    scraped_since = pytz.utc.localize(datetime.datetime.now())
                    while True:
                        targets = await prometheus_connector.targets()
                        if targets:
                            if not any(filter(lambda t: t.last_scraped_at is None or t.last_scraped_at < scraped_since, targets.active)):
                                # NOTE: filter targets to match our namespace in
                                # case there are other things running in the cluster
                                return list(filter(lambda t: t.labels["kubernetes_namespace"] == kube.namespace, targets.active))

                await wait_for_targets_to_be_scraped()
                await assert_check(checks.run_one(id=f"check_prometheus_targets"))

                # Step 5
                servo.logger.critical("Step 5 - Check that traffic metrics are coming in from Envoy")
                await assert_check_raises(
                    checks.run_one(id=f"check_envoy_sidecar_metrics"),
                    servo.checks.CheckError,
                    re.escape("Envoy is not reporting any traffic to Prometheus")
                )

                servo.logger.info(f"Sending test traffic to Envoy through deploy/fiber-http")
                async with kube_port_forward("deploy/fiber-http", envoy_proxy_port) as envoy_url:
                    await load_generator(envoy_url).run_until(
                        wait_for_check_to_pass(
                            functools.partial(checks.run_one, id=f"check_envoy_sidecar_metrics")
                        )
                    )

                # Let Prometheus scrape to see the traffic
                await wait_for_check_to_pass(functools.partial(checks.run_one, id=f"check_prometheus_targets"))

                # Step 6
                servo.logger.critical("Step 6 - Proxy Service traffic through Envoy")
                await assert_check_raises(
                    checks.run_one(id=f"check_service_proxy"),
                    servo.checks.CheckError,
                    re.escape(f"service 'fiber-http' is not routing traffic through Envoy sidecar on port {envoy_proxy_port}")
                )

                # Update the port to point to the sidecar
                service = await servo.connectors.kubernetes.Service.read("fiber-http", checks.config.namespace)
                service.ports[0].target_port = envoy_proxy_port
                async with change_to_resource(service):
                    await service.patch()
                await wait_for_check_to_pass(functools.partial(checks.run_one, id=f"check_service_proxy"))

                # Send traffic through the service and verify it shows up in Envoy
                port = service.ports[0].port
                servo.logger.info(f"Sending test traffic through proxied Service fiber-http on port {port}")

                async with kube_port_forward(f"service/fiber-http", port) as service_url:
                    await load_generator(envoy_url).run_until(wait_for_targets_to_be_scraped())

                # Let Prometheus scrape to see the traffic
                await assert_check(checks.run_one(id=f"check_prometheus_targets"))

                # Step 7
                servo.logger.critical("Step 7 - Bring tuning Pod online")
                async with change_to_resource(deployment):
                    await deployment.ensure_tuning_pod()
                await assert_check(checks.run_one(id=f"check_tuning_is_running"))

                # Step 8
                servo.logger.critical("Step 8 - Verify Service traffic makes it through Envoy and gets aggregated by Prometheus")
                async with kube_port_forward(f"service/fiber-http", port) as service_url:
                    await load_generator(service_url).run_until(wait_for_targets_to_be_scraped())

                targets = await wait_for_targets_to_be_scraped()
                assert len(targets) == 2
                main = next(filter(lambda t: "opsani_role" not in t.labels, targets))
                tuning = next(filter(lambda t: "opsani_role" in t.labels, targets))
                assert main.pool == "opsani-envoy-sidecars"
                assert main.health == "up"
                assert main.labels["app_kubernetes_io_name"] == "fiber-http"

                assert tuning.pool == "opsani-envoy-sidecars"
                assert tuning.health == "up"
                assert tuning.labels["opsani_role"] == "tuning"
                assert tuning.discovered_labels["__meta_kubernetes_pod_name"] == "fiber-http-tuning"
                assert tuning.discovered_labels["__meta_kubernetes_pod_label_opsani_role"] == "tuning"

                async with kube_port_forward(f"service/fiber-http", port) as service_url:
                    await load_generator(service_url).run_until(
                        wait_for_check_to_pass(
                            functools.partial(checks.run_one, id=f"check_traffic_metrics")
                        )
                    )

                servo.logger.success("🥷 Opsani Dev is now deployed.")
                servo.logger.critical("🔥 Now witness the firepower of this fully ARMED and OPERATIONAL battle station!")

                # Cancel outstanding tasks
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                [task.cancel() for task in tasks]

                await asyncio.gather(*tasks, return_exceptions=True)

        async def test_install_wait(
            self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks,
            kube_port_forward: Callable[[str, int], AsyncContextManager[str]],
            load_generator: Callable[[], 'LoadGenerator'],
            tmp_path: pathlib.Path
        ) -> None:
            servo.logging.set_level("TRACE")

            async with kube_port_forward("deploy/servo", 9090) as prometheus_base_url:
                # Connect the checks to our port forward interface
                checks.config.prometheus_base_url = prometheus_base_url

                deployment = await servo.connectors.kubernetes.Deployment.read(checks.config.deployment, checks.config.namespace)
                assert deployment, f"failed loading deployment '{checks.config.deployment}' in namespace '{checks.config.namespace}'"

                async def loop_checks() -> None:
                    while True:
                        results = await checks.run_all()
                        next_failure = next(filter(lambda r: r.success is False, results), None)
                        if next_failure:
                            servo.logger.critical(f"Attempting to remedy failing check: {devtools.pformat(next_failure)}")#, exception=next_failure.exception)
                            await _remedy_check(
                                next_failure.id,
                                config=checks.config,
                                deployment=deployment,
                                kube_port_forward=kube_port_forward,
                                load_generator=load_generator,
                                checks=checks
                            )
                        else:
                            break

                await asyncio.wait_for(loop_checks(), timeout=420.0)

            servo.logger.success("🥷 Opsani Dev is now deployed.")
            servo.logger.critical("🔥 Now witness the firepower of this fully ARMED and OPERATIONAL battle station!")



##
# FIXME: Migrate these assertions into a better home and fix the line number mess

async def assert_check_fails(
    check: servo.checks.Check,
    message: Optional[str] = None
) -> None:
    """Assert that a check fails.

    The check provided can be a previously executed Check object or a coroutine that returns a Check.
    This assertion does not differentiate between boolean and exceptional failures. If you want to test
    a particular exceptional failure condition, take a look at `assert_check_raises`.
    """
    return await assert_check(check, message, _success=False)

@runtime_checkable
class Assertable(Protocol):
    """An object that can set an assertable value."""

    def set(value: Any) -> None:
        """Set the value of the assertion."""
        ...

# TODO: doesn't have to be async
@contextlib.asynccontextmanager
async def assert_check_raises_in_context(
    type_: Type[Exception],
    match: Optional[str] = None,
    *,
    message: Optional[str] = None,
) -> AsyncContextManager[Assertable]:
    """Assert that a check fails due to a specific exception being raised within an execution context.

    The check provided can be a previously executed Check object or a coroutine that returns a Check.
    The exception type is evaluated and the `match` parameter is matched against the underlying exception
    via `pytest.assert_raises` and supports strings and regex objects.

    This method is an asynchronous context manager for use via the `async with ..` syntax:
        ```
        async with assert_check_raises_in_context(AttributeError) as assertion:
            assertion.set(check.whatever(True))
        ```

    The opaque `Assertable` object yielded exposes a single method `set` that accepts a `Check` object value or
    a callable that returns a Check object. The callable can be asynchronous and will be awaited.
    This syntax can be more readable and enables setup/teardown and debugging logic that would otherwise
    be rather unergonomic.

    Args:
        type_: The type of exception expected.
        match: A string or regular expression for matching against the error raised.
        message: An optional override for the error message returned on failure.
    """

    class _Assertion(Assertable):
        def set(self, value) -> None:
            self._value = value

        async def get(self) -> servo.checks.Check:
            if asyncio.iscoroutine(self._value):
                return await self._value
            else:
                return self._value

    assertion = _Assertion()
    yield assertion
    value = await assertion.get()

    assert value is not None, f"invalid use as context manager: must return a Check"
    await assert_check(value, message, _success=False, _exception_type=type_, _exception_match=match)

async def assert_check_raises(
    check,
    type_: Type[Exception] = Exception,
    match: Optional[str] = None,
    *,
    message: Optional[str] = None,
) -> None:
    """Assert that a check fails due to a specific exception being raised.

    The check provided can be a previously executed Check object or a coroutine that returns a Check.
    The exception type is evaluated and the `match` parameter is matched against the underlying exception
    via `pytest.assert_raises` and supports strings and regex objects.

    Args:
        check: A check object or callable that returns a check object to be evaluated.
        type_: The type of exception expected to fail the check.
        match: A string or regular expression to be evaluated against the message of the exception that triggered the failure.
    """
    async with assert_check_raises_in_context(type_, match, message=message) as assertion:
        assertion.set(check)

async def assert_check(
    check: Union[servo.checks.Check, Callable],
    message: Optional[str] = None,
    *,
    _success: bool = True,
    _exception_type: Optional[Type[Exception]] = None,
    _exception_match: Optional[str] = None
) -> None:
    """Assert the outcome of a check matches expectations.

    The check provided can be a previously executed Check object or a coroutine that returns a Check.
    In the event of an exceptional failure, underlying exceptions are chained to facilitate debugging.
    This method underlies other higher level semantically named assertions prefixed with `assert_check_`.

    Args:
        check: The check object or coroutine that returns a check object to be evaluated.
        message: An optional message to annotate a failing assertion. When omitted, a message is synthesized.
    """
    if asyncio.iscoroutine(check):
        result = await check
        if callable(result):
            result = await result
    elif isinstance(check, servo.checks.Check):
        result = check
    else:
        raise TypeError(f"unknown check: {check}")

    # NOTE: Take care to chain the exceptions rescued by the Check for attribution
    if _success is False and result.success is False:
        # Make sure we failed in the right way
        if result.exception:
            if _exception_type or _exception_match:
                with pytest.raises(_exception_type, match=_exception_match):
                    raise result.exception
        elif _exception_type:
            raise AssertionError(
                f"Check(id='{result.id}') '{result.name}' was expected to raise a {_exception_type.__name__} but it did not: {message}"
            ) from result.exception

    if result.success != _success:
        if result.success:
            raise AssertionError(f"Check(id='{result.id}') '{result.name}' succeeded when you were expecting a failure")
        else:
            raise AssertionError(
                f"Check(id='{result.id}') '{result.name}' failed: {message or result.message}"
            ) from result.exception


# TODO: Move these into library functions. Do we want replace/merge versions?
async def add_annotations_to_podspec_of_deployment(deployment, annotations: Dict[str, str]) -> None:
    servo.logger.info(f"adding annotations {annotations} to PodSpec of Deployment '{deployment.name}'")
    existing_annotations = deployment.pod_template_spec.metadata.annotations or {}
    existing_annotations.update(annotations)
    deployment.pod_template_spec.metadata.annotations = existing_annotations
    await deployment.patch()


async def add_labels_to_podspec_of_deployment(deployment, labels: List[str]) -> None:
    servo.logger.info(f"adding labels {labels} to PodSpec of Deployment '{deployment.name}'")
    existing_labels = deployment.pod_template_spec.metadata.labels or {}
    existing_labels.update(labels)
    deployment.pod_template_spec.metadata.labels = existing_labels
    await deployment.patch()

@contextlib.asynccontextmanager
async def change_to_resource(resource: servo.connectors.kubernetes.KubernetesModel):
    status = resource.status
    metadata = resource.obj.metadata

    if isinstance(resource, servo.connectors.kubernetes.Deployment):
        async with resource.rollout():
            yield
    else:
        # allow the resource to be changed
        yield

    await resource.refresh()

    # early exit if nothing changed
    if (resource.obj.metadata.resource_version == metadata.resource_version):
        servo.logger.debug(f"exiting early: metadata resource version has not changed")
        return

    if hasattr(status, "observed_generation"):
        while status.observed_generation == resource.status.observed_generation:
            await resource.refresh()

    # wait for the change to roll out
    if isinstance(resource, servo.connectors.kubernetes.Deployment):
        await resource.wait_until_ready()
    elif isinstance(resource, servo.connectors.kubernetes.Pod):
        await resource.wait_until_ready()
    elif isinstance(resource, servo.connectors.kubernetes.Service):
        pass
    else:
        servo.logger.warning(f"no change observation strategy for Kubernetes resource of type `{resource.__class__.__name__}`")

class LoadGenerator(pydantic.BaseModel):
    request_count: int = 0
    _request: httpx.Request = pydantic.PrivateAttr()
    _event: asyncio.Event = pydantic.PrivateAttr(default_factory=asyncio.Event)
    _task: Optional[asyncio.Task] = pydantic.PrivateAttr(None)

    def __init__(self, target: Union[str, httpx.Request]) -> None:
        super().__init__()
        if isinstance(target, httpx.Request):
            self._request = target
        elif isinstance(target, str):
            self._request = httpx.Request("GET", target)
        else:
            raise TypeError(f"unknown target type '{target.__class__.__name__}': expected str or httpx.Request")

    @property
    def request(self) -> httpx.Request:
        return self._request

    @property
    def url(self) -> str:
        return self._request.url

    def start(self) -> None:
        """Start sending traffic."""
        if self.is_running:
            raise RuntimeError("Cannot start a load generator that is already running")

        async def _send_requests() -> None:
            started_at = datetime.datetime.now()
            async with httpx.AsyncClient() as client:
                while not self._event.is_set():
                    servo.logger.trace(f"Sending traffic to {self.url}...")
                    try:
                        await client.send(self.request, timeout=1.0)
                    except httpx.TimeoutException as err:
                        servo.logger.warning(f"httpx.TimeoutException encountered sending request {self.request}: {err}")
                        pass
                    self.request_count += 1

            duration = servo.Duration(datetime.datetime.now() - started_at)
            servo.logger.success(f"Sent {self.request_count} requests to {self.url} over {duration} seconds.")

        self._event.clear()
        self._task = asyncio.create_task(_send_requests())

    @property
    def is_running(self) -> bool:
        """Return True if traffic is being sent."""
        return self._task is not None

    def stop(self) -> None:
        """Stop sending traffic."""
        self._event.set()

        if self._task:
            self._task.cancel()

    async def run_until(
        self,
        condition: Union[servo.Futuristic, servo.DurationDescriptor],
        *,
        timeout: servo.DurationDescriptor = servo.Duration("1m")
    ) -> None:
        """Send traffic until a condition is met or a timeout expires.

        If the load generator is not already running, it is started.

        Args:
            condition: A futuristic object (async Task, coroutine, or awaitable)
                to monitor for completion or a time duration descriptor (e.g. "30s",
                15, 2.5, or a servo.Duration object) to send for a fixed time
                interval.
            timeout: A time duration descriptor describing the timeout interval.

        Raises:
            asyncio.TimeoutError: Raised if the timeout expires before the
                condition is met.
        """
        if servo.isfuturistic(condition):
            future = asyncio.ensure_future(condition)
        else:
            # create a sleeping coroutine for the desired duration
            duration = servo.Duration(condition)
            future = asyncio.create_task(asyncio.sleep(duration.total_seconds()))

        if not self.is_running:
            self.start()

        try:
            duration = servo.Duration(timeout)
            await asyncio.wait_for(
                future,
                timeout=duration.total_seconds()
            )
        except asyncio.TimeoutError:
            servo.logger.error(f"Timed out after {duration} waiting for condition: {condition}")
        finally:
            self.stop()

            with contextlib.suppress(asyncio.CancelledError):
                await self._task

@pytest.fixture
def load_generator() -> Callable[[Union[str, httpx.Request]], LoadGenerator]:
    return LoadGenerator

async def wait_for_check_to_pass(
    check: Coroutine[None, None, servo.Check],
    *,
    timeout: servo.Duration = servo.Duration("15s")
) -> servo.Check:
    async def _loop_check() -> servo.Check:
        while True:
            result = await check()
            if result.success:
                break

        return result

    try:
        check = await asyncio.wait_for(
            _loop_check(),
            timeout=timeout.total_seconds()
        )
    except asyncio.TimeoutError as err:
        servo.logger.error(f"Check timed out after {timeout}. Final state: {devtools.pformat(check)}")
        raise err

    return check

async def _remedy_check(id: str, *, config, deployment, kube_port_forward, load_generator, checks) -> None:
    envoy_proxy_port = servo.connectors.opsani_dev.ENVOY_SIDECAR_DEFAULT_PORT
    servo.logger.warning(f"Remedying failing check '{id}'...")
    if id == 'check_deployment_annotations':
        ## Step 1
        servo.logger.critical("Step 1 - Annotate the Deployment PodSpec")
        async with change_to_resource(deployment):
            await add_annotations_to_podspec_of_deployment(deployment,
                {
                    "prometheus.opsani.com/path": "/stats/prometheus",
                    "prometheus.opsani.com/port": "9901",
                    "prometheus.opsani.com/scrape": "true",
                    "prometheus.opsani.com/scheme": "http",
                    "servo.opsani.com/optimizer": config.optimizer.id,
                }
            )

    elif id == 'check_deployment_labels':
        # Step 2: Verify the labels are set on the Deployment pod spec
        servo.logger.critical("Step 2 - Label the Deployment PodSpec")
        async with change_to_resource(deployment):
            await add_labels_to_podspec_of_deployment(deployment,
                {
                    "sidecar.opsani.com/type": "envoy",
                    "servo.opsani.com/optimizer": servo.connectors.kubernetes.labelize(config.optimizer.id),
                }
            )

    elif id == 'check_deployment_envoy_sidecars':
        # Step 3
        servo.logger.critical("Step 3 - Inject Envoy sidecar container")
        async with change_to_resource(deployment):
            servo.logger.info(f"injecting Envoy sidecar to Deployment {deployment.name} PodSpec")
            await deployment.inject_sidecar('opsani-envoy', 'opsani/envoy-proxy:latest', service="fiber-http")

    elif id in {'check_prometheus_sidecar_exists', 'check_pod_envoy_sidecars', 'check_prometheus_is_accessible'}:
        servo.logger.warning(f"check failed: {id}")
        pass

    elif id == 'check_prometheus_targets':
        # Step 4
        servo.logger.critical("Step 4 - Check that Prometheus is discovering and scraping annotated Pods")
        servo.logger.info("waiting for Prometheus to scrape our Pods")

    elif id == 'check_envoy_sidecar_metrics':
        # Step 5
        servo.logger.critical("Step 5 - Check that traffic metrics are coming in from Envoy")
        servo.logger.info(f"Sending test traffic to Envoy through deploy/fiber-http")
        async with kube_port_forward("deploy/fiber-http", envoy_proxy_port) as envoy_url:
            await load_generator(envoy_url).run_until(
                wait_for_check_to_pass(
                    functools.partial(checks.run_one, id=f"check_envoy_sidecar_metrics")
                )
            )


    elif id == 'check_service_proxy':
        # Step 6
        servo.logger.critical("Step 6 - Proxy Service traffic through Envoy")

        # Update the port to point to the sidecar
        service = await servo.connectors.kubernetes.Service.read("fiber-http", config.namespace)
        service.ports[0].target_port = envoy_proxy_port
        async with change_to_resource(service):
            await service.patch()

    elif id == 'check_tuning_is_running':
        servo.logger.critical("Step 7 - Bring tuning Pod online")
        async with change_to_resource(deployment):
            await deployment.ensure_tuning_pod()

    elif id == 'check_traffic_metrics':
        # Step 8
        servo.logger.critical("Step 8 - Verify Service traffic makes it through Envoy and gets aggregated by Prometheus")
        async with kube_port_forward(f"service/fiber-http", 80) as service_url:
            await load_generator(service_url).run_until(
                wait_for_check_to_pass(
                    functools.partial(checks.run_one, id=f"check_traffic_metrics")
                )
            )

    else:
        raise AssertionError(f"unhandled check: '{id}'")
