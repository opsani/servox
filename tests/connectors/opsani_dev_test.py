import asyncio
import os
import re

import contextlib

from typing import Callable, AsyncIterator, Dict, List, Optional, Any, Type, Set, Union, Protocol, runtime_checkable

import httpx
import pytest
import respx

import servo
import servo.connectors.kubernetes
import servo.connectors.opsani_dev
import servo.connectors.prometheus

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
def config() -> servo.connectors.opsani_dev.OpsaniDevConfiguration:
    return servo.connectors.opsani_dev.OpsaniDevConfiguration(
        namespace="default",
        deployment="fiber-http",
        container="fiber-http",
        service="fiber-http",
    )


@pytest.fixture
def checks(config: servo.connectors.opsani_dev.OpsaniDevConfiguration) -> servo.connectors.opsani_dev.OpsaniDevChecks:
    return servo.connectors.opsani_dev.OpsaniDevChecks(config=config)


@pytest.mark.applymanifests(
    "opsani_dev",
    files=[
        "deployment.yaml",
        "service.yaml",
        "prometheus.yaml",
    ],
)
class TestChecksOriginalState:
    @pytest.fixture(autouse=True)
    async def load_manifests(
        self, kube, kubeconfig, kubernetes_asyncio_config, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        kube.wait_for_registered(timeout=30)
        checks.config.namespace = kube.namespace

        # Fake out the servo metadata in the environment
        # These env vars are set by our manifests
        pods = kube.get_pods(labels={ "app.kubernetes.io/name": "servo"})
        assert pods, "servo is not deployed"
        os.environ['POD_NAME'] = list(pods.keys())[0]
        os.environ["POD_NAMESPACE"] = kube.namespace

    @pytest.mark.parametrize(
        "resource", ["namespace", "deployment", "container", "service"]
    )
    async def test_resource_exists(
        self, resource: str, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_kubernetes_{resource}")
        assert result.success

    async def test_prometheus_configmap_exists(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        result = await checks.run_one(id=f"check_prometheus_config_map")
        assert result.success

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
                "/api/v1/targets",
                name="targets"
            ).mock(return_value=httpx.Response(200, json=[]))

            respx_mock.get(
                re.compile(r"/api/v1/query_range.+"),
                name="query",
            ).mock(return_value=httpx.Response(200, json=go_memstats_gc_sys_bytes))
            yield respx_mock

    # check_kubernetes_service_type
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


class TestChecksServiceUpdated:
    ...


class TestChecksSidecarsInjected:
    ...


# Errors:
# Permissions, (Namespace, Deployment, Container, Service -> Doesnt Exist, Cant Read), ports don't match

# Warnings:
# 9980 port conflict

# init containers are ignored


# TODO: Test deployment, pod with init container, test nginx not match,
# TODO: check namespace affinity only scrapes in current namespace


# Map to the app, 


# To a full check on Opsani dev...
# App exists
# App has resources
# No sidecar
# Inject the sidecar
# Sidecar inhected
# Sidecar getting traffic
# Service port and envoy proxy port match
# I send traffic through the service and it shows up in Envoy
# The canary comes up

# class IntegrationTests:
#     @pytest.mark.applymanifests(
#         "../manifests",
#         files=[
#             "fiber-http.yaml",
#             # "k6.yaml",
#             # "prometheus.yaml",
#         ],
#     )
#     async def test_all(
#         self,
#         optimizer: servo.Optimizer,
#         kube_port_forward: Callable[[str, int], AsyncIterator[str]],
#     ) -> None:
        # Deploy fiber
        # Verify single container
        # Do sidecar injection
        # Update the service
        # Check for Prometheus
        # Check for Prometheus Targets scraping
        # Check for Envoy
        # Check for Envoy traffic
        
        
        # Deploy fiber-http with annotations and Prometheus will start scraping it
        # async with kube_port_forward("deploy/prometheus", 9090) as url:
        #     config = servo.connectors.prometheus.PrometheusConfiguration.generate(base_url=url)
        #     connector = servo.connectors.prometheus.PrometheusConnector(config=config, optimizer=optimizer)
        #     while True:
        #         metrics = await asyncio.wait_for(
        #             asyncio.gather(connector.measure(control=servo.Control(duration="5s"))),
        #             timeout=30
        #         )
        #         debug(metrics)
        #         break
# async def test_

@pytest.mark.applymanifests(
    "opsani_dev",
    files=[
        "deployment.yaml",
        "service.yaml",
        # "servo.yaml",
        "prometheus.yaml",
    ],
)
class TestEverything:
    @pytest.fixture(autouse=True)
    async def load_manifests(
        self, kube, kubeconfig, kubernetes_asyncio_config, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    ) -> None:
        kube.wait_for_registered(timeout=30)
        checks.config.namespace = kube.namespace
        
        # Fake out the servo metadata in the environment
        # These env vars are set by our manifests
        deployment = kube.get_deployments()["servo"]
        pod = deployment.get_pods()[0]
        os.environ['POD_NAME'] = pod.name
        os.environ["POD_NAMESPACE"] = kube.namespace
    
    # async def test_target_discovery(
    #     self,
    #     optimizer: servo.Optimizer,
    #     kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    # ) -> None:
    #     # Deploy fiber-http with annotations and Prometheus will start scraping it
    #     async with kube_port_forward("deploy/prometheus", 9090) as url:
    #         config = PrometheusConfiguration.generate(base_url=url)
    #         connector = PrometheusConnector(config=config, optimizer=optimizer)
    #         while True:
    #             metrics = await asyncio.wait_for(
    #                 asyncio.gather(connector.measure(control=servo.Control(duration="5s"))),
    #                 timeout=30
    #             )
    #             debug(metrics)
    #             break

    # @pytest.mark.parametrize(
    #     "resource", ["namespace", "deployment", "container", "service"]
    # )
    # async def test_resource_exists(
    #     self, resource: str, checks: servo.connectors.opsani_dev.OpsaniDevChecks
    # ) -> None:
    #     result = await checks.run_one(id=f"check_kubernetes_{resource}")
    #     assert result.success

    async def test_all(
        self, kube, checks: servo.connectors.opsani_dev.OpsaniDevChecks,
        kube_port_forward: Callable[[str, int], AsyncIterator[str]],
    ) -> None:
        
        # Deploy fiber-http with annotations and Prometheus will start scraping it
        # TODO: this should be referencing a servo as a sidecar
        # TODO: the name here is misleading -- its in prometheus.yaml
        async with kube_port_forward("deploy/servo", 9090) as url:
            # Connect the checks to our port forward interface
            checks.config.prometheus_base_url = url
            
            deployment = await servo.connectors.kubernetes.Deployment.read(checks.config.deployment, checks.config.namespace)
            assert deployment, f"failed loading deployment '{checks.config.deployment}' in namespace '{checks.config.namespace}'"
            
            # TODO: Move these into library functions. Do we want replace/merge versions?
            async def add_annotations_to_podspec_of_deployment(deployment, annotations: Dict[str, str]) -> None:
                existing_annotations = deployment.pod_template_spec.metadata.annotations or {}
                existing_annotations.update(annotations)
                deployment.pod_template_spec.metadata.annotations = existing_annotations
                await deployment.patch()
                await deployment.refresh()
                
            
            async def add_labels_to_podspec_of_deployment(deployment, labels: List[str]) -> None:
                await deployment.refresh()
                existing_labels = deployment.pod_template_spec.metadata.labels or {}
                existing_labels.update(labels)
                deployment.pod_template_spec.metadata.labels = existing_labels
                await deployment.patch()
                await deployment.refresh()  # TODO: Figure out a better logic for this...
            
            # NOTE: Step 1 - No annotations and fail
            async with assert_check_raises_in_context(
                AssertionError,
                match="deployment 'fiber-http' does not have any annotations"
            ) as assertion:
                assertion.set(checks.run_one(id=f"check_deployment_annotations"))
            
            # NOTE: Step 2 - Add annotations and pass
            # Add a subset of the required annotations to catch partial setup cases
            await add_annotations_to_podspec_of_deployment(deployment,
                {
                    "prometheus.opsani.com/path": "/stats/prometheus",
                    "prometheus.opsani.com/port": "9901",
                }
            )
            await assert_check_raises(
                checks.run_one(id=f"check_deployment_annotations"),
                AssertionError,
                re.escape("missing annotations: ['prometheus.opsani.com/scheme', 'prometheus.opsani.com/scrape']")
            )
            
            # Fill in the missing annotations
            await add_annotations_to_podspec_of_deployment(deployment,
                {
                    "prometheus.opsani.com/scrape": "true",
                    "prometheus.opsani.com/scheme": "http",
                }
            )
            await assert_check(checks.run_one(id=f"check_deployment_annotations"))
            
            # Step 3: Check the labels and pass
            await assert_check_raises(
                checks.run_one(id=f"check_deployment_labels"),
                AssertionError,
                re.escape("missing labels: {'sidecar.opsani.com/type': 'envoy'}")
            )
            
            await add_labels_to_podspec_of_deployment(deployment,
                {
                    "sidecar.opsani.com/type": "envoy"
                }
            )
            await assert_check(checks.run_one(id=f"check_deployment_labels"))
            
            # Step 4: Look for sidecars running on our app and fail
            await assert_check_raises(
                checks.run_one(id=f"check_deployment_labels"),
                AssertionError,
                re.escape("missing labels: {'sidecar.opsani.com/type': 'envoy'}")
            )
            
            # Add the sidecar and pass
            
            # Step 5: Check that Prometheus has a target for the sidecar
            
            # Step 6: Check that the traffic metrics are flatlined for Envoy
            # Send traffic directly through to envoy and make sure it shows up            
            
            # Step 7: Check for our service
            # Update the port to point to the sidecar
            # Send traffic through the service and verify it shows up in Envoy

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

@contextlib.asynccontextmanager
async def assert_check_raises_in_context(    
    type_: Type[Exception], 
    match: Optional[str] = None,
    *,    
    message: Optional[str] = None,
) -> AsyncIterator[Assertable]:
    """Assert that a check fails due to a specific exception being raised within an execution context.
    
    The check provided can be a previously executed Check object or a coroutine that returns a Check.
    The exception type is evalauted and the `match` parameter is matched against the underlying exception
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
    The exception type is evalauted and the `match` parameter is matched against the underlying exception
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
        message: An optional message to annotate a failing assertion. When omitted, a message is syntehsized.
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
                f"Check(id='{result.id}') '{result.name}' was expected to raise a {_exception_type.__name__} but it did not: {assertion_message}"
            ) from result.exception
    
    if result.success != _success:
        if result.success:
            raise AssertionError(f"Check(id='{result.id}') '{result.name}' succeeded when you were expecting a failure")
        else:    
            raise AssertionError(
                f"Check(id='{result.id}') '{result.name}' failed: {message or result.message}"
            ) from result.exception
