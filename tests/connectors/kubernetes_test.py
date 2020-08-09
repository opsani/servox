import asyncio
from kubetest.objects import namespace
import pytest

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client.api_client import ApiClient

from servo.connectors.kubernetes import KubernetesConfiguration, KubernetesConnector
from servo.types import Component, Setting

class TestKubernetesConfiguration:
    pass

class TestKubernetesConnector:
    pass

@pytest.mark.integration
class TestKubernetesConnectorIntegration:
    async def test_describe(self, config):
        connector = KubernetesConnector(config=config)
        description = await connector.describe()
        assert description.get_setting("co-http-deployment.cpu").value == 1.0
        assert description.get_setting("co-http-deployment.memory").value == "3G"
        assert description.get_setting("co-http-deployment.replicas").value == 1

async def test_measure(config):
    connector = KubernetesConnector(config=config)
    description = await connector.measure()

from servo.api import descriptor_to_adjustments
async def test_adjust(config, adjustment):
    connector = KubernetesConnector(config=config)
    description = await connector.adjust(descriptor_to_adjustments(adjustment))
    debug(description)

async def test_apply_no_changes():
    # resource_version stays the same and early exits
    pass

async def test_apply_metadata_changes():
    # Update labels or something that doesn't matter
    # Detect by never getting a progressing event
    pass

async def test_apply_replica_change():
    # bump the count, observed_generation goes up
    # wait for the counts to settle
    ...

async def test_apply_memory_change():
    # bump the count, observed_generation goes up
    # wait for the counts to settle
    ...

async def test_apply_cpu_change():
    # bump the count, observed_generation goes up
    # wait for the counts to settle
    ...

async def test_apply_unschedulable_memory_request():
    # bump the count, observed_generation goes up
    # wait for the counts to settle
    ...

async def test_apply_restart_strategy():
    # Make sure we can watch a non-rolling update
    # .spec.strategy specifies the strategy used to replace old Pods by new ones. .spec.strategy.type can be "Recreate" or "RollingUpdate". "RollingUpdate" is the default value.
    # Recreate Deployment
    ...
# TODO: Put a co-http deployment live. Create a config and describe it.
# TODO: Test talking to multiple namespaces. Test kubeconfig file
# Test describe an empty config.
# Version ID checks
# Timeouts, Encoders, refresh, ready
# Add watch, test create, read, delete, patch
# TODO: settlement time, recovery behavior (rollback, delete), "adjust_on"?, restart detection
# TODO: wait/watch tests with conditionals...
# TODO: Test cases will be: change memory, change cpu, change replica count. 
# Test setting limit and request independently
# Detect scheduling error

# TODO: We want to compute progress by looking at observed generation,
# then watching as all the replicas are updated until the counts match
# If we never see a progressing condition, then whatever we did
# did not affect the deployment
# Handle: CreateContainerError

from servo.connectors.kubernetes import KubernetesChecks

async def test_checks(config: KubernetesConfiguration):
    await KubernetesChecks.run(config)

from servo.connectors.kubernetes import Millicore
import pydantic

def test_millicpu():
    class Model(pydantic.BaseModel):
        cpu: Millicore

    assert Model(cpu=0.1).cpu == 100
    assert Model(cpu=0.5).cpu == 500
    assert Model(cpu=1).cpu == 1000
    assert Model(cpu="100m").cpu == 100
    assert str(Model(cpu=1.5).cpu) == "1500m"
    assert float(Model(cpu=1.5).cpu) == 1.5
    assert Model(cpu=0.1).cpu == "100m"
    assert Model(cpu="100m").cpu == 0.1


@pytest.fixture
def config() -> KubernetesConfiguration:
    return KubernetesConfiguration(
        namespace="default",
        deployments=[
            Component(
                name="co-http-deployment",
                settings=[
                    Setting(
                        name="cpu",
                        min=0.1,
                        max=0.8,
                        step=0.125,
                        type="range"
                    ),
                    Setting(
                        name="memory",
                        min=0.1,
                        max=0.8,
                        step=0.125,
                        type="range"
                    ),
                    Setting(
                        name="replicas",
                        min=1,
                        max=2,
                        step=1,
                        type="range"
                    ),
                ]
            )
        ]
    )

@pytest.fixture
def adjustment() -> dict:
    return {
        'application': {
            'components': {
                'co-http-deployment': {
                    'settings': {
                        'cpu': {
                            'value': 1.80, #0.725,
                        },
                        'memory': {
                            'value': "2.5G", #2.25,
                        },
                        'replicas': {
                            'value': 3.0,
                        },
                    },
                },
            },
        },
        'control': {},
    }


    # servo/connectors/kubernetes.py:1394 adjust
    # data:  (dict) len=2
    # adjustments: [
    #     Component(
    #         name='co-http-deployment',
    #         settings=[
    #             Setting(
    #                 name='cpu',
    #                 type=<SettingType.RANGE: 'range'>,
    #                 min=1.0,
    #                 max=4.0,
    #                 step=1.0,
    #                 value=0.725,
    #                 pinned=False,
    #             ),
    #             Setting(
    #                 name='memory',
    #                 type=<SettingType.RANGE: 'range'>,
    #                 min=1.0,
    #                 max=4.0,
    #                 step=1.0,
    #                 value=2.25,
    #                 pinned=False,
    #             ),
    #             Setting(
    #                 name='replicas',
    #                 type=<SettingType.RANGE: 'range'>,
    #                 min=1.0,
    #                 max=4.0,
    #                 step=1.0,
    #                 value=2.0,
    #                 pinned=False,
    #             ),
    #         ],
    #         env=None,
    #         command=None,
    #     ),
    # ] (list) len=1


# event: {
#         'type': 'ERROR',
#         'object': {
#             'kind': 'Status',
#             'apiVersion': 'v1',
#             'metadata': {},
#             'status': 'Failure',
#             'message': 'too old resource version: 1226459 (1257919)',
#             'reason': 'Expired',
#             'code': 410,
#         },