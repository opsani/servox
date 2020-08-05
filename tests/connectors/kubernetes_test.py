import asyncio
from servo.kubernetes import Deployment
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

async def test_adjust(config, adjustment):
    connector = KubernetesConnector(config=config)
    description = await connector.adjust(adjustment)

# TODO: Test the input units for CPU and memory
# TODO: Put a co-http deployment live. Create a config and describe it.
# TODO: Test talking to multiple namespaces. Test kubeconfig file
# TODO: Adjust multiple resources at once
# Test describe an empty config. EXCLUDE_LABEL
# Version ID checks
# Timeouts, Encoders, refresh, ready
# Add watch, test create, read, delete, patch
# TODO: settlement time, recovery behavior, "adjust_on"?, restart detection, EXCLUDE_LABEL

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
                            'value': 1.0, #0.725,
                        },
                        'memory': {
                            'value': "2G", #2.25,
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