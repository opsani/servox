import asyncio
from kubetest.objects import namespace
from pydantic.error_wrappers import ValidationError
import pytest
import json
import yaml

from kubernetes_asyncio import client, config
from kubernetes_asyncio.client.api_client import ApiClient

from servo.connectors.kubernetes import CPU, Deployment, ResourceConstraint, Memory, Replicas, DeploymentConfiguration, ContainerConfiguration, KubernetesConfiguration, KubernetesConnector, Replicas, Pod, FailureMode, DNSSubdomainName, DNSLabelName, ContainerTagName
from servo.types import Component, Setting, SettingType
from pydantic import BaseModel
from typing import Type

class TestDNSSubdomainName:
    @pytest.fixture
    def model(self) -> Type[BaseModel]:
        class Model(BaseModel):
            name: DNSSubdomainName
        return Model

    def test_cannot_be_blank(self, model) -> None:
        valid_name = "ab"
        invalid_name = ""

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': 'ensure this value has at least 1 characters',
            'type': 'value_error.any_str.min_length',
            'ctx': {
                'limit_value': 1,
            },
        } in e.value.errors()
    
    def test_handles_uppercase_chars(self, model) -> None:
        valid_name = "ABCD"
        assert model(name=valid_name)
    
    def test_cannot_be_longer_than_253_chars(self, model) -> None:
        valid_name = "a" * 253
        invalid_name = valid_name + "b"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': 'ensure this value has at most 253 characters',
            'type': 'value_error.any_str.max_length',
            'ctx': {
                'limit_value': 253,
            },
        } in e.value.errors()

    
    def test_can_only_contain_alphanumerics_hyphens_and_dots(self, model) -> None:
        valid_name = "abcd1234.-sss"
        invalid_name = "abcd1234.-sss_$%!"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e        
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSSubdomainName.regex.pattern,
            },
        } in e.value.errors()
    
    def test_must_start_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "-abcd"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSSubdomainName.regex.pattern,
            },
        } in e.value.errors()
    
    def test_must_end_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "abcd-"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSSubdomainName.regex.pattern,
            },
        } in e.value.errors()


class TestDNSLabelName:
    @pytest.fixture
    def model(self) -> Type[BaseModel]:
        class Model(BaseModel):
            name: DNSLabelName
        return Model

    def test_cannot_be_blank(self, model) -> None:
        valid_name = "ab"
        invalid_name = ""

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': 'ensure this value has at least 1 characters',
            'type': 'value_error.any_str.min_length',
            'ctx': {
                'limit_value': 1,
            },
        } in e.value.errors()
    
    def test_handles_uppercase_chars(self, model) -> None:
        valid_name = "ABCD"
        assert model(name=valid_name)
    
    def test_cannot_be_longer_than_63_chars(self, model) -> None:
        valid_name = "a" * 63
        invalid_name = valid_name + "b"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': 'ensure this value has at most 63 characters',
            'type': 'value_error.any_str.max_length',
            'ctx': {
                'limit_value': 63,
            },
        } in e.value.errors()

    
    def test_can_only_contain_alphanumerics_and_hyphens(self, model) -> None:
        valid_name = "abcd1234-sss"
        invalid_name = "abcd1234.-sss_$%!"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e        
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSLabelName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSLabelName.regex.pattern,
            },
        } in e.value.errors()
    
    def test_must_start_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "-abcd"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSLabelName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSLabelName.regex.pattern,
            },
        } in e.value.errors()
    
    def test_must_end_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "abcd-"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': f'string does not match regex "{DNSLabelName.regex.pattern}"',
            'type': 'value_error.str.regex',
            'ctx': {
                'pattern': DNSLabelName.regex.pattern,
            },
        } in e.value.errors()

class TestContainerTagName:
    @pytest.fixture
    def model(self) -> Type[BaseModel]:
        class Model(BaseModel):
            name: ContainerTagName
        return Model

    def test_cant_be_more_than_128_characters(self, model) -> None:
        valid_name = "a" * 128
        invalid_name = valid_name + "b"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            'loc': ('name',),
            'msg': 'ensure this value has at most 128 characters',
            'type': 'value_error.any_str.max_length',
            'ctx': {
                'limit_value': 128,
            },
        } in e.value.errors()
    
    @pytest.mark.parametrize(
        "tag_name,valid",
        [
            ("image/tag:v1.0.0", True),
            ("123.123.123.123:123/image/tag:v1.0.0", True),
            ("your-domain.com/image/tag", True),
            ("your-domain.com/image/tag:v1.1.1-patch1", True),
            ("image/tag", True),
            ("image", True),
            ("image:v1.1.1-patch", True),
            ("ubuntu@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2", True),
            ("-", False),
            (".", False),
        ]
    )
    def test_tags(self, model, tag_name, valid) -> None:
        if valid:
            assert model(name=tag_name)
        else:
            with pytest.raises(ValidationError) as e:
                model(name=tag_name)
            assert e
            assert {
                'loc': ('name',),
                'msg': f'string does not match regex "{ContainerTagName.regex.pattern}"',
                'type': 'value_error.str.regex',
                'ctx': {
                    'pattern': ContainerTagName.regex.pattern,
                },
            } in e.value.errors()

class TestEnvironmentConfiguration:
    pass

class TestCommandConfiguration:
    pass

class TestKubernetesConfiguration:
    pass

class TestKubernetesConnector:
    pass

class TestContainerConfiguration:
    pass

class TestDeploymentConfiguration:
    pass

    def test_inheritance_of_default_namespace(self) -> None:
        ...
    

class TestReplicas:
    @pytest.fixture
    def replicas(self) -> Replicas:
        return Replicas(min=1, max=4)

    def test_parsing(self, replicas) -> None:
        assert {
            'name': 'replicas',
            'type': SettingType.RANGE,
            'min': 1,
            'max': 4,
            'step': 1,
            'value': None,
            'pinned': False,
        } == replicas.dict()
    
    def test_to_opsani_dict(self, replicas) -> None:
        replicas.value = "3"
        assert cpu.opsani_dict() == {
            'cpu': {
                'max': 4.0, 
                'min': 0.1, 
                'step': 0.125, 
                'value': 3.0,
                'type': SettingType.RANGE,
                'pinned': False
            }
        }

class TestCPU:
    @pytest.fixture
    def cpu(self) -> CPU:
        return CPU(min="100m", max=4, step="125m")

    def test_parsing(self, cpu) -> None:        
        assert {
            'name': 'cpu',
            'type': SettingType.RANGE,
            'min': 100,
            'max': 4000,
            'step': 125,
            'value': None,
            'pinned': False,
            'constraint': ResourceConstraint.both,
        } == cpu.dict()
    
    def test_to_opsani_dict(self, cpu) -> None:
        cpu.value = "3"
        assert cpu.opsani_dict() == {
            'cpu': {
                'max': 4.0, 
                'min': 0.1, 
                'step': 0.125, 
                'value': 3.0,
                'type': SettingType.RANGE,
                'pinned': False
            }
        }
    
    def test_validates_value_in_range(self, cpu) -> None:
        ...

class TestMemory:
    @pytest.fixture
    def memory(self) -> Memory:
        return Memory(min="128 MiB", max="4.0 GiB", step="0.25 GiB")

    def test_parsing(self, memory) -> None:
        assert {
            'name': 'memory',
            'type': SettingType.RANGE,
            'min': 134217728,
            'max': 4294967296,
            'step': 268435456,
            'value': None,
            'pinned': False,
            'constraint': ResourceConstraint.both,
        } == memory.dict()
    
    def test_to_opsani_dict(self, memory) -> None:
        memory.value = "3.0 GiB"
        assert memory.opsani_dict() == {
            'memory': {
                'max': 4.0, 
                'min': 0.125, 
                'step': 0.25, 
                'value': 3.0,
                'type': SettingType.RANGE,
                'pinned': False
            }
        }
    
    def test_handling_float_input(self) -> None:
        memory = Memory(min=0.5, max=4.0, step=0.125, value="3.0 GiB")
        assert memory.opsani_dict() == {
            'memory': {
                'max': 4.0, 
                'min': 0.5, 
                'step': 0.125, 
                'value': 3.0,
                'type': SettingType.RANGE,
                'pinned': False
            }
        }
    
    def test_validates_value_in_range(self, cpu) -> None:
        ...

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

def test_config():
    debug(KubernetesConfiguration.generate())

async def test_read_pod(config, adjustment):
    connector = KubernetesConnector(config=config)
    await config.load_kubeconfig()
    # dep = await Deployment.read("opsani-servo", "default")
    # debug(dep)
    pod = await Pod.read("web-canary", "default")
    debug(pod)
    # description = await connector.adjust(descriptor_to_adjustments(adjustment))
    # debug(description)

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

def test_proposed_config(canary_config) -> None:    
    dep = DeploymentComponent(
        name="co-http",
        container="main",
        namespace="web-apps",
        cpu=CPU(
            min="100m",
            max="800m",
            step="125m",
            value="300m",
        ),
        memory=Memory(
            min="0.1 GiB",
            max="0.8 GiB",
            step="125 MiB",
            value="500 MiB",
        ),
        replicas=Replicas(
            min=1,
            max=2,
            step=1,
            value=1,
        ),
        settings=[
        ]
    )
    dep_json = dep.json(by_alias=True, exclude_defaults=False, exclude_unset=False, exclude_none=False)
    debug(dep_json)
    dep_yaml = yaml.dump(json.loads(dep_json))
    debug(dep_yaml)
    
## 
# Canary Tests
async def test_create_canary(canary_config, adjustment):
    connector = KubernetesConnector(config=canary_config)
    description = await connector.startup()
    debug(description)

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
def canary_config(config) -> KubernetesConfiguration:
    canary_config = config.copy()
    for dep in canary_config.deployments:
        dep.strategy = "canary"
    return canary_config

@pytest.fixture
def config() -> KubernetesConfiguration:
    return KubernetesConfiguration(
        namespace="default",
        deployments=[
            DeploymentConfiguration(
                name="co-http-deployment",
                replicas=Replicas(
                    min=1,
                    max=2,
                ),
                containers=[
                    ContainerConfiguration(
                        name="opsani/co-http:latest",
                        cpu=CPU(
                            min="100m",
                            max="800m",
                            step="125m"
                        ),
                        memory=Memory(
                            min="100 MiB",
                            max="0.8 GiB",
                            step="128 MiB"
                        )
                    )
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