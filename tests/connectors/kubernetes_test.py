from __future__ import annotations

from typing import Type

import httpx
import kubetest.client
from kubetest.objects import Deployment as KubetestDeployment
import kubernetes.client.models
import kubernetes.client.exceptions
import platform
import pydantic
import pytest
import pytest_mock
import re
import respx
import traceback
from kubernetes_asyncio import client
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError

import servo.connectors.kubernetes
from servo.connectors.kubernetes import (
    CPU,
    CanaryOptimization,
    CanaryOptimizationStrategyConfiguration,
    Container,
    ContainerConfiguration,
    ContainerTagName,
    DefaultOptimizationStrategyConfiguration,
    Deployment,
    DeploymentConfiguration,
    DNSLabelName,
    DNSSubdomainName,
    FailureMode,
    KubernetesChecks,
    KubernetesConfiguration,
    KubernetesConnector,
    Memory,
    Millicore,
    OptimizationStrategy,
    Pod,
    ResourceRequirement,
    Rollout,
    RolloutConfiguration,
)
from servo.errors import AdjustmentFailedError, AdjustmentRejectedError
import servo.runner
from servo.types import Adjustment
from tests.helpers import *


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
            "loc": ("name",),
            "msg": "ensure this value has at least 1 characters",
            "type": "value_error.any_str.min_length",
            "ctx": {
                "limit_value": 1,
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
            "loc": ("name",),
            "msg": "ensure this value has at most 253 characters",
            "type": "value_error.any_str.max_length",
            "ctx": {
                "limit_value": 253,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSSubdomainName.regex.pattern,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSSubdomainName.regex.pattern,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSSubdomainName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSSubdomainName.regex.pattern,
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
            "loc": ("name",),
            "msg": "ensure this value has at least 1 characters",
            "type": "value_error.any_str.min_length",
            "ctx": {
                "limit_value": 1,
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
            "loc": ("name",),
            "msg": "ensure this value has at most 63 characters",
            "type": "value_error.any_str.max_length",
            "ctx": {
                "limit_value": 63,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSLabelName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSLabelName.regex.pattern,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSLabelName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSLabelName.regex.pattern,
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
            "loc": ("name",),
            "msg": f'string does not match regex "{DNSLabelName.regex.pattern}"',
            "type": "value_error.str.regex",
            "ctx": {
                "pattern": DNSLabelName.regex.pattern,
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
            "loc": ("name",),
            "msg": "ensure this value has at most 128 characters",
            "type": "value_error.any_str.max_length",
            "ctx": {
                "limit_value": 128,
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
            (
                "ubuntu@sha256:45b23dee08af5e43a7fea6c4cf9c25ccf269ee113168c19722f87876677c5cb2",
                True,
            ),
            ("-", False),
            (".", False),
        ],
    )
    def test_tags(self, model, tag_name, valid) -> None:
        if valid:
            assert model(name=tag_name)
        else:
            with pytest.raises(ValidationError) as e:
                model(name=tag_name)
            assert e
            assert {
                "loc": ("name",),
                "msg": f'string does not match regex "{ContainerTagName.regex.pattern}"',
                "type": "value_error.str.regex",
                "ctx": {
                    "pattern": ContainerTagName.regex.pattern,
                },
            } in e.value.errors()


class TestEnvironmentConfiguration:
    pass


class TestCommandConfiguration:
    pass


class TestKubernetesConfiguration:
    @pytest.fixture
    def funkytown(self, config: KubernetesConfiguration) -> KubernetesConfiguration:
        return config.copy(update={"namespace": "funkytown"})

    def test_cascading_defaults(self, config: KubernetesConfiguration) -> None:
        # Verify that by default we get a null namespace
        assert DeploymentConfiguration.__fields__["namespace"].default is None
        assert (
            DeploymentConfiguration(
                name="testing", containers=[], replicas=servo.Replicas(min=0, max=1)
            ).namespace
            is None
        )

        # Verify that we inherit when nested
        assert config.namespace == "default"
        assert config.deployments[0].namespace == "default"

    def test_explicit_cascade(self, config: KubernetesConfiguration) -> None:
        model = config.copy(update={"namespace": "funkytown"})
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "default"

        model.cascade_common_settings(overwrite=True)
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "funkytown"

    def test_respects_explicit_override(self, config: KubernetesConfiguration) -> None:
        # set the property explictly to value equal to default, then trigger
        model = config.copy(update={"namespace": "funkytown"})
        model.deployments[0].namespace = "default"
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "default"

        model.cascade_common_settings()
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "default"

    @pytest.mark.parametrize(
        "yaml_path, expected_value",
        [
            # CPU in millicores
            ("deployments[0].containers[0].cpu.min", "250m"),
            ("deployments[0].containers[0].cpu.max", "4"),
            ("deployments[0].containers[0].cpu.step", "125m"),
            # Memory
            ("deployments[0].containers[0].memory.min", "256.0Mi"),
            ("deployments[0].containers[0].memory.max", "4.0Gi"),
            ("deployments[0].containers[0].memory.step", "128.0Mi"),
        ],
    )
    def test_generate_emits_human_readable_values(
        self, yaml_path, expected_value
    ) -> None:
        # import yamlpath
        config = KubernetesConfiguration.generate()

        # assert yaml_key_path(config.yaml(), key_path) == expected_value

        from yamlpath import Processor, YAMLPath
        from yamlpath.func import get_yaml_editor

        # Process command-line arguments and initialize the output writer
        # args = processcli()
        # log = ConsolePrinter(args)
        # Prep the YAML parser and round-trip editor (tweak to your needs)
        yaml = get_yaml_editor()

        # At this point, you'd load or parse your YAML file, stream, or string.  When
        # loading from file, I typically follow this pattern:
        # yaml_data = get_yaml_data(yaml, logger, config.yaml())
        yaml_data = yaml.load(config.yaml())
        assert yaml_data

        processor = Processor(logger, yaml_data)
        path = YAMLPath(yaml_path)
        matches = list(processor.get_nodes(path))
        assert len(matches) == 1, "expected only a single matching node"
        assert matches[0].node == expected_value


class TestKubernetesConnector:
    pass


class TestContainerConfiguration:
    pass


class TestDeploymentConfiguration:
    def test_inheritance_of_default_namespace(self) -> None:
        ...

    def test_strategy_enum(self) -> None:
        config = DeploymentConfiguration(
            name="testing",
            containers=[],
            replicas=servo.Replicas(min=1, max=4),
            strategy=OptimizationStrategy.default,
        )
        assert config.yaml(exclude_unset=True) == (
            "name: testing\n"
            "containers: []\n"
            "strategy: default\n"
            "replicas:\n"
            "  min: 1\n"
            "  max: 4\n"
        )

    def test_strategy_object_default(self) -> None:
        config = DeploymentConfiguration(
            name="testing",
            containers=[],
            replicas=servo.Replicas(min=1, max=4),
            strategy=DefaultOptimizationStrategyConfiguration(
                type=OptimizationStrategy.default
            ),
        )
        assert config.yaml(exclude_unset=True) == (
            "name: testing\n"
            "containers: []\n"
            "strategy:\n"
            "  type: default\n"
            "replicas:\n"
            "  min: 1\n"
            "  max: 4\n"
        )

    def test_strategy_object_canary(self) -> None:
        config = DeploymentConfiguration(
            name="testing",
            containers=[],
            replicas=servo.Replicas(min=1, max=4),
            strategy=CanaryOptimizationStrategyConfiguration(
                type=OptimizationStrategy.canary, alias="tuning"
            ),
        )
        assert config.yaml(exclude_unset=True) == (
            "name: testing\n"
            "containers: []\n"
            "strategy:\n"
            "  type: canary\n"
            "  alias: tuning\n"
            "replicas:\n"
            "  min: 1\n"
            "  max: 4\n"
        )

    def test_strategy_object_default_parsing(self) -> None:
        config_yaml = (
            "containers: []\n"
            "name: testing\n"
            "replicas:\n"
            "  max: 4\n"
            "  min: 1\n"
            "strategy:\n"
            "  type: default\n"
        )
        config_dict = yaml.load(config_yaml, Loader=yaml.FullLoader)
        config = DeploymentConfiguration.parse_obj(config_dict)
        assert isinstance(config.strategy, DefaultOptimizationStrategyConfiguration)
        assert config.strategy.type == OptimizationStrategy.default

    def test_strategy_object_tuning_parsing(self) -> None:
        config_yaml = (
            "containers: []\n"
            "name: testing\n"
            "replicas:\n"
            "  max: 4\n"
            "  min: 1\n"
            "strategy:\n"
            "  type: canary\n"
        )
        config_dict = yaml.load(config_yaml, Loader=yaml.FullLoader)
        config = DeploymentConfiguration.parse_obj(config_dict)
        assert isinstance(config.strategy, CanaryOptimizationStrategyConfiguration)
        assert config.strategy.type == OptimizationStrategy.canary
        assert config.strategy.alias is None

    def test_strategy_object_tuning_parsing_with_alias(self) -> None:
        config_yaml = (
            "containers: []\n"
            "name: testing\n"
            "replicas:\n"
            "  max: 4\n"
            "  min: 1\n"
            "strategy:\n"
            "  alias: tuning\n"
            "  type: canary\n"
        )
        config_dict = yaml.load(config_yaml, Loader=yaml.FullLoader)
        config = DeploymentConfiguration.parse_obj(config_dict)
        assert isinstance(config.strategy, CanaryOptimizationStrategyConfiguration)
        assert config.strategy.type == OptimizationStrategy.canary
        assert config.strategy.alias == "tuning"


class TestCanaryOptimization:
    @pytest.mark.xfail
    def test_to_components_default_name(self, config) -> None:
        config.deployments[0].strategy = OptimizationStrategy.canary
        optimization = CanaryOptimization.construct(
            name="fiber-http-deployment/opsani/fiber-http:latest-canary",
            target_deployment_config=config.deployments[0],
            target_container_config=config.deployments[0].containers[0],
        )
        assert (
            optimization.target_name == "fiber-http-deployment/opsani/fiber-http:latest"
        )
        assert (
            optimization.tuning_name
            == "fiber-http-deployment/opsani/fiber-http:latest-canary"
        )

    @pytest.mark.xfail
    def test_to_components_respects_aliases(self, config) -> None:
        config.deployments[0].strategy = CanaryOptimizationStrategyConfiguration(
            type=OptimizationStrategy.canary, alias="tuning"
        )
        config.deployments[0].containers[0].alias = "main"
        optimization = CanaryOptimization.construct(
            name="fiber-http-deployment/opsani/fiber-http:latest-canary",
            target_deployment_config=config.deployments[0],
            target_container_config=config.deployments[0].containers[0],
        )
        assert optimization.target_name == "main"
        assert optimization.tuning_name == "tuning"


def test_compare_strategy() -> None:
    config = CanaryOptimizationStrategyConfiguration(
        type=OptimizationStrategy.canary, alias="tuning"
    )
    assert config == OptimizationStrategy.canary


# class TestResourceRequirements:
#     @pytest.mark.parametrize(
#         "requirement, val",
#         [
#             (ResourceRequirements.limit, True),
#             (ResourceRequirements.request, True),
#             (ResourceRequirements.compute, False),
#         ],
#     )
#     def test_flag_introspection(self, requirement, val) -> None:
#         assert requirement.flag is val
#         assert requirement.flags is not val


# class TestContainer:
#     @pytest.fixture
#     def container(self, mocker) -> Container:
#         stub_pod = mocker.stub(name="Pod")
#         return Container(client.V1Container(name="container"), stub_pod)

#     @pytest.mark.parametrize(
#         "name, requirements, kwargs, value",
#         [
#             ("cpu", ..., ..., ("100m", "15000m")),
#             ("cpu", ResourceRequirements.compute, ..., ("100m", "15000m")),
#             ("cpu", ResourceRequirements.request, ..., ("100m",)),
#             ("cpu", ResourceRequirements.limit, dict(first=True), "15000m"),
#             (
#                 "cpu",
#                 ResourceRequirements.compute,
#                 dict(first=True, reverse=True),
#                 "15000m",
#             ),
#             ("memory", ..., ..., ("3G", None)),
#             ("memory", ResourceRequirements.compute, ..., ("3G", None)),
#             ("memory", ResourceRequirements.request, ..., ("3G",)),
#             ("memory", ResourceRequirements.compute, dict(first=True), "3G"),
#             ("memory", ResourceRequirements.request, dict(first=True), "3G"),
#             ("memory", ResourceRequirements.limit, dict(first=True), None),
#             (
#                 "memory",
#                 ResourceRequirements.limit,
#                 dict(first=True, default="1TB"),
#                 "1TB",
#             ),
#             ("invalid", ResourceRequirements.compute, ..., (None, None)),
#             (
#                 "invalid",
#                 ResourceRequirements.compute,
#                 dict(first=True, default="3.125"),
#                 "3.125",
#             ),
#         ],
#     )
#     def test_get_resource_requirements(
#         self, container, name, requirements, kwargs, value
#     ) -> None:
#         resources = client.V1ResourceRequirements()
#         resources.requests = {"cpu": "100m", "memory": "3G"}
#         resources.limits = {"cpu": "15000m"}
#         container.resources = resources

#         # Support testing default arguments
#         if requirements == ...:
#             requirements = container.get_resource_requirements.__defaults__[0]
#         if kwargs == ...:
#             kwargs = container.get_resource_requirements.__kwdefaults__

#         assert (
#             container.get_resource_requirements(name, requirements, **kwargs) == value
#         )

#     @pytest.mark.parametrize(
#         "name, value, requirements, kwargs, resources_dict",
#         [
#             (
#                 "cpu",
#                 ("100m", "250m"),
#                 ...,
#                 ...,
#                 {"limits": {"cpu": "250m"}, "requests": {"cpu": "100m", "memory": "3G"}},
#             ),
#             (
#                 "cpu",
#                 "500m",
#                 ResourceRequirements.limit,
#                 dict(clear_others=True),
#                 {"limits": {"cpu": "500m"}, "requests": {"memory": "3G"}},
#             ),
#         ],
#     )
#     def test_set_resource_requirements(
#         self, container, name, value, requirements, kwargs, resources_dict
#     ) -> None:
#         resources = client.V1ResourceRequirements()
#         resources.requests = {"cpu": "100m", "memory": "3G"}
#         resources.limits = {"cpu": "15000m"}
#         container.resources = resources

#         # Support testing default arguments
#         if requirements == ...:
#             requirements = container.set_resource_requirements.__defaults__[0]
#         if kwargs == ...:
#             kwargs = container.set_resource_requirements.__kwdefaults__

#         container.set_resource_requirements(name, value, requirements, **kwargs)
#         assert container.resources.to_dict() == resources_dict

#     def test_set_resource_requirements_handles_null_requirements_dict(self, container):
        # container.resources = client.V1ResourceRequirements()

        # container.set_resource_requirements("cpu", "1000m")
        # assert container.resources.to_dict() == {
        #     "limits": {"cpu": "1000m"},
        #     "requests": {"cpu": "1000m"},
        # }


class TestReplicas:
    @pytest.fixture
    def replicas(self) -> servo.Replicas:
        return servo.Replicas(min=1, max=4)

    def test_parsing(self, replicas) -> None:
        assert {
            "name": "replicas",
            "type": "range",
            "min": 1,
            "max": 4,
            "step": 1,
            "value": None,
            "pinned": False,
        } == replicas.dict()

    def test_to___opsani_repr__(self, replicas) -> None:
        replicas.value = 3
        assert replicas.__opsani_repr__() == {
            "replicas": {
                "max": 4.0,
                "min": 1.0,
                "step": 1,
                "value": 3.0,
                "type": "range",
                "pinned": False,
            }
        }


class TestCPU:
    @pytest.fixture
    def cpu(self) -> CPU:
        return CPU(min="125m", max="4000m", step="125m")

    def test_parsing(self, cpu) -> None:
        assert {
            "name": "cpu",
            "type": "range",
            "min": 125,
            "max": 4000,
            "step": 125,
            "value": None,
            "pinned": False,
            'request': None,
            'limit': None,
            'get': [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
            'set': [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ]
        } == cpu.dict()

    def test_to___opsani_repr__(self, cpu) -> None:
        cpu.value = "3"
        assert cpu.__opsani_repr__() == {
            "cpu": {
                "max": 4.0,
                "min": 0.125,
                "step": 0.125,
                "value": 3.0,
                "type": "range",
                "pinned": False,
            }
        }

    def test_resolving_equivalent_units(self) -> None:
        cpu = CPU(min="125m", max=4.0, step=0.125)
        assert cpu.min == 125
        assert cpu.max == 4000
        assert cpu.step == 125

    def test_resources_encode_to_json_human_readable(self, cpu) -> None:
        serialization = json.loads(cpu.json())
        assert serialization["min"] == "125m"
        assert serialization["max"] == "4"
        assert serialization["step"] == "125m"

    def test_min_cannot_be_less_than_step(self) -> None:
        with pytest.raises(ValueError, match=re.escape('min cannot be less than step (125m < 250m)')):
            CPU(min="125m", max=4.0, step=0.250)


class TestMillicore:
    @pytest.mark.parametrize(
        "input, millicores",
        [
            ("100m", 100),
            ("1", 1000),
            (1, 1000),
            ("0.1", 100),
            (0.1, 100),
            (2.0, 2000),
            ("2.0", 2000),
        ],
    )
    def test_parsing(self, input: Union[str, int, float], millicores: int) -> None:
        assert Millicore.parse(input) == millicores

    @pytest.mark.parametrize(
        "input, output",
        [
            ("100m", "100m"),
            ("1", "1"),
            ("1.0", "1"),
            (1, "1"),
            ("0.1", "100m"),
            (0.1, "100m"),
            (2.5, "2500m"),
            ("123m", "123m"),
        ],
    )
    def test_string_serialization(
        self, input: Union[str, int, float], output: str
    ) -> None:
        millicores = Millicore.parse(input)
        assert str(millicores) == output


class TestMemory:
    @pytest.fixture
    def memory(self) -> Memory:
        return Memory(min="0.25 GiB", max="4.0 GiB", step="128 MiB")

    def test_parsing(self, memory) -> None:
        assert {
            'name': 'mem',
            'type': 'range',
            'pinned': False,
            'value': None,
            'min': 268435456,
            'max': 4294967296,
            'step': 134217728,
            'request': None,
            'limit': None,
            'get': [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
            'set': [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
        } == memory.dict()

    def test_to___opsani_repr__(self, memory) -> None:
        memory.value = "3.0 GiB"
        assert memory.__opsani_repr__() == {
            "mem": {
                "max": 4.0,
                "min": 0.25,
                "step": 0.125,
                "value": 3.0,
                "type": "range",
                "pinned": False,
            }
        }

    def test_handling_float_input(self) -> None:
        memory = Memory(min=0.5, max=4.0, step=0.125, value="3.0 GiB")
        assert memory.__opsani_repr__() == {
            "mem": {
                "max": 4.0,
                "min": 0.5,
                "step": 0.125,
                "value": 3.0,
                "type": "range",
                "pinned": False,
            }
        }

    def test_resolving_equivalent_units(self) -> None:
        memory = Memory(min=268435456, max=4.0, step="128 MiB")
        assert memory.min == 268435456
        assert memory.max == 4294967296
        assert memory.step == 134217728

    def test_resources_encode_to_json_human_readable(self, memory) -> None:
        serialization = json.loads(memory.json())
        assert serialization["min"] == "256.0Mi"
        assert serialization["max"] == "4.0Gi"
        assert serialization["step"] == "128.0Mi"

    def test_min_cannot_be_less_than_step(self) -> None:
        with pytest.raises(ValueError, match=re.escape('min cannot be less than step (33554432 < 268435456)')):
            Memory(min="32 MiB", max=4.0, step=268435456)

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
def tuning_config(config) -> KubernetesConfiguration:
    tuning_config = config.copy()
    for dep in tuning_config.deployments:
        dep.strategy = "canary"
    return tuning_config

@pytest.fixture
def namespace() -> str:
    return "default"

@pytest.fixture
def config(namespace: str) -> KubernetesConfiguration:
    return KubernetesConfiguration(
        namespace=namespace,
        deployments=[
            DeploymentConfiguration(
                name="fiber-http",
                replicas=servo.Replicas(
                    min=1,
                    max=4,
                ),
                containers=[
                    ContainerConfiguration(
                        name="fiber-http",
                        cpu=CPU(min="125m", max="875m", step="125m"),
                        memory=Memory(min="128MiB", max="0.75GiB", step="32MiB"),
                    )
                ],
            )
        ],
    )

@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-opsani-dev.yaml"])
class TestKubernetesConnectorIntegration:
    @pytest.fixture(autouse=True)
    async def _wait_for_manifests(self, kube, config):
        kube.wait_for_registered()
        config.timeout = "5m"

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    async def test_describe(self, config) -> None:
        connector = KubernetesConnector(config=config)
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 125
        assert description.get_setting("fiber-http/fiber-http.mem").human_readable_value == "128.0Mi"
        assert description.get_setting("fiber-http/fiber-http.replicas").value == 1

    async def test_adjust_cpu(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".150",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.cpu')
        assert setting
        assert setting.value == 150

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 150

    async def test_adjust_cpu_with_settlement(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".250",
        )
        control = servo.Control(settlement='1s')
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.cpu')
        assert setting
        assert setting.value == 250

    async def test_adjust_cpu_at_non_zero_container_index(self, config):
        # Inject a sidecar at index zero
        deployment = await servo.connectors.kubernetes.Deployment.read('fiber-http', config.namespace)
        assert deployment, f"failed loading deployment 'fiber-http' in namespace '{config.namespace}'"
        async with deployment.rollout(timeout=config.timeout) as deployment_update:
            await deployment_update.inject_sidecar('opsani-envoy', 'opsani/envoy-proxy:latest', port="8480", service_port=8091, index=0)

        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".250",
        )

        control = servo.Control(settlement='1s')
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.cpu')
        assert setting
        assert setting.value == 250

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 250

    async def test_adjust_cpu_matchlabels_dont_match_metadata_labels(self, config, kube: kubetest.client.TestClient):
        deployments = kube.get_deployments()
        target_deploy = deployments.get("fiber-http")
        assert target_deploy is not None

        # Update metadata labels so they don't match the match_labels selector
        target_deploy.obj.metadata.labels["app.kubernetes.io/name"] = "web"
        target_deploy.api_client.patch_namespaced_deployment(target_deploy.name, target_deploy.namespace, target_deploy.obj)
        kube.wait_for_registered()

        config.timeout = "15s"
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".150",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.cpu')
        assert setting
        assert setting.value == 150

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 150

    async def test_adjust_memory(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="700Mi",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.mem')
        assert setting
        assert setting.value == 734003200

        # Get deployment and check the pods
        # deployment = await Deployment.read("web", "default")
        # debug(deployment)
        # debug(deployment.obj.spec.template.spec.containers)

    async def test_adjust_deployment_insufficient_resources(self, config: KubernetesConfiguration):
        config.timeout = "3s"
        config.deployments[0].containers[0].memory.max = "256Gi"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Gi",
        )
        with pytest.raises(
            AdjustmentRejectedError,
            match=(
                re.escape("Requested adjustment(s) (fiber-http/fiber-http.mem=128Gi) cannot be scheduled due to ")
                + r"\"\d+/\d+ nodes are available: \d+ Insufficient memory\.\""
            )
        ) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert rejection_info.value.reason == "unschedulable"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_deployment_image_pull_backoff(
        self,
        config: KubernetesConfiguration,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        servo.logging.set_level("TRACE")
        config.timeout = "10s"
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="256Mi",
        )

        mocker.patch(
            "kubernetes_asyncio.client.models.v1_container.V1Container.image",
            new_callable=mocker.PropertyMock,
            return_value="opsani/bababooey:latest"
        )

        with pytest.raises(AdjustmentFailedError, match="Container image pull failure detected"):
            await connector.adjust([adjustment])

    async def test_adjust_replicas(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="replicas",
            value="2",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.replicas')
        assert setting
        assert setting.value == 2

    async def test_read_pod(self, config, kube) -> None:
        connector = KubernetesConnector(config=config)
        pods = kube.get_pods()
        pod_name = next(iter(pods.keys()))
        assert pod_name.startswith("fiber-http")
        pod = await Pod.read(pod_name, kube.namespace)
        assert pod

    ##
    # Canary Tests
    # async def test_create_canary(self, tuning_config, namespace: str) -> None:
   #      connector = KubernetesConnector(config=tuning_config)
   #      dep = await Deployment.read("fiber-http", namespace)
   #      debug(dep)
        # description = await connector.startup()
        # debug(description)

    async def test_adjust_tuning_insufficient_mem(
        self,
        tuning_config: KubernetesConfiguration
    ) -> None:
        tuning_config.timeout = "10s"
        tuning_config.deployments[0].containers[0].memory = Memory(min="128MiB", max="128GiB", step="32MiB")
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Gi", # impossible right?
        )
        with pytest.raises(
            AdjustmentRejectedError,
            match=(
                re.escape("Requested adjustment(s) (fiber-http/fiber-http-tuning.mem=128Gi) cannot be scheduled due to ")
                + r"\"\d+/\d+ nodes are available: \d+ Insufficient memory\.\""
            )
        ) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert rejection_info.value.reason == "unschedulable"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_tuning_insufficient_cpu_and_mem(
        self,
        tuning_config: KubernetesConfiguration
    ) -> None:
        tuning_config.timeout = "10s"
        tuning_config.deployments[0].containers[0].memory = Memory(min="128MiB", max="128GiB", step="32MiB")
        tuning_config.deployments[0].containers[0].cpu = CPU(min="125m", max="200", step="125m")
        connector = KubernetesConnector(config=tuning_config)

        adjustments = [
            Adjustment(
                component_name="fiber-http/fiber-http-tuning",
                setting_name="mem",
                value="128Gi", # impossible right?
            ),
            Adjustment(
                component_name="fiber-http/fiber-http-tuning",
                setting_name="cpu",
                value="100", # impossible right?
            )
        ]
        with pytest.raises(
            AdjustmentRejectedError,
            match=(
                re.escape("Requested adjustment(s) (fiber-http/fiber-http-tuning.mem=128Gi, fiber-http/fiber-http-tuning.cpu=100) cannot be scheduled due to ")
                + r"\"\d+/\d+ nodes are available: \d+ Insufficient cpu\, \d+ Insufficient memory\.\""
            )
        ) as rejection_info:
            await connector.adjust(adjustments)

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert rejection_info.value.reason == "unschedulable"
        except AssertionError as e:
            raise e from rejection_info.value


    async def test_create_tuning_image_pull_backoff(
        self,
        tuning_config: KubernetesConfiguration,
        mocker: pytest_mock.MockerFixture,
        kube
    ) -> None:
        tuning_config.timeout = "10s"
        connector = KubernetesConnector(config=tuning_config)

        mocker.patch(
            "kubernetes_asyncio.client.models.v1_container.V1Container.image",
            new_callable=mocker.PropertyMock,
            return_value="opsani/bababooey:latest"
        )

        # NOTE: describe logic currently invokes the same creation as adjust and allows for a faster test.
        # If tuning creation is removed from describe this test will need to be refactored and have a longer timeout and runtime
        with pytest.raises(AdjustmentFailedError, match="Container image pull failure detected"):
            await connector.describe()


    async def test_bad_request_error_handled_gracefully(self, tuning_config: KubernetesConfiguration, mocker: pytest_mock.MockerFixture) -> None:
        """Verify a failure to create a pod is not poorly handled in the handle_error destroy logic"""

        # Passing in an intentionally mangled memory setting to trigger an API error that prevents pod creation
        mocker.patch("servo.connectors.kubernetes.Memory.__config__.validate_assignment", new_callable=mocker.PropertyMock(return_value=False))
        mocker.patch("servo.connectors.kubernetes._normalize_adjustment", return_value=("memory","256.0MiBGiB"))

        tuning_config.deployments[0].on_failure = FailureMode.rollback
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="256Mi",
        )

        # Catch info log messages
        messages = []
        connector.logger.add(lambda m: messages.append(m.record['message']), level=10)

        with pytest.raises(kubernetes_asyncio.client.exceptions.ApiException) as error:
            await connector.adjust([adjustment])

        # Check logs
        assert 'no tuning pod exists, ignoring destroy' in messages[-30:]
        # Check error
        assert 'quantities must match the regular expression' in str(error.value)
        assert error.value.status == 400


    async def test_adjust_tuning_cpu_with_settlement(self, tuning_config, namespace, kube):
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )

        control = servo.Control(settlement='50ms')
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert setting
        assert setting.value == 250

    async def test_adjust_handle_error_respects_nested_config(self, config: KubernetesConfiguration, kube: kubetest.client.TestClient):
        config.timeout = "3s"
        config.on_failure = FailureMode.destroy
        config.deployments[0].on_failure = FailureMode.exception
        config.deployments[0].containers[0].memory.max = "256Gi"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Gi",
        )
        with pytest.raises(AdjustmentRejectedError, match="Insufficient memory.") as rejection_info:
            description = await connector.adjust([adjustment])
            debug(description)

        await Deployment.read("fiber-http", kube.namespace)

    # async def test_apply_no_changes(self):
#         # resource_version stays the same and early exits
#         pass
#
#
#     async def test_apply_metadata_changes(self):
#         # Update labels or something that doesn't matter
#         # Detect by never getting a progressing event
#         pass
#
#
#     async def test_apply_replica_change(self):
#         # bump the count, observed_generation goes up
#         # wait for the counts to settle
#         ...
#
#
#     async def test_apply_memory_change(self):
#         # bump the count, observed_generation goes up
#         # wait for the counts to settle
#         ...
#
#
#     async def test_apply_cpu_change(self):
#         # bump the count, observed_generation goes up
#         # wait for the counts to settle
#         ...
#
#
#     async def test_apply_unschedulable_memory_request(self):
#         # bump the count, observed_generation goes up
#         # wait for the counts to settle
#         ...
#
#
#     async def test_apply_restart_strategy(self):
#         # Make sure we can watch a non-rolling update
#         # .spec.strategy specifies the strategy used to replace old Pods by new ones. .spec.strategy.type can be "Recreate" or "RollingUpdate". "RollingUpdate" is the default value.
#         # Recreate Deployment
#         ...


    # TODO: Put a fiber-http deployment live. Create a config and describe it.
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

    async def test_checks(self, config: KubernetesConfiguration):
        await KubernetesChecks.run(config)

    # Deployment readiness check was returning false positives, guard against regression
    async def test_check_deployment_readiness_failure(self, config: KubernetesConfiguration, kube: kubetest.client.TestClient):
        deployments = kube.get_deployments()
        target_deploy = deployments.get("fiber-http")
        assert target_deploy is not None

        target_container = next(filter(lambda c: c.name == "fiber-http", target_deploy.obj.spec.template.spec.containers))
        assert target_container is not None

        # Update to put deployment in unready state
        target_container.readiness_probe = kubernetes.client.models.V1Probe(
            _exec=kubernetes.client.models.V1ExecAction(command=["exit", "1"]),
            failure_threshold=1
        )
        target_deploy.obj.spec.strategy.rolling_update.max_surge = '0%'
        target_deploy.api_client.patch_namespaced_deployment(target_deploy.name, target_deploy.namespace, target_deploy.obj)

        while target_deploy.is_ready():
            await asyncio.sleep(0.1)

        result = await KubernetesChecks(config).run_one(id="check_deployments_are_ready_item_0")
        assert result.success == False and result.message == "caught exception (RuntimeError): Deployment \"fiber-http\" is not ready"


##
# Rejection Tests using modified deployment, skips the standard manifest application
@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesConnectorIntegrationUnreadyCmd:
    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.fixture
    def kubetest_deployment(self, kube: kubetest.client.TestClient, rootpath: pathlib.Path) -> KubetestDeployment:
        deployment = kube.load_deployment(rootpath.joinpath("tests/manifests/fiber-http-opsani-dev.yaml"))
        deployment.obj.spec.template.spec.termination_grace_period_seconds = 10
        fiber_container = deployment.obj.spec.template.spec.containers[0]
        fiber_container.resources.requests['memory'] = '256Mi'
        fiber_container.resources.limits['memory'] = '256Mi'
        fiber_container.readiness_probe = kubernetes.client.models.V1Probe(
            failure_threshold=3,
            http_get=kubernetes.client.models.V1HTTPGetAction(
              path= "/",
              port= 9980,
              scheme="HTTP",
            ),
            initial_delay_seconds=1,
            period_seconds=5,
            success_threshold=1,
            timeout_seconds=1,
        )

        return deployment

    @pytest.fixture
    def kubetest_deployment_never_ready(self, kubetest_deployment: KubetestDeployment) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = [ "/bin/sh" ]
        # Simulate a deployment which fails to start when memory adjusted to < 192Mi
        fiber_container.args = [
            "-c", "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; then /bin/fiber-http; else sleep 1d; fi"
        ]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment

    @pytest.fixture
    def kubetest_deployemnt_oom_killed(self, kubetest_deployment: KubetestDeployment) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = [ "/bin/sh" ]
        # Simulate a deployment which will be OOMKilled when memory adjusted to < 192Mi
        fiber_container.args = [ "-c", (
            "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; "
                "then /bin/fiber-http; "
                "else tail /dev/zero; "
            "fi"
        )]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment

    @pytest.fixture
    def kubetest_deployment_becomes_unready(self, kubetest_deployment: KubetestDeployment) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = [ "/bin/sh" ]
        # Simulate a deployment which passes initial readiness checks when memory adjusted to < 192Mi then fails them a short time later
        fiber_container.args = [ "-c", (
            "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; "
                "then /bin/fiber-http; "
                "else (/bin/fiber-http &); sleep 10s; kill %1; "
            "fi"
        )]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment



    async def test_adjust_deployment_never_ready(self, config: KubernetesConfiguration, kubetest_deployment_never_ready: KubetestDeployment) -> None:
        config.timeout = "3s"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Mi",
        )

        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert "(reason ContainersNotReady) containers with unready status: [fiber-http" in str(rejection_info.value)
            assert rejection_info.value.reason == "start-failed"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_deployment_oom_killed(self, config: KubernetesConfiguration, kubetest_deployemnt_oom_killed: KubetestDeployment) -> None:
        config.timeout = "10s"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Mi",
        )

        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert "Deployment fiber-http pod(s) crash restart detected: fiber-http-" in str(rejection_info.value)
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_deployment_settlement_failed(
        self,
        config: KubernetesConfiguration,
        kubetest_deployment_becomes_unready: KubetestDeployment
    ) -> None:
        config.timeout = "15s"
        config.settlement = "20s"
        config.deployments[0].on_failure = FailureMode.destroy
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Mi",
        )
        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert (
                "(reason ContainersNotReady) containers with unready status: [fiber-http]" in str(rejection_info.value)
                or "Deployment fiber-http pod(s) crash restart detected" in str(rejection_info.value)
            ), str(rejection_info.value)
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate deployment destroyed
        with pytest.raises(kubernetes.client.exceptions.ApiException) as not_found_error:
            kubetest_deployment_becomes_unready.refresh()

        assert not_found_error.value.status == 404 and not_found_error.value.reason == "Not Found", str(not_found_error.value)

    async def test_adjust_tuning_never_ready(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployment_never_ready: KubetestDeployment,
        kube: kubetest.client.TestClient
    ) -> None:
        tuning_config.timeout = "30s"
        tuning_config.on_failure = FailureMode.destroy
        tuning_config.deployments[0].on_failure = FailureMode.destroy
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Mi",
        )

        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert "(reason ContainersNotReady) containers with unready status: [fiber-http" in str(rejection_info.value)
            assert rejection_info.value.reason == "start-failed"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(filter(lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers))
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"

    async def test_adjust_tuning_oom_killed(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployemnt_oom_killed: KubetestDeployment,
        kube: kubetest.client.TestClient
    ) -> None:
        tuning_config.timeout = "25s"
        tuning_config.on_failure = FailureMode.destroy
        tuning_config.deployments[0].on_failure = FailureMode.destroy
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Mi",
        )

        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert "Tuning optimization fiber-http-tuning crash restart detected on container(s): fiber-http" in str(rejection_info.value)
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(filter(lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers))
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"

    async def test_adjust_tuning_settlement_failed(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployment_becomes_unready: KubetestDeployment,
        recwarn: pytest.WarningsRecorder,
        kube: kubetest.client.TestClient
    ) -> None:
        tuning_config.timeout = "25s"
        tuning_config.settlement = "15s"
        tuning_config.on_failure = FailureMode.destroy
        tuning_config.deployments[0].on_failure = FailureMode.destroy
        connector = KubernetesConnector(config=tuning_config)


        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Mi",
        )
        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            await connector.adjust([adjustment])

        # Validate raised warnings to ensure all coroutines were awaited
        assert not any(filter(lambda warn: "was never awaited" in warn.message, recwarn)), list(map(lambda warn: warn.message, recwarn))

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert (
                "(reason ContainersNotReady) containers with unready status: [fiber-http]" in str(rejection_info.value)
                or "Tuning optimization fiber-http-tuning crash restart detected on container(s): fiber-http" in str(rejection_info.value)
            )
            rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(filter(lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers))
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"


@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesResourceRequirementsIntegration:
    @pytest.fixture(autouse=True)
    async def _wait_for_manifests(self, kube, config):
        kube.wait_for_registered()
        config.timeout = "5m"

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_no_resource_limits.yaml"])
    async def test_get_resource_requirements_no_limits(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        deployment = await Deployment.read('fiber-http', tuning_config.namespace)
        await deployment.wait_until_ready()

        pods = await deployment.get_pods()
        assert len(pods) == 1, "expected a fiber-http pod"
        pod = pods[0]
        container = pod.get_container('fiber-http')
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        }

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_no_resource_limits.yaml"])
    async def test_set_resource_requirements_no_limits(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        deployment = await Deployment.read('fiber-http', tuning_config.namespace)
        await deployment.wait_until_ready()

        pods = await deployment.get_pods()
        assert len(pods) == 1, "expected a fiber-http pod"
        pod = pods[0]
        container = pod.get_container('fiber-http')
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        }

        # Set request and limit
        container.set_resource_requirements('cpu', {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '250m'
        })
        container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '250m'
        }

        # Set limit, leaving request alone
        container.set_resource_requirements('cpu', {
            servo.connectors.kubernetes.ResourceRequirement.limit: '750m'
        })
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '750m'
        }

        # Set request, clearing limit
        container.set_resource_requirements('cpu', {
            servo.connectors.kubernetes.ResourceRequirement.request: '250m',
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        })
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '250m',
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        }

        # Clear request and limit
        container.set_resource_requirements('cpu', {
            servo.connectors.kubernetes.ResourceRequirement.request: None,
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        })
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: None,
            servo.connectors.kubernetes.ResourceRequirement.limit: None
        }

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_no_resource_limits.yaml"])
    async def test_initialize_tuning_pod_set_defaults_for_no_limits(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        container_config = tuning_config.deployments[0].containers[0]
        container_config.cpu.limit = '1000m'
        container_config.memory.limit = '1GiB'

        # NOTE: Create the optimizations class to bring up the canary
        await servo.connectors.kubernetes.KubernetesOptimizations.create(tuning_config)

        # Read the Tuning Pod and check resources
        pod = await Pod.read('fiber-http-tuning', tuning_config.namespace)
        container = pod.get_container('fiber-http')
        cpu_requirements = container.get_resource_requirements('cpu')
        memory_requirements = container.get_resource_requirements('memory')

        assert cpu_requirements[servo.connectors.kubernetes.ResourceRequirement.limit] == '1'
        assert memory_requirements[servo.connectors.kubernetes.ResourceRequirement.limit] == '1073741824'

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_no_cpu_limit.yaml"])
    async def test_no_cpu_limit(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        tuning_config.deployments[0].containers[0].cpu.limit = '1000m'
        tuning_config.deployments[0].containers[0].cpu.set = ['request']

        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )

        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert setting
        assert setting.value == 250

        # Read the Tuning Pod and check resources
        pod = await Pod.read('fiber-http-tuning', tuning_config.namespace)
        container = pod.get_container('fiber-http')

        # CPU picks up the 1000m default and then gets adjust to 250m
        assert container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '250m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '1'
        }

        # Memory is untouched from the mainfest
        assert container.get_resource_requirements('memory') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '128Mi',
            servo.connectors.kubernetes.ResourceRequirement.limit: '128Mi'
        }

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_no_resource_limits.yaml"])
    async def test_reading_values_from_no_limits_optimization_class(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        # NOTE: Create the optimizations class to bring up the canary
        kubernetes_optimizations = await servo.connectors.kubernetes.KubernetesOptimizations.create(tuning_config)
        canary_optimization = kubernetes_optimizations.optimizations[0]

        # Validate Tuning
        assert canary_optimization.tuning_cpu, "Expected Tuning CPU"
        assert canary_optimization.tuning_cpu.value == 125
        assert canary_optimization.tuning_cpu.request == 125
        assert canary_optimization.tuning_cpu.limit is None
        assert canary_optimization.tuning_cpu.pinned is False

        assert canary_optimization.tuning_memory, "Expected Tuning Memory"
        assert canary_optimization.tuning_memory.value == 134217728
        assert canary_optimization.tuning_memory.value.human_readable() == '128.0Mi'
        assert canary_optimization.tuning_memory.request == 134217728
        assert canary_optimization.tuning_memory.limit is None
        assert canary_optimization.tuning_memory.pinned is False

        assert canary_optimization.tuning_replicas.value == 1
        assert canary_optimization.tuning_replicas.pinned is True

        # Validate Main
        assert canary_optimization.main_cpu, "Expected Main CPU"
        assert canary_optimization.main_cpu.value == 125
        assert canary_optimization.main_cpu.request == 125
        assert canary_optimization.main_cpu.limit is None
        assert canary_optimization.main_cpu.pinned is True

        assert canary_optimization.main_memory, "Expected Main Memory"
        assert canary_optimization.main_memory.value == 134217728
        assert canary_optimization.main_memory.value.human_readable() == '128.0Mi'
        assert canary_optimization.main_memory.request == 134217728
        assert canary_optimization.main_memory.limit is None
        assert canary_optimization.main_memory.pinned is True

        assert canary_optimization.main_replicas.value == 1
        assert canary_optimization.main_replicas.pinned is True

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_bursty_memory.yaml"])
    async def test_reading_values_from_bursty_memory_optimization_class(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to read limits instead of requests
        container_config = tuning_config.deployments[0].containers[0]
        container_config.cpu.get = ['limit']
        container_config.memory.get = ['limit']
        container_config.memory.max = '3.0GiB'  # Raise max so we validate

        # NOTE: Create the optimizations class to bring up the canary
        kubernetes_optimizations = await servo.connectors.kubernetes.KubernetesOptimizations.create(tuning_config)
        canary_optimization = kubernetes_optimizations.optimizations[0]

        # Validate Tuning
        assert canary_optimization.tuning_cpu, "Expected Tuning CPU"
        assert canary_optimization.tuning_cpu.value == 250
        assert canary_optimization.tuning_cpu.request == 125
        assert canary_optimization.tuning_cpu.limit == 250
        assert canary_optimization.tuning_cpu.pinned is False

        assert canary_optimization.tuning_memory, "Expected Tuning Memory"
        assert canary_optimization.tuning_memory.value == 2147483648
        assert canary_optimization.tuning_memory.value.human_readable() == '2.0Gi'
        assert canary_optimization.tuning_memory.request == 134217728
        assert canary_optimization.tuning_memory.limit == 2147483648
        assert canary_optimization.tuning_memory.pinned is False

        assert canary_optimization.tuning_replicas.value == 1
        assert canary_optimization.tuning_replicas.pinned is True

        # Validate Main
        assert canary_optimization.main_cpu, "Expected Main CPU"
        assert canary_optimization.main_cpu.value == 250
        assert canary_optimization.main_cpu.request == 125
        assert canary_optimization.main_cpu.limit == 250
        assert canary_optimization.main_cpu.pinned is True

        assert canary_optimization.main_memory, "Expected Main Memory"
        assert canary_optimization.main_memory.value == 2147483648
        assert canary_optimization.main_memory.value.human_readable() == '2.0Gi'
        assert canary_optimization.main_memory.request == 134217728
        assert canary_optimization.main_memory.limit == 2147483648
        assert canary_optimization.main_memory.pinned is True

        assert canary_optimization.main_replicas.value == 2
        assert canary_optimization.main_replicas.pinned is True

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_bursty_memory.yaml"])
    async def test_preflight_cycle(self, kube, tuning_config: KubernetesConfiguration) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        tuning_config.deployments[0].containers[0].cpu.get = ['limit']
        tuning_config.deployments[0].containers[0].memory.max = '2.0GiB'
        tuning_config.deployments[0].containers[0].memory.get = ['limit']

        connector = KubernetesConnector(config=tuning_config)

        # Describe to get our baseline
        baseline_description = await connector.describe()
        baseline_main_cpu_setting = baseline_description.get_setting('fiber-http/fiber-http.cpu')
        assert baseline_main_cpu_setting
        assert baseline_main_cpu_setting.value == 250

        baseline_main_memory_setting = baseline_description.get_setting('fiber-http/fiber-http.mem')
        assert baseline_main_memory_setting
        assert baseline_main_memory_setting.value.human_readable() == '2.0Gi'

        ## Tuning settings
        baseline_tuning_cpu_setting = baseline_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert baseline_tuning_cpu_setting
        assert baseline_tuning_cpu_setting.value == 250

        baseline_tuning_memory_setting = baseline_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert baseline_tuning_memory_setting
        assert baseline_tuning_memory_setting.value.human_readable() == '2.0Gi'

        ##
        # Adjust CPU and Memory
        cpu_adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".500",
        )
        memory_adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="memory",
            value="1.0",
        )

        adjusted_description = await connector.adjust([cpu_adjustment, memory_adjustment])
        assert adjusted_description is not None

        ## Main settings
        adjusted_main_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http.cpu')
        assert adjusted_main_cpu_setting
        assert adjusted_main_cpu_setting.value == 250

        adjusted_main_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http.mem')
        assert adjusted_main_mem_setting
        assert adjusted_main_mem_setting.value.human_readable() == "2.0Gi"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 500

        adjusted_tuning_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0Gi"

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        ## Main settings
        adjusted_main_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http.cpu')
        assert adjusted_main_cpu_setting
        assert adjusted_main_cpu_setting.value == 250

        adjusted_main_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http.mem')
        assert adjusted_main_mem_setting
        assert adjusted_main_mem_setting.value.human_readable() == "2.0Gi"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 500

        adjusted_tuning_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0Gi"

        ## Read the Main Pod and check resources
        main_deployment = await Deployment.read('fiber-http', tuning_config.namespace)
        main_pods = await main_deployment.get_pods()
        main_pod_container = main_pods[0].get_container('fiber-http')

        ## CPU is set to 500m on both requirements
        assert main_pod_container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '125m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '250m'
        }

        ## Read the Tuning Pod and check resources
        tuning_pod = await Pod.read('fiber-http-tuning', tuning_config.namespace)
        tuning_pod_container = tuning_pod.get_container('fiber-http')

        ## CPU is set to 500m on both requirements
        assert tuning_pod_container.get_resource_requirements('cpu') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '500m',
            servo.connectors.kubernetes.ResourceRequirement.limit: '500m'
        }

        ## Memory is set to 1Gi on both requirements
        assert tuning_pod_container.get_resource_requirements('memory') == {
            servo.connectors.kubernetes.ResourceRequirement.request: '1Gi',
            servo.connectors.kubernetes.ResourceRequirement.limit: '1Gi'
        }

        ##
        # Adjust back to baseline

        cpu_adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )
        memory_adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="memory",
            value="2.0",
        )

        adjusted_description = await connector.adjust([cpu_adjustment, memory_adjustment])
        assert adjusted_description is not None

        adjusted_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_cpu_setting
        assert adjusted_cpu_setting.value == 250

        adjusted_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_mem_setting
        assert adjusted_mem_setting.value.human_readable() == '2.0Gi'

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        adjusted_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_cpu_setting
        assert adjusted_cpu_setting.value == 250

        adjusted_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_mem_setting
        assert adjusted_mem_setting.value.human_readable() == '2.0Gi'


# TODO: test_inject_by_source_port_int, test_inject_by_source_port_name

##
# Sidecar injection tests

ENVOY_SIDECAR_IMAGE_TAG = 'opsani/envoy-proxy:servox-v0.9.0'

@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestSidecarInjection:
    @pytest.fixture(autouse=True)
    async def _wait_for_manifests(self, kube, config):
        kube.wait_for_registered()
        config.timeout = "5m"

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace


    @pytest.mark.applymanifests("../manifests/sidecar_injection",
                                files=["fiber-http_single_port.yaml"])
    @pytest.mark.parametrize(
        "port, service",
        [
            (None, 'fiber-http'),
            (80, 'fiber-http'),
            ('http', 'fiber-http'),
        ],
    )
    async def test_inject_single_port_deployment(self, namespace: str, service: str, port: Union[str, int]) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read('fiber-http', namespace)
        assert len(deployment.containers) == 1, "expected a single container"
        service = await servo.connectors.kubernetes.Service.read('fiber-http', namespace)
        assert len(service.ports) == 1
        port_obj = service.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 8480

        await deployment.inject_sidecar(
            'opsani-envoy', ENVOY_SIDECAR_IMAGE_TAG, service='fiber-http', port=port
        )

        # Examine new sidecar
        await deployment.refresh()
        assert len(deployment.containers) == 2, "expected an injected container"
        sidecar_container = deployment.containers[1]
        assert sidecar_container.name == 'opsani-envoy'

        # Check ports and env
        assert sidecar_container.ports == [
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9980,
                host_ip=None,
                host_port=None,
                name='opsani-proxy',
                protocol='TCP'
            ),
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9901,
                host_ip=None,
                host_port=None,
                name='opsani-metrics',
                protocol='TCP'
            )
        ]
        assert sidecar_container.obj.env == [
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXY_SERVICE_PORT',
                value='9980',
                value_from=None
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXIED_CONTAINER_PORT',
                value='8480',
                value_from=None
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXY_METRICS_PORT',
                value='9901',
                value_from=None
            ),
        ]

    @pytest.mark.applymanifests("../manifests/sidecar_injection",
                                files=["fiber-http_multiple_ports.yaml"])
    @pytest.mark.parametrize(
        "port, service, error",
        [
            (None, 'fiber-http', ValueError("Target Service 'fiber-http' exposes multiple ports -- target port must be specified")),
            (80, 'fiber-http', None),
            ('http', 'fiber-http', None),
        ],
    )
    async def test_inject_multiport_deployment(self, namespace: str, service: str, port: Union[str, int], error: Optional[Exception]) -> None:
        deployment = await servo.connectors.kubernetes.Deployment.read('fiber-http', namespace)
        assert len(deployment.containers) == 1, "expected a single container"
        service = await servo.connectors.kubernetes.Service.read('fiber-http', namespace)
        assert len(service.ports) == 2
        port_obj = service.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 8480

        try:
            await deployment.inject_sidecar(
                'opsani-envoy', ENVOY_SIDECAR_IMAGE_TAG, service='fiber-http', port=port
            )
        except Exception as e:
            assert repr(e) == repr(error)

        # Examine new sidecar (if success is expected)
        if error is None:
            await deployment.refresh()
            assert len(deployment.containers) == 2, "expected an injected container"
            sidecar_container = deployment.containers[1]
            assert sidecar_container.name == 'opsani-envoy'

            # Check ports and env
            assert sidecar_container.ports == [
                kubernetes_asyncio.client.V1ContainerPort(
                    container_port=9980,
                    host_ip=None,
                    host_port=None,
                    name='opsani-proxy',
                    protocol='TCP'
                ),
                kubernetes_asyncio.client.V1ContainerPort(
                    container_port=9901,
                    host_ip=None,
                    host_port=None,
                    name='opsani-metrics',
                    protocol='TCP'
                )
            ]
            assert sidecar_container.obj.env == [
                kubernetes_asyncio.client.V1EnvVar(
                    name='OPSANI_ENVOY_PROXY_SERVICE_PORT',
                    value='9980',
                    value_from=None
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name='OPSANI_ENVOY_PROXIED_CONTAINER_PORT',
                    value='8480',
                    value_from=None
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name='OPSANI_ENVOY_PROXY_METRICS_PORT',
                    value='9901',
                    value_from=None
                ),
            ]

    @pytest.mark.applymanifests("../manifests/sidecar_injection",
                                files=["fiber-http_multiple_ports_symbolic_targets.yaml"])
    @pytest.mark.parametrize(
        "port, service",
        [
            (None, 'fiber-http'),
            (80, 'fiber-http'),
            ('http', 'fiber-http'),
        ],
    )
    async def test_inject_symbolic_target_port(self, namespace: str, service: str, port: Union[str, int]) -> None:
        """test_inject_by_source_port_name_with_symbolic_target_port"""
        deployment = await servo.connectors.kubernetes.Deployment.read('fiber-http', namespace)
        assert len(deployment.containers) == 1, "expected a single container"
        service = await servo.connectors.kubernetes.Service.read('fiber-http', namespace)
        assert len(service.ports) == 1
        port_obj = service.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 'collector'

        await deployment.inject_sidecar(
            'opsani-envoy', ENVOY_SIDECAR_IMAGE_TAG, service='fiber-http', port=port
        )

        # Examine new sidecar
        await deployment.refresh()
        assert len(deployment.containers) == 2, "expected an injected container"
        sidecar_container = deployment.containers[1]
        assert sidecar_container.name == 'opsani-envoy'

        # Check ports and env
        assert sidecar_container.ports == [
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9980,
                host_ip=None,
                host_port=None,
                name='opsani-proxy',
                protocol='TCP'
            ),
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9901,
                host_ip=None,
                host_port=None,
                name='opsani-metrics',
                protocol='TCP'
            )
        ]
        assert sidecar_container.obj.env == [
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXY_SERVICE_PORT',
                value='9980',
                value_from=None
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXIED_CONTAINER_PORT',
                value='8480',
                value_from=None
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name='OPSANI_ENVOY_PROXY_METRICS_PORT',
                value='9901',
                value_from=None
            ),
        ]

@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesClusterConnectorIntegration:
    """Tests not requiring manifests setup, just an active cluster
    """

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @respx.mock
    async def test_telemetry_hello(self, namespace: str, config: KubernetesConfiguration, servo_runner: servo.runner.Runner) -> None:
        async with client.api_client.ApiClient() as api:
            v1 = kubernetes_asyncio.client.VersionApi(api)
            version_obj = await v1.get_code()

        expected = (
            f'"telemetry": {{"servox.version": "{servo.__version__}", "servox.platform": "{platform.platform()}", '
            f'"kubernetes.namespace": "{namespace}", "kubernetes.version": "{version_obj.major}.{version_obj.minor}", "kubernetes.platform": "{version_obj.platform}"}}'
        )

        connector = KubernetesConnector(config=config, telemetry=servo_runner.servo.telemetry)
        # attach connector
        await servo_runner.servo.add_connector("kubernetes", connector)

        request = respx.post(
            "https://api.opsani.com/accounts/servox.opsani.com/applications/tests/servo"
        ).mock(return_value=httpx.Response(200, text=f'{{"status": "{servo.api.OptimizerStatuses.ok}"}}'))

        await servo_runner._post_event(servo.api.Events.hello, dict(
            agent=servo.api.user_agent(),
            telemetry=servo_runner.servo.telemetry.values
        ))

        assert request.called
        print(request.calls.last.request.content.decode())
        assert expected in request.calls.last.request.content.decode()


##
# Tests against an ArgoCD rollout
@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config", "manage_rollout")
@pytest.mark.rollout_manifest.with_args("tests/manifests/argo_rollouts/fiber-http-opsani-dev.yaml")
class TestKubernetesConnectorRolloutIntegration:
    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.fixture()
    def _rollout_tuning_config(self, tuning_config: KubernetesConfiguration) -> KubernetesConfiguration:
        tuning_config.rollouts = [ RolloutConfiguration.parse_obj(d) for d in tuning_config.deployments ]
        tuning_config.deployments = []
        return tuning_config

    ##
    # Canary Tests
    async def test_create_rollout_tuning(self, _rollout_tuning_config: KubernetesConfiguration, namespace: str) -> None:
        connector = KubernetesConnector(config=_rollout_tuning_config)
        rol = await Rollout.read("fiber-http", namespace)
        await connector.describe()

        # verify tuning pod is registered as service endpoint
        service = await servo.connectors.kubernetes.Service.read("fiber-http", namespace)
        endpoints = await service.get_endpoints()
        tuning_name = f"{_rollout_tuning_config.rollouts[0].name}-tuning"
        tuning_endpoint = next(filter(
            lambda epa: epa.target_ref.name == tuning_name,
            endpoints[0].subsets[0].addresses
        ), None)
        if tuning_endpoint is None:
            raise AssertionError(f"Tuning pod {tuning_name} not contained in service endpoints: {endpoints}")



    async def test_adjust_rollout_tuning_cpu_with_settlement(self, _rollout_tuning_config, namespace):
        connector = KubernetesConnector(config=_rollout_tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )
        control = servo.Control(settlement='1s')
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert setting
        assert setting.value == 250

    async def test_adjust_rollout_tuning_insufficient_resources(self, _rollout_tuning_config, namespace) -> None:
        _rollout_tuning_config.timeout = "10s"
        _rollout_tuning_config.rollouts[0].containers[0].memory.max = "256Gi"
        connector = KubernetesConnector(config=_rollout_tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Gi", # impossible right?
        )
        with pytest.raises(AdjustmentRejectedError) as rejection_info:
            description = await connector.adjust([adjustment])

        rej_msg = str(rejection_info.value)
        assert "Insufficient memory." in rej_msg or "Pod Node didn't have enough resource: memory" in rej_msg

@pytest.mark.integration
@pytest.mark.clusterrolebinding('cluster-admin')
@pytest.mark.usefixtures("kubernetes_asyncio_config", "manage_rollout")
class TestRolloutSidecarInjection:
    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.mark.parametrize(
        "port, service",
        [
            (None, 'fiber-http'),
            (80, 'fiber-http'),
            ('http', 'fiber-http'),
        ],
    )
    @pytest.mark.rollout_manifest.with_args("tests/manifests/argo_rollouts/fiber-http_single_port.yaml")
    async def test_inject_single_port_rollout(self, namespace: str, service: str, port: Union[str, int]) -> None:
        rollout = await servo.connectors.kubernetes.Rollout.read('fiber-http', namespace)
        assert len(rollout.containers) == 1, "expected a single container"
        service = await servo.connectors.kubernetes.Service.read('fiber-http', namespace)
        assert len(service.ports) == 1
        port_obj = service.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 8480

        await rollout.inject_sidecar(
            'opsani-envoy', ENVOY_SIDECAR_IMAGE_TAG, service='fiber-http', port=port
        )

        # Examine new sidecar
        await rollout.refresh()
        assert len(rollout.containers) == 2, "expected an injected container"
        sidecar_container = rollout.containers[1]
        assert sidecar_container.name == 'opsani-envoy'

        # Check ports and env
        assert sidecar_container.ports == [
            servo.connectors.kubernetes.RolloutV1ContainerPort(
                container_port=9980,
                host_ip=None,
                host_port=None,
                name='opsani-proxy',
                protocol='TCP'
            ),
            servo.connectors.kubernetes.RolloutV1ContainerPort(
                container_port=9901,
                host_ip=None,
                host_port=None,
                name='opsani-metrics',
                protocol='TCP'
            )
        ]
        assert sidecar_container.obj.env == [
            servo.connectors.kubernetes.RolloutV1EnvVar(
                name='OPSANI_ENVOY_PROXY_SERVICE_PORT',
                value='9980',
                value_from=None
            ),
            servo.connectors.kubernetes.RolloutV1EnvVar(
                name='OPSANI_ENVOY_PROXIED_CONTAINER_PORT',
                value='8480',
                value_from=None
            ),
            servo.connectors.kubernetes.RolloutV1EnvVar(
                name='OPSANI_ENVOY_PROXY_METRICS_PORT',
                value='9901',
                value_from=None
            ),
        ]
