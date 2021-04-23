from __future__ import annotations

from typing import Type

import kubetest.client
import pydantic
import pytest
import re
from kubernetes_asyncio import client
from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError

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
    KubernetesChecks,
    KubernetesConfiguration,
    KubernetesConnector,
    Memory,
    Millicore,
    OptimizationStrategy,
    Pod,
    ResourceRequirement,
)
from servo.errors import AdjustmentRejectedError
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
            ("deployments[0].containers[0].memory.min", "256.0MiB"),
            ("deployments[0].containers[0].memory.max", "4.0GiB"),
            ("deployments[0].containers[0].memory.step", "128.0MiB"),
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
        assert serialization["min"] == "256.0MiB"
        assert serialization["max"] == "4.0GiB"
        assert serialization["step"] == "128.0MiB"

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
        assert description.get_setting("fiber-http/fiber-http.mem").human_readable_value == "128.0MiB"
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
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".250",
        )

        # Inject a sidecar at index zero
        deployment = await servo.connectors.kubernetes.Deployment.read('fiber-http', config.namespace)
        assert deployment, f"failed loading deployment 'fiber-http' in namespace '{config.namespace}'"
        await deployment.inject_sidecar('opsani-envoy', 'opsani/envoy-proxy:latest', port="8090", service_port=8091, index=0)

        control = servo.Control(settlement='1s')
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting('fiber-http/fiber-http.cpu')
        assert setting
        assert setting.value == 250

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 250

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
        with pytest.raises(AdjustmentRejectedError, match='Insufficient memory.'):
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

    async def test_adjust_tuning_insufficient_resources(
        self,
        tuning_config: KubernetesConfiguration,
        namespace,
        kube
    ) -> None:
        tuning_config.timeout = "3s"
        tuning_config.deployments[0].containers[0].memory.max = '128Gi'
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Gi", # impossible right?
        )
        with pytest.raises(AdjustmentRejectedError, match="Insufficient memory.") as rejection_info:
            await connector.adjust([adjustment])


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


##
# Rejection Tests using modified deployment
@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
@pytest.mark.applymanifests("../manifests", files=["fiber-http-unready-cmd.yaml"])
class TestKubernetesConnectorIntegrationUnreadyCmd:
    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    async def test_adjust_never_ready(self, config, kube: kubetest.client.TestClient) -> None:
        # new_dep = kube.load_deployment(abspath("../manifests/fiber-http-opsani-dev.yaml")) Why doesn't this work???? Had to use apply_manifests instead
        config.timeout = "3s"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Mi",
        )
        # NOTE: This can generate a 409 Conflict failure under CI
        with pytest.raises(AdjustmentRejectedError):
            for _ in range(3):
                try:
                    await connector.adjust([adjustment])

                except kubernetes_asyncio.client.exceptions.ApiException as e:
                    if e.status == 409 and e.reason == 'Conflict':
                        pass
                    else:
                        raise e

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
    async def test_reading_values_from_optimization_class_no_limits(self, kube, tuning_config: KubernetesConfiguration) -> None:
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
        assert canary_optimization.tuning_memory.value.human_readable() == '128.0MiB'
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
        assert canary_optimization.main_memory.value.human_readable() == '128.0MiB'
        assert canary_optimization.main_memory.request == 134217728
        assert canary_optimization.main_memory.limit is None
        assert canary_optimization.main_memory.pinned is True

        assert canary_optimization.main_replicas.value == 1
        assert canary_optimization.main_replicas.pinned is True

    @pytest.mark.applymanifests("../manifests/resource_requirements",
                                files=["fiber-http_bursty_memory.yaml"])
    async def test_reading_values_from_optimization_class_bursty_memory(self, kube, tuning_config: KubernetesConfiguration) -> None:
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
        assert canary_optimization.tuning_memory.value.human_readable() == '2.0GiB'
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
        assert canary_optimization.main_memory.value.human_readable() == '2.0GiB'
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
        assert baseline_main_memory_setting.value.human_readable() == '2.0GiB'

        ## Tuning settings
        baseline_tuning_cpu_setting = baseline_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert baseline_tuning_cpu_setting
        assert baseline_tuning_cpu_setting.value == 250

        baseline_tuning_memory_setting = baseline_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert baseline_tuning_memory_setting
        assert baseline_tuning_memory_setting.value.human_readable() == '2.0GiB'

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
        assert adjusted_main_mem_setting.value.human_readable() == "2.0GiB"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 500

        adjusted_tuning_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0GiB"

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        ## Main settings
        adjusted_main_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http.cpu')
        assert adjusted_main_cpu_setting
        assert adjusted_main_cpu_setting.value == 250

        adjusted_main_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http.mem')
        assert adjusted_main_mem_setting
        assert adjusted_main_mem_setting.value.human_readable() == "2.0GiB"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 500

        adjusted_tuning_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0GiB"

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
        assert adjusted_mem_setting.value.human_readable() == '2.0GiB'

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        adjusted_cpu_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.cpu')
        assert adjusted_cpu_setting
        assert adjusted_cpu_setting.value == 250

        adjusted_mem_setting = adjusted_description.get_setting('fiber-http/fiber-http-tuning.mem')
        assert adjusted_mem_setting
        assert adjusted_mem_setting.value.human_readable() == '2.0GiB'
