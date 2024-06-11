from __future__ import annotations

from typing import Annotated, Type

import aiohttp
import httpx
import kubetest.client
from kubetest.objects import Deployment as KubetestDeployment
import kubernetes.client.models
import kubernetes.client.exceptions
import loguru
import platform
import pydantic
import pytest
import pytest_mock
import re
import respx
import traceback
from kubernetes_asyncio.client import (
    ApiClient,
    V1Container,
    V1ContainerPort,
    V1ResourceRequirements,
    V1EnvVar,
    V1ServicePort,
    VersionApi,
    VersionInfo,
)
from pydantic import BaseModel
from pydantic import ValidationError

import servo
import servo.connectors.kubernetes
from servo.connectors.kubernetes import (
    CPU,
    CanaryOptimization,
    CanaryOptimizationStrategyConfiguration,
    ContainerConfiguration,
    ContainerTagName,
    ContainerTagNameField,
    DefaultOptimizationStrategyConfiguration,
    DeploymentConfiguration,
    DNS_SUBDOMAIN_NAME_REGEX,
    DNSLabelName,
    DNSLabelNameField,
    DNSSubdomainName,
    FailureMode,
    KubernetesChecks,
    KubernetesConfiguration,
    KubernetesConnector,
    Memory,
    Core,
    OptimizationStrategy,
)
from servo.types.kubernetes import Resource, ResourceRequirement
from servo.connectors.kubernetes_helpers import (
    find_container,
    get_containers,
    ContainerHelper,
    DeploymentHelper,
    PodHelper,
    ServiceHelper,
)
from servo.errors import AdjustmentFailedError, AdjustmentRejectedError
import servo.runner
from servo.types.api import Adjustment, Component, Description
from servo.types.settings import Replicas, EnvironmentEnumSetting
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
            "ctx": {"min_length": 1},
            "input": "",
            "loc": ("name",),
            "msg": "String should have at least 1 character",
            "type": "string_too_short",
            "url": "https://errors.pydantic.dev/2.7/v/string_too_short",
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
            "ctx": {"max_length": 253},
            "input": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
            "loc": ("name",),
            "msg": "String should have at most 253 characters",
            "type": "string_too_long",
            "url": "https://errors.pydantic.dev/2.7/v/string_too_long",
        } in e.value.errors()

    def test_can_only_contain_alphanumerics_hyphens_and_dots(self, model) -> None:
        valid_name = "abcd1234.-sss"
        invalid_name = "abcd1234.-sss_$%!"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$"},
            "input": "abcd1234.-sss_$%!",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
        } in e.value.errors()

    def test_must_start_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "-abcd"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$"},
            "input": "-abcd",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
        } in e.value.errors()

    def test_must_end_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "abcd-"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$"},
            "input": "abcd-",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z\\\\.-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
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
            "ctx": {"min_length": 1},
            "input": "",
            "loc": ("name",),
            "msg": "String should have at least 1 character",
            "type": "string_too_short",
            "url": "https://errors.pydantic.dev/2.7/v/string_too_short",
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
            "ctx": {"max_length": 63},
            "input": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
            "loc": ("name",),
            "msg": "String should have at most 63 characters",
            "type": "string_too_long",
            "url": "https://errors.pydantic.dev/2.7/v/string_too_long",
        } in e.value.errors()

    def test_can_only_contain_alphanumerics_and_hyphens(self, model) -> None:
        valid_name = "abcd1234-sss"
        invalid_name = "abcd1234.-sss_$%!"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$"},
            "input": "abcd1234.-sss_$%!",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
        } in e.value.errors()

    def test_must_start_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "-abcd"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$"},
            "input": "-abcd",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
        } in e.value.errors()

    def test_must_end_with_alphanumeric_character(self, model) -> None:
        valid_name = "abcd"
        invalid_name = "abcd-"

        assert model(name=valid_name)
        with pytest.raises(ValidationError) as e:
            model(name=invalid_name)
        assert e
        assert {
            "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$"},
            "input": "abcd-",
            "loc": ("name",),
            "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z-])*[0-9A-Za-z]$'",
            "type": "string_pattern_mismatch",
            "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
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
            "ctx": {"max_length": 128},
            "input": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
            "loc": ("name",),
            "msg": "String should have at most 128 characters",
            "type": "string_too_long",
            "url": "https://errors.pydantic.dev/2.7/v/string_too_long",
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
            e = next(
                iter((v for v in e.value.errors() if v.get("loc", None) == ("name",))),
                None,
            )
            assert e
            assert e.pop("input", None) in ["-", "."]
            assert e == {
                "ctx": {"pattern": "^[0-9a-zA-Z]([0-9a-zA-Z_\\.\\-/:@])*$"},
                "loc": ("name",),
                "msg": "String should match pattern '^[0-9a-zA-Z]([0-9a-zA-Z_\\.\\-/:@])*$'",
                "type": "string_pattern_mismatch",
                "url": "https://errors.pydantic.dev/2.7/v/string_pattern_mismatch",
            }


class TestEnvironmentConfiguration:
    pass


class TestCommandConfiguration:
    pass


class TestKubernetesConfiguration:
    @pytest.fixture
    def funkytown(self, config: KubernetesConfiguration) -> KubernetesConfiguration:
        return config.model_computed_fields(
            update={"namespace": "funkytown"}, deep=True
        )

    def test_cascading_defaults(self, config: KubernetesConfiguration) -> None:
        # Verify that by default we get a null namespace
        assert DeploymentConfiguration.model_fields["namespace"].default is None
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
        model = config.model_copy(update={"namespace": "funkytown"}, deep=True)
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "default"

        model.cascade_common_settings(overwrite=True)
        assert model.namespace == "funkytown"
        assert model.deployments[0].namespace == "funkytown"

    def test_respects_explicit_override(self, config: KubernetesConfiguration) -> None:
        # set the property explictly to value equal to default, then trigger
        model = config.model_copy(update={"namespace": "funkytown"})
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

    def test_failure_mode_destroy(self) -> None:
        """test that the old 'destroy' setting is converted to 'shutdown'"""
        config = servo.connectors.kubernetes.KubernetesConfiguration(
            namespace="default",
            description="Update the namespace, deployment, etc. to match your Kubernetes cluster",
            on_failure=servo.connectors.kubernetes.FailureMode.destroy,
            deployments=[
                servo.connectors.kubernetes.DeploymentConfiguration(
                    name="fiber-http",
                    replicas=servo.Replicas(
                        min=1,
                        max=2,
                    ),
                    containers=[
                        servo.connectors.kubernetes.ContainerConfiguration(
                            name="fiber-http",
                            cpu=servo.connectors.kubernetes.CPU(
                                min="250m", max="4000m", step="125m"
                            ),
                            memory=servo.connectors.kubernetes.Memory(
                                min="128MiB", max="4.0GiB", step="128MiB"
                            ),
                        )
                    ],
                )
            ],
        )
        assert config.on_failure == FailureMode.shutdown
        assert config.deployments[0].on_failure == FailureMode.shutdown


class TestKubernetesConnector:
    pass


class TestContainerConfiguration:
    pass


class TestDeploymentConfiguration:
    def test_inheritance_of_default_namespace(self) -> None: ...

    def test_strategy_enum(self) -> None:
        config = DeploymentConfiguration(
            name="testing",
            containers=[],
            replicas=servo.Replicas(min=1, max=4),
            strategy=OptimizationStrategy.default.value,
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

    def test_strategy_object_default(self) -> None:
        config = DeploymentConfiguration(
            name="testing",
            containers=[],
            replicas=servo.Replicas(min=1, max=4),
            strategy=OptimizationStrategy.default.value,
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
                type=OptimizationStrategy.canary.value, alias="tuning"
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
        config = DeploymentConfiguration.model_validate(config_dict)
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
        config = DeploymentConfiguration.model_validate(config_dict)

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
        config = DeploymentConfiguration.model_validate(config_dict)
        assert isinstance(config.strategy, CanaryOptimizationStrategyConfiguration)
        assert config.strategy.type == OptimizationStrategy.canary
        assert config.strategy.alias == "tuning"


class TestCanaryOptimization:
    @pytest.mark.xfail
    def test_to_components_default_name(self, config) -> None:
        config.deployments[0].strategy = OptimizationStrategy.canary.value
        optimization = CanaryOptimization.model_construct(
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
            type=OptimizationStrategy.canary.value, alias="tuning"
        )
        config.deployments[0].containers[0].alias = "main"
        optimization = CanaryOptimization.model_construct(
            name="fiber-http-deployment/opsani/fiber-http:latest-canary",
            target_deployment_config=config.deployments[0],
            target_container_config=config.deployments[0].containers[0],
        )
        assert optimization.target_name == "main"
        assert optimization.tuning_name == "tuning"


def test_compare_strategy() -> None:
    config = CanaryOptimizationStrategyConfiguration(
        type=OptimizationStrategy.canary.value, alias="tuning"
    )
    assert config == OptimizationStrategy.canary


class TestResourceRequirement:
    @pytest.mark.parametrize(
        "requirement, val",
        [
            (ResourceRequirement.limit, "limits"),
            (ResourceRequirement.request, "requests"),
        ],
    )
    def test_resource_key(self, requirement: ResourceRequirement, val) -> None:
        assert requirement.resources_key == val


class TestContainer:
    @pytest.fixture
    def container(self) -> V1Container:
        container = V1Container(name="fiber-http")

        resources = V1ResourceRequirements()
        resources.requests = {"cpu": "100m", "memory": "3G"}
        resources.limits = {"cpu": "15000m"}
        container.resources = resources

        container.env = [
            V1EnvVar(name="TEST1", value="TEST2"),
        ]

        return container

    @pytest.mark.parametrize(
        "resource, requirement, value",
        [
            (
                "cpu",
                None,
                {
                    ResourceRequirement.request: "100m",
                    ResourceRequirement.limit: "15000m",
                },
            ),
            ("cpu", ResourceRequirement.request, "100m"),
            ("cpu", ResourceRequirement.limit, "15000m"),
            (
                "memory",
                None,
                {ResourceRequirement.request: "3G", ResourceRequirement.limit: None},
            ),
            ("memory", ResourceRequirement.request, "3G"),
            ("memory", ResourceRequirement.limit, None),
            (
                "invalid",
                None,
                {ResourceRequirement.request: None, ResourceRequirement.limit: None},
            ),
        ],
    )
    def test_get_resource_requirements(
        self,
        container: V1Container,
        resource: str,
        requirement: ResourceRequirement,
        value,
    ) -> None:
        assert (
            all_requirements := ContainerHelper.get_resource_requirements(
                container, resource
            )
        ) is not None
        if requirement:
            assert all_requirements.get(requirement) == value
        else:
            assert all_requirements == value

    @pytest.mark.parametrize(
        "resource, value, resources_dict",
        [
            (
                "cpu",
                {
                    ResourceRequirement.request: "100m",
                    ResourceRequirement.limit: "250m",
                },
                {
                    "limits": {"cpu": "250m"},
                    "requests": {"cpu": "100m", "memory": "3G"},
                },
            ),
            (
                "cpu",
                {ResourceRequirement.limit: "500m"},
                {
                    "limits": {"cpu": "500m"},
                    "requests": {"cpu": "100m", "memory": "3G"},
                },
            ),
        ],
    )
    def test_set_resource_requirements(
        self,
        container: V1Container,
        resource: str,
        value: dict[ResourceRequirement, Optional[str]],
        resources_dict,
    ) -> None:
        ContainerHelper.set_resource_requirements(container, resource, value)
        assert container.resources.to_dict() == resources_dict

    def test_set_resource_requirements_handles_null_requirements_dict(
        self, container: V1Container
    ):
        container.resources = V1ResourceRequirements()

        ContainerHelper.set_resource_requirements(
            container,
            Resource.cpu.value,
            {ResourceRequirement.request: "1000m", ResourceRequirement.limit: "1000m"},
        )
        assert container.resources.to_dict() == {
            "limits": {"cpu": "1000m"},
            "requests": {"cpu": "1000m"},
        }

    def test_get_environment_variable(self, container: V1Container):
        assert ContainerHelper.get_environment_variable(container, "TEST1") == "TEST2"

    def test_set_environment_variable(self, container: V1Container):
        ContainerHelper.set_environment_variable(container, "TEST1", "TEST3")
        ContainerHelper.set_environment_variable(container, "TEST4", "TEST5")

        assert container.env == [
            V1EnvVar(name="TEST1", value="TEST3"),
            V1EnvVar(name="TEST4", value="TEST5"),
        ]


class TestReplicas:
    @pytest.fixture
    def replicas(self) -> servo.Replicas:
        return servo.Replicas(min=1, max=4)

    def test_parsing(self, replicas: servo.Replicas) -> None:
        assert {
            "name": "replicas",
            "type": "range",
            "min": 1,
            "max": 4,
            "step": 1,
            "unit": None,
            "value": None,
            "pinned": False,
        } == replicas.model_dump()

    def test_to___opsani_repr__(self, replicas: servo.Replicas) -> None:
        replicas.value = 3
        assert replicas.__opsani_repr__() == {
            "replicas": {
                "max": 4,
                "min": 1,
                "step": 1,
                "value": 3,
                "type": "range",
                "pinned": False,
            }
        }


class TestCPU:
    @pytest.fixture
    def cpu(self) -> CPU:
        return CPU(min="125m", max="4000m", step="125m")

    def test_parsing(self, cpu: CPU) -> None:
        assert {
            "name": "cpu",
            "type": "range",
            "min": "125m",
            "max": 4,
            "step": "125m",
            "value": None,
            "unit": "cores",
            "pinned": False,
            "request": None,
            "limit": None,
            "get": [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
            "set": [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
        } == cpu.model_dump()

    def test_to___opsani_repr__(self, cpu: CPU) -> None:
        cpu.value = "3"
        assert cpu.__opsani_repr__() == {
            "cpu": {
                "max": 4.0,
                "min": 0.125,
                "step": 0.125,
                "value": 3.0,
                "unit": "cores",
                "type": "range",
                "pinned": False,
            }
        }

    def test_resolving_equivalent_units(self) -> None:
        cpu = CPU(min="125m", max=4.0, step=0.125)
        assert cpu.min == 0.125
        assert cpu.max == 4
        assert cpu.step.millicores == 125

    def test_resources_encode_to_json_human_readable(self, cpu) -> None:
        serialization = json.loads(cpu.model_dump_json())
        assert serialization["min"] == "125m"
        assert serialization["max"] == "4"
        assert serialization["step"] == "125m"

    def test_cpu_must_be_step_aligned(
        self, captured_logs: list["loguru.Message"]
    ) -> None:
        CPU(min="125m", max=4.0, step=0.250)
        assert (
            captured_logs[0].record["message"]
            == "CPU('cpu' 125m-4, 250m) min/max difference is not step aligned: 3.875 is not a multiple of 250m (consider min 250m or 0n, max 3.875 or 4.125)."
        )

    def test_min_can_be_less_than_step(self) -> None:
        CPU(min="125m", max=4.125, step=0.250)


class TestCore:
    @pytest.mark.parametrize(
        "input, cores",
        [
            ("100m", 0.1),
            ("1", 1),
            (1, 1),
            ("0.1", 0.1),
            ("0.1", 0.1),
            (2.0, 2),
            ("2.0", 2),
        ],
    )
    def test_parsing(
        self, input: Union[str, int, float], cores: Union[float, int]
    ) -> None:
        assert Core.parse(input) == cores

    @pytest.mark.parametrize(
        "input, output",
        [
            ("100m", "100m"),
            ("1", "1"),
            ("1.0", "1"),
            (1, "1"),
            (100, "100"),
            ("0.1", "100m"),
            ("0.1", "100m"),
            (2.5, "2.5"),
            ("2500m", "2.5"),
            ("123m", "123m"),
            ("100u", "100u"),
            ("0.0001", "100u"),
            ("100n", "100n"),
            ("0.0000001", "100n"),
        ],
    )
    def test_string_serialization(
        self, input: Union[str, int, float], output: str
    ) -> None:
        cores = Core.parse(input)
        assert str(cores) == output


class TestMemory:
    @pytest.fixture
    def memory(self) -> Memory:
        return Memory(min="0.25 GiB", max="4.0 GiB", step="128 MiB")

    def test_parsing(self, memory: Memory) -> None:
        assert {
            "name": "mem",
            "type": "range",
            "pinned": False,
            "value": None,
            "unit": "GiB",
            "min": 268435456,
            "max": 4294967296,
            "step": 134217728,
            "request": None,
            "limit": None,
            "get": [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
            "set": [
                ResourceRequirement.request,
                ResourceRequirement.limit,
            ],
        } == memory.model_dump()

    def test_to___opsani_repr__(self, memory: Memory) -> None:
        memory.value = "3.0 GiB"
        assert memory.__opsani_repr__() == {
            "mem": {
                "max": 4.0,
                "min": 0.25,
                "step": 0.125,
                "value": 3.0,
                "unit": "GiB",
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
                "unit": "GiB",
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
        serialization = json.loads(memory.model_dump_json())
        assert serialization["min"] == "256.0Mi"
        assert serialization["max"] == "4.0Gi"
        assert serialization["step"] == "128.0Mi"

    def test_mem_must_be_step_aligned(
        self, captured_logs: list["loguru.Message"]
    ) -> None:
        Memory(min="32 MiB", max=4.0, step="256MiB")
        assert (
            captured_logs[0].record["message"]
            == "Memory('mem' 32Mi-4Gi, 256Mi) min/max difference is not step aligned: 3.96875Gi is not a multiple of 256Mi (consider min 256Mi or 0B, max 3.78125Gi or 4.03125Gi)."
        )

    def test_min_can_be_less_than_step(self) -> None:
        Memory(min="32 MiB", max=4.03125, step="256MiB")


def test_millicpu():
    class Model(pydantic.BaseModel):
        cpu: Core

        @pydantic.field_validator("cpu", mode="before")
        def _parse_cpu(v: Any):
            if v is None:
                return v
            return Core.parse(v)

    assert Model(cpu=0.1).cpu.millicores == 100
    assert Model(cpu=0.5).cpu.millicores == 500
    assert Model(cpu=1).cpu.millicores == 1000
    assert Model(cpu="100m").cpu.millicores == 100
    assert "{0:m}".format(Model(cpu=1.5).cpu) == "1500m"
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
                        env=[
                            EnvironmentEnumSetting(
                                name="INIT_MEMORY_SIZE",
                                values=["32MB", "64MB", "128MB"],
                            )
                        ],
                    )
                ],
            )
        ],
    )


@pytest.mark.integration
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
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 0.125
        assert (
            description.get_setting("fiber-http/fiber-http.mem").human_readable_value
            == "128.0Mi"
        )
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
        setting = description.get_setting("fiber-http/fiber-http.cpu")
        assert setting
        assert setting.value == 0.15

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 0.15

    async def test_adjust_cpu_out_of_range(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".100",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.cpu")
        assert setting
        assert setting.value == 0.1

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 0.1

    async def test_adjust_env(self, config: KubernetesConfiguration) -> None:
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="INIT_MEMORY_SIZE",
            value="64MB",
        )

        control = servo.Control(settlement="1s")
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.INIT_MEMORY_SIZE")
        assert setting
        assert setting.value == "64MB"

    async def test_adjust_cpu_with_settlement(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".250",
        )
        control = servo.Control(settlement="1s")
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.cpu")
        assert setting
        assert setting.value == 0.25

    async def test_adjust_cpu_at_non_zero_container_index(
        self, config: KubernetesConfiguration
    ):
        # Inject a sidecar at index zero
        deployment = await DeploymentHelper.read("fiber-http", config.namespace)
        assert (
            deployment
        ), f"failed loading deployment 'fiber-http' in namespace '{config.namespace}'"
        await DeploymentHelper.inject_sidecar(
            deployment,
            "opsani-envoy",
            "opsani/envoy-proxy:latest",
            port="8480",
            service_port=8091,
            index=0,
        )
        await asyncio.wait_for(
            DeploymentHelper.wait_until_ready(deployment),
            timeout=config.timeout.total_seconds(),
        )

        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".250",
        )

        control = servo.Control(settlement="1s")
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.cpu")
        assert setting
        assert setting.value == 0.25

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 0.25

    async def test_adjust_cpu_matchlabels_dont_match_metadata_labels(
        self, config, kube: kubetest.client.TestClient
    ):
        deployments = kube.get_deployments()
        target_deploy = deployments.get("fiber-http")
        assert target_deploy is not None

        # Update metadata labels so they don't match the match_labels selector
        target_deploy.obj.metadata.labels["app.kubernetes.io/name"] = "web"
        target_deploy.api_client.patch_namespaced_deployment(
            target_deploy.name, target_deploy.namespace, target_deploy.obj
        )
        kube.wait_for_registered()

        config.timeout = "15s"
        config.cascade_common_settings(overwrite=True)
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="cpu",
            value=".150",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.cpu")
        assert setting
        assert setting.value == 0.15

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.cpu").value == 0.15

    async def test_adjust_memory(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="700Mi",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.mem")
        assert setting
        assert setting.value == 734003200

        # Get deployment and check the pods
        # deployment = await Deployment.read("web", "default")
        # debug(deployment)
        # debug(deployment.obj.spec.template.spec.containers)

    async def test_adjust_memory_out_of_range(self, config):
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="64Mi",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http.mem")
        assert setting
        assert setting.value == 67108864

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http.mem").value == 67108864

    async def test_adjust_deployment_insufficient_resources(
        self, config: KubernetesConfiguration
    ):
        config.timeout = "3s"
        config.cascade_common_settings(overwrite=True)
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
                re.escape(
                    "Requested adjustment(s) (fiber-http/fiber-http.mem=128Gi) cannot be scheduled due to "
                )
                + r"\"\d+/\d+ nodes are available:.* \d+ Insufficient memory.*\""
            ),
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
        config.cascade_common_settings(overwrite=True)
        connector = KubernetesConnector(config=config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="256Mi",
        )

        mocker.patch(
            "kubernetes_asyncio.client.models.v1_container.V1Container.image",
            new_callable=mocker.PropertyMock,
            return_value="opsani/bababooey:latest",
        )

        with pytest.raises(
            AdjustmentFailedError, match="Container image pull failure detected"
        ):
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
        setting = description.get_setting("fiber-http/fiber-http.replicas")
        assert setting
        assert setting.value == 2

    async def test_read_pod(
        self, config: KubernetesConfiguration, kube: kubetest.client.TestClient
    ) -> None:
        connector = KubernetesConnector(config=config)
        pods = kube.get_pods()
        pod_name = next(iter(pods.keys()))
        assert pod_name.startswith("fiber-http")
        pod = await PodHelper.read(pod_name, kube.namespace)
        assert pod

    ##
    # Canary Tests
    async def test_create_tuning(
        self, tuning_config: KubernetesConfiguration, kube: kubetest.client.TestClient
    ) -> None:
        # verify existing env vars are overriden by config var with same name
        main_dep = kube.get_deployments()["fiber-http"]
        main_dep.obj.spec.template.spec.containers[0].env = [
            kubernetes.client.models.V1EnvVar(name="FOO", value="BAZ")
        ]
        main_dep.api_client.patch_namespaced_deployment(
            main_dep.name, main_dep.namespace, main_dep.obj
        )
        tuning_config.deployments[0].containers[0].static_environment_variables = {
            "FOO": "BAR"
        }

        connector = KubernetesConnector(config=tuning_config)
        description = await connector.describe()

        assert description == Description(
            components=[
                Component(
                    name="fiber-http/fiber-http",
                    settings=[
                        CPU(
                            name="cpu",
                            type="range",
                            pinned=True,
                            value="125m",
                            min="125m",
                            max="875m",
                            step="125m",
                            request="125m",
                            limit="125m",
                            get=["request", "limit"],
                            set=["request", "limit"],
                        ),
                        Memory(
                            name="mem",
                            type="range",
                            pinned=True,
                            value=134217728,
                            min=134217728,
                            max=805306368,
                            step=33554432,
                            request=134217728,
                            limit=134217728,
                            get=["request", "limit"],
                            set=["request", "limit"],
                        ),
                        Replicas(
                            name="replicas",
                            type="range",
                            pinned=True,
                            value=1,
                            min=0,
                            max=99999,
                            step=1,
                        ),
                        EnvironmentEnumSetting(
                            name="INIT_MEMORY_SIZE",
                            type="enum",
                            pinned=True,
                            values=["32MB", "64MB", "128MB"],
                            value="32MB",
                        ),
                    ],
                ),
                Component(
                    name="fiber-http/fiber-http-tuning",
                    settings=[
                        CPU(
                            name="cpu",
                            type="range",
                            pinned=False,
                            value="125m",
                            min="125m",
                            max="875m",
                            step="125m",
                            request="125m",
                            limit="125m",
                            get=["request", "limit"],
                            set=["request", "limit"],
                        ),
                        Memory(
                            name="mem",
                            type="range",
                            pinned=False,
                            value=134217728,
                            min=134217728,
                            max=805306368,
                            step=33554432,
                            request=134217728,
                            limit=134217728,
                            get=["request", "limit"],
                            set=["request", "limit"],
                        ),
                        Replicas(
                            name="replicas",
                            type="range",
                            pinned=True,
                            value=1,
                            min=0,
                            max=1,
                            step=1,
                        ),
                        EnvironmentEnumSetting(
                            name="INIT_MEMORY_SIZE",
                            type="enum",
                            pinned=False,
                            values=["32MB", "64MB", "128MB"],
                            value="32MB",
                        ),
                    ],
                ),
            ]
        )

        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        assert (
            tuning_pod.obj.metadata.annotations["opsani.com/opsani_tuning_for"]
            == "fiber-http/fiber-http-tuning"
        )
        assert tuning_pod.obj.metadata.labels["opsani_role"] == "tuning"
        target_container = next(
            filter(lambda c: c.name == "fiber-http", tuning_pod.obj.spec.containers)
        )
        assert target_container.resources.requests == {"cpu": "125m", "memory": "128Mi"}
        assert target_container.resources.limits == {"cpu": "125m", "memory": "128Mi"}
        assert target_container.env == [
            kubernetes.client.models.V1EnvVar(name="INIT_MEMORY_SIZE", value="32MB"),
            kubernetes.client.models.V1EnvVar(name="FOO", value="BAR"),
        ]

    async def test_adjust_tuning_insufficient_mem(
        self, tuning_config: KubernetesConfiguration
    ) -> None:
        tuning_config.timeout = "10s"
        tuning_config.cascade_common_settings(overwrite=True)
        tuning_config.deployments[0].containers[0].memory = Memory(
            min="128MiB", max="128GiB", step="32MiB"
        )
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Gi",  # impossible right?
        )
        try:
            with pytest.raises(
                AdjustmentRejectedError,
                match=(
                    re.escape(
                        "Requested adjustment(s) (fiber-http/fiber-http-tuning.mem=128Gi) cannot be scheduled due to "
                    )
                    + r"\"\d+/\d+ nodes are available:.* \d+ Insufficient memory.*\""
                ),
            ) as rejection_info:
                await connector.adjust([adjustment])
        except AssertionError as ae:
            if "does not match '(reason ContainersNotReady)" in str(ae):
                pytest.xfail("Unschedulable condition took too long to show up")

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert rejection_info.value.reason == "unschedulable"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_tuning_insufficient_cpu_and_mem(
        self, tuning_config: KubernetesConfiguration
    ) -> None:
        tuning_config.timeout = "10s"
        tuning_config.cascade_common_settings(overwrite=True)
        tuning_config.deployments[0].containers[0].memory = Memory(
            min="128MiB", max="128GiB", step="32MiB"
        )
        tuning_config.deployments[0].containers[0].cpu = CPU(
            min="125m", max="200", step="125m"
        )
        connector = KubernetesConnector(config=tuning_config)

        adjustments = [
            Adjustment(
                component_name="fiber-http/fiber-http-tuning",
                setting_name="mem",
                value="128Gi",  # impossible right?
            ),
            Adjustment(
                component_name="fiber-http/fiber-http-tuning",
                setting_name="cpu",
                value="100",  # impossible right?
            ),
        ]
        with pytest.raises(
            AdjustmentRejectedError,
            match=(
                re.escape(
                    "Requested adjustment(s) (fiber-http/fiber-http-tuning.mem=128Gi, fiber-http/fiber-http-tuning.cpu=100) cannot be scheduled due to "
                )
                + r"\"\d+/\d+ nodes are available:.* \d+ Insufficient cpu.* \d+ Insufficient memory.*\""
            ),
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
        kube,
    ) -> None:
        tuning_config.timeout = "10s"
        tuning_config.cascade_common_settings(overwrite=True)
        connector = KubernetesConnector(config=tuning_config)

        mocker.patch(
            "kubernetes_asyncio.client.models.v1_container.V1Container.image",
            new_callable=mocker.PropertyMock,
            return_value="opsani/bababooey:latest",
        )

        # NOTE: describe logic currently invokes the same creation as adjust and allows for a faster test.
        # If tuning creation is removed from describe this test will need to be refactored and have a longer timeout and runtime
        try:
            await connector.describe()
        except AdjustmentFailedError as e:
            if "Container image pull failure detected" in str(e):
                pass
            elif "Unknown Pod status for 'fiber-http-tuning'" in str(e):
                # Catchall triggered
                pytest.xfail("Pod status update took too long")

    async def test_bad_request_error_handled_gracefully(
        self, tuning_config: KubernetesConfiguration, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Verify a failure to create a pod is not poorly handled in the handle_error destroy logic"""

        # Passing in an intentionally mangled memory setting to trigger an API error that prevents pod creation
        mocker.patch(
            "servo.connectors.kubernetes.Memory.__config__.validate_assignment",
            new_callable=mocker.PropertyMock(return_value=False),
        )
        mocker.patch(
            "servo.connectors.kubernetes._normalize_adjustment",
            return_value=("memory", "256.0MiBGiB"),
        )

        tuning_config.deployments[0].on_failure = FailureMode.shutdown
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="256Mi",
        )

        # Catch debug log messages
        messages = []
        connector.logger.add(lambda m: messages.append(m.record["message"]), level=10)

        with pytest.raises(servo.AdjustmentFailedError) as error:
            await connector.adjust([adjustment])

        # Check logs
        assert "no tuning pod exists, ignoring destroy" in messages[-30:]
        # Check error
        assert "quantities must match the regular expression" in str(error.value)
        top_cause = unwrap_exception_group(
            error.value.__cause__, kubernetes_asyncio.client.ApiException
        )
        if isinstance(top_cause, list):
            top_cause = top_cause[0]
        assert top_cause.status == 400

    async def test_adjust_tuning_cpu_with_settlement(
        self, tuning_config, namespace, kube
    ):
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )

        control = servo.Control(settlement="50ms")
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http-tuning.cpu")
        assert setting
        assert setting.value == 0.25

    async def test_adjust_tuning_env(self, tuning_config: KubernetesConfiguration):
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="INIT_MEMORY_SIZE",
            value="64MB",
        )

        control = servo.Control(settlement="50ms")
        description = await connector.adjust([adjustment], control)
        assert description is not None
        setting = description.get_setting(
            "fiber-http/fiber-http-tuning.INIT_MEMORY_SIZE"
        )
        assert setting
        assert setting.value == "64MB"

    async def test_adjust_handle_error_respects_nested_config(
        self, config: KubernetesConfiguration, kube: kubetest.client.TestClient
    ):
        config.timeout = "3s"
        config.on_failure = FailureMode.shutdown
        config.cascade_common_settings(overwrite=True)
        config.deployments[0].on_failure = FailureMode.exception
        config.deployments[0].containers[0].memory.max = "256Gi"
        connector = KubernetesConnector(config=config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http",
            setting_name="mem",
            value="128Gi",
        )
        with pytest.raises(
            AdjustmentRejectedError, match="Insufficient memory."
        ) as rejection_info:
            description = await connector.adjust([adjustment])
            debug(description)

        deployment = await DeploymentHelper.read("fiber-http", kube.namespace)
        # check deployment was not scaled to 0 replicas (i.e., the outer-level 'shutdown' was overridden)
        assert deployment.spec.replicas != 0

    async def test_adjust_tuning_cpu_out_of_range(self, tuning_config):
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".100",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http-tuning.cpu")
        assert setting
        assert setting.value == 0.1

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert description.get_setting("fiber-http/fiber-http-tuning.cpu").value == 0.1

    async def test_adjust_tuning_memory_out_of_range(self, tuning_config):
        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="64Mi",
        )
        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http-tuning.mem")
        assert setting
        assert setting.value == 67108864

        # Describe it again and make sure it matches
        description = await connector.describe()
        assert (
            description.get_setting("fiber-http/fiber-http-tuning.mem").value
            == 67108864
        )

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
    async def test_check_deployment_readiness_failure(
        self, config: KubernetesConfiguration, kube: kubetest.client.TestClient
    ):
        deployments = kube.get_deployments()
        target_deploy = deployments.get("fiber-http")
        assert target_deploy is not None

        target_container = next(
            filter(
                lambda c: c.name == "fiber-http",
                target_deploy.obj.spec.template.spec.containers,
            )
        )
        assert target_container is not None

        # Update to put deployment in unready state
        target_container.readiness_probe = kubernetes.client.models.V1Probe(
            _exec=kubernetes.client.models.V1ExecAction(command=["exit", "1"]),
            failure_threshold=1,
        )
        target_deploy.obj.spec.strategy.rolling_update.max_surge = "0%"
        target_deploy.api_client.patch_namespaced_deployment(
            target_deploy.name, target_deploy.namespace, target_deploy.obj
        )

        while target_deploy.is_ready():
            await asyncio.sleep(0.1)

        result = await KubernetesChecks(config).run_one(
            id="check_kubernetes_deployments_are_ready_item_0"
        )
        assert (
            result.success == False
            and result.message
            == 'caught exception (RuntimeError): Deployment "fiber-http" is not ready'
        )


##
# Rejection Tests using modified deployment, skips the standard manifest application
@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesConnectorIntegrationUnreadyCmd:
    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.fixture
    def config(self, config: KubernetesConfiguration) -> KubernetesConfiguration:
        config.container_logs_in_error_status = True
        return config

    @pytest.fixture
    def kubetest_deployment(
        self, kube: kubetest.client.TestClient, rootpath: pathlib.Path
    ) -> KubetestDeployment:
        deployment = kube.load_deployment(
            rootpath.joinpath("tests/manifests/fiber-http-opsani-dev.yaml")
        )
        deployment.obj.spec.template.spec.termination_grace_period_seconds = 10
        fiber_container = deployment.obj.spec.template.spec.containers[0]
        fiber_container.resources.requests["memory"] = "256Mi"
        fiber_container.resources.limits["memory"] = "256Mi"
        fiber_container.readiness_probe = kubernetes.client.models.V1Probe(
            failure_threshold=3,
            http_get=kubernetes.client.models.V1HTTPGetAction(
                path="/",
                port=9980,
                scheme="HTTP",
            ),
            initial_delay_seconds=1,
            period_seconds=5,
            success_threshold=1,
            timeout_seconds=1,
        )

        return deployment

    @pytest.fixture
    def kubetest_deployment_never_ready(
        self, kubetest_deployment: KubetestDeployment
    ) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = ["/bin/sh"]
        # Simulate a deployment which fails to start when memory adjusted to < 192Mi
        fiber_container.args = [
            "-c",
            "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; then /bin/fiber-http; else sleep 1d; fi",
        ]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment

    @pytest.fixture
    def kubetest_deployemnt_oom_killed(
        self, kubetest_deployment: KubetestDeployment
    ) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = ["/bin/sh"]
        # Simulate a deployment which will be OOMKilled when memory adjusted to < 192Mi
        fiber_container.args = [
            "-c",
            (
                "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; "
                "then /bin/fiber-http; "
                "else tail /dev/zero; "
                "fi"
            ),
        ]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment

    @pytest.fixture
    def kubetest_deployment_becomes_unready(
        self, kubetest_deployment: KubetestDeployment
    ) -> KubetestDeployment:
        fiber_container = kubetest_deployment.obj.spec.template.spec.containers[0]
        fiber_container.command = ["/bin/sh"]
        # Simulate a deployment which passes initial readiness checks when memory adjusted to < 192Mi then fails them a short time later
        fiber_container.args = [
            "-c",
            (
                "if [ $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) -gt 201326592 ]; "
                "then /bin/fiber-http; "
                "else (/bin/fiber-http &); sleep 10s; "
                "fi"
            ),
        ]

        kubetest_deployment.create()
        kubetest_deployment.wait_until_ready(timeout=30)
        return kubetest_deployment

    async def test_adjust_deployment_never_ready(
        self,
        config: KubernetesConfiguration,
        kubetest_deployment_never_ready: KubetestDeployment,
    ) -> None:
        config.timeout = "5s"
        config.cascade_common_settings(overwrite=True)
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
                "(reason ContainersNotReady) containers with unready status: [fiber-http"
                in str(rejection_info.value)
            )
            assert rejection_info.value.reason == "start-failed"
        except AssertionError as e:
            raise e from rejection_info.value

    async def test_adjust_deployment_oom_killed(
        self,
        config: KubernetesConfiguration,
        kubetest_deployemnt_oom_killed: KubetestDeployment,
    ) -> None:
        config.timeout = "10s"
        config.cascade_common_settings(overwrite=True)
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
                "Deployment fiber-http pod(s) crash restart detected: fiber-http-"
                in str(rejection_info.value)
            )
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            if "Found 1 unready pod(s) for deployment fiber-http" in str(
                rejection_info.value
            ):
                pytest.xfail("Restart count update took too long")
            raise e from rejection_info.value

    async def test_adjust_deployment_settlement_failed(
        self,
        config: KubernetesConfiguration,
        kubetest_deployment_becomes_unready: KubetestDeployment,
    ) -> None:
        config.timeout = "15s"
        config.settlement = "20s"
        config.on_failure = FailureMode.shutdown
        config.cascade_common_settings(overwrite=True)
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
            assert "(reason ContainersNotReady) containers with unready status: [fiber-http]" in str(
                rejection_info.value
            ) or "Deployment fiber-http pod(s) crash restart detected" in str(
                rejection_info.value
            ), str(
                rejection_info.value
            )
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate deployment scaled down to 0 instances
        kubetest_deployment_becomes_unready.refresh()
        assert kubetest_deployment_becomes_unready.obj.spec.replicas == 0

    async def test_adjust_tuning_never_ready(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployment_never_ready: KubetestDeployment,
        kube: kubetest.client.TestClient,
    ) -> None:
        tuning_config.timeout = "30s"
        tuning_config.on_failure = FailureMode.shutdown
        tuning_config.cascade_common_settings(overwrite=True)
        connector = KubernetesConnector(config=tuning_config)

        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="mem",
            value="128Mi",
        )

        try:
            with pytest.raises(AdjustmentRejectedError) as rejection_info:
                await connector.adjust([adjustment])
        except RuntimeError as e:
            if (
                f"Time out after {tuning_config.timeout} waiting for tuning pod shutdown"
                in str(e)
            ):
                pytest.xfail("Tuning pod shutdown took over 30 seconds")
            else:
                raise

        # Validate the correct error was raised, re-raise if not for additional debugging context
        try:
            assert (
                "(reason ContainersNotReady) containers with unready status: [fiber-http"
                in str(rejection_info.value)
            )
            assert rejection_info.value.reason == "start-failed"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(
            filter(
                lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers
            )
        )
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"

    async def test_adjust_tuning_oom_killed(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployemnt_oom_killed: KubetestDeployment,
        kube: kubetest.client.TestClient,
    ) -> None:
        tuning_config.timeout = "25s"
        tuning_config.on_failure = FailureMode.shutdown
        tuning_config.cascade_common_settings(overwrite=True)
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
            assert (
                "Tuning optimization fiber-http-tuning crash restart detected on container(s): fiber-http"
                in str(rejection_info.value)
            )
            assert rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(
            filter(
                lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers
            )
        )
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"

    async def test_adjust_tuning_settlement_failed(
        self,
        tuning_config: KubernetesConfiguration,
        kubetest_deployment_becomes_unready: KubetestDeployment,
        kube: kubetest.client.TestClient,
    ) -> None:
        tuning_config.timeout = "25s"
        tuning_config.settlement = "15s"
        tuning_config.on_failure = FailureMode.shutdown
        tuning_config.cascade_common_settings(overwrite=True)
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
            assert "(reason ContainersNotReady) containers with unready status: [fiber-http]" in str(
                rejection_info.value
            ) or "Tuning optimization fiber-http-tuning crash restart detected on container(s): fiber-http" in str(
                rejection_info.value
            )
            rejection_info.value.reason == "unstable"
        except AssertionError as e:
            raise e from rejection_info.value

        # Validate baseline was restored during handle_error
        tuning_pod = kube.get_pods()["fiber-http-tuning"]
        fiber_container = next(
            filter(
                lambda cont: cont.name == "fiber-http", tuning_pod.obj.spec.containers
            )
        )
        assert fiber_container.resources.requests["memory"] == "256Mi"
        assert fiber_container.resources.limits["memory"] == "256Mi"


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesResourceRequirementsIntegration:
    @pytest.fixture(autouse=True)
    async def _wait_for_manifests(self, kube, config):
        kube.wait_for_registered()
        config.timeout = "5m"

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements",
        files=["fiber-http_no_resource_limits.yaml"],
    )
    async def test_get_resource_requirements_no_limits(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        deployment = await DeploymentHelper.read("fiber-http", tuning_config.namespace)
        await DeploymentHelper.wait_until_ready(deployment)

        pods = await DeploymentHelper.get_latest_pods(deployment)
        assert len(pods) == 1, "expected a fiber-http pod"
        pod = pods[0]
        container = find_container(pod, "fiber-http")
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "125m",
            servo.connectors.kubernetes.ResourceRequirement.limit: None,
        }

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements",
        files=["fiber-http_no_resource_limits.yaml"],
    )
    async def test_set_resource_requirements_no_limits(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        deployment = await DeploymentHelper.read("fiber-http", tuning_config.namespace)
        await asyncio.wait_for(
            DeploymentHelper.wait_until_ready(deployment), timeout=300
        )

        pods = await DeploymentHelper.get_latest_pods(deployment)
        assert len(pods) == 1, "expected a fiber-http pod"
        pod = pods[0]
        container = find_container(pod, "fiber-http")
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "125m",
            servo.connectors.kubernetes.ResourceRequirement.limit: None,
        }

        # Set request and limit
        ContainerHelper.set_resource_requirements(
            container,
            "cpu",
            {
                servo.connectors.kubernetes.ResourceRequirement.request: "125m",
                servo.connectors.kubernetes.ResourceRequirement.limit: "250m",
            },
        )
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "125m",
            servo.connectors.kubernetes.ResourceRequirement.limit: "250m",
        }

        # Set limit, leaving request alone
        ContainerHelper.set_resource_requirements(
            container,
            "cpu",
            {servo.connectors.kubernetes.ResourceRequirement.limit: "750m"},
        )
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "125m",
            servo.connectors.kubernetes.ResourceRequirement.limit: "750m",
        }

        # Set request, clearing limit
        ContainerHelper.set_resource_requirements(
            container,
            "cpu",
            {
                servo.connectors.kubernetes.ResourceRequirement.request: "250m",
                servo.connectors.kubernetes.ResourceRequirement.limit: None,
            },
        )
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "250m",
            servo.connectors.kubernetes.ResourceRequirement.limit: None,
        }

        # Clear request and limit
        ContainerHelper.set_resource_requirements(
            container,
            "cpu",
            {
                servo.connectors.kubernetes.ResourceRequirement.request: None,
                servo.connectors.kubernetes.ResourceRequirement.limit: None,
            },
        )
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: None,
            servo.connectors.kubernetes.ResourceRequirement.limit: None,
        }

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements",
        files=["fiber-http_no_resource_limits.yaml"],
    )
    async def test_initialize_tuning_pod_set_defaults_for_no_limits(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        container_config = tuning_config.deployments[0].containers[0]
        container_config.cpu.limit = "1000m"
        container_config.memory.limit = "1GiB"

        # NOTE: Create the optimizations class to bring up the canary
        await servo.connectors.kubernetes.KubernetesOptimizations.create(tuning_config)

        # Read the Tuning Pod and check resources
        pod = await PodHelper.read("fiber-http-tuning", tuning_config.namespace)
        container = find_container(pod, "fiber-http")
        cpu_requirements = ContainerHelper.get_resource_requirements(container, "cpu")
        memory_requirements = ContainerHelper.get_resource_requirements(
            container, "memory"
        )

        assert (
            cpu_requirements[servo.connectors.kubernetes.ResourceRequirement.limit]
            == "1"
        )
        assert (
            memory_requirements[servo.connectors.kubernetes.ResourceRequirement.limit]
            == "1Gi"
        )

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements", files=["fiber-http_no_cpu_limit.yaml"]
    )
    async def test_no_cpu_limit(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        tuning_config.deployments[0].containers[0].cpu.limit = "1000m"
        tuning_config.deployments[0].containers[0].cpu.set = ["request"]

        connector = KubernetesConnector(config=tuning_config)
        adjustment = Adjustment(
            component_name="fiber-http/fiber-http-tuning",
            setting_name="cpu",
            value=".250",
        )

        description = await connector.adjust([adjustment])
        assert description is not None
        setting = description.get_setting("fiber-http/fiber-http-tuning.cpu")
        assert setting
        assert setting.value == 0.25

        # Read the Tuning Pod and check resources
        pod = await PodHelper.read("fiber-http-tuning", tuning_config.namespace)
        container = find_container(pod, "fiber-http")

        # CPU picks up the 1000m default and then gets adjust to 250m
        assert ContainerHelper.get_resource_requirements(container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "250m",
            servo.connectors.kubernetes.ResourceRequirement.limit: "1",
        }

        # Memory is untouched from the mainfest
        assert ContainerHelper.get_resource_requirements(container, "memory") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "128Mi",
            servo.connectors.kubernetes.ResourceRequirement.limit: "128Mi",
        }

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements",
        files=["fiber-http_no_resource_limits.yaml"],
    )
    async def test_reading_values_from_no_limits_optimization_class(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        # NOTE: Create the optimizations class to bring up the canary
        kubernetes_optimizations = (
            await servo.connectors.kubernetes.KubernetesOptimizations.create(
                tuning_config
            )
        )
        canary_optimization = kubernetes_optimizations.optimizations[0]

        # Validate Tuning
        assert canary_optimization.tuning_cpu, "Expected Tuning CPU"
        assert canary_optimization.tuning_cpu.value == 0.125
        assert canary_optimization.tuning_cpu.request == 0.125
        assert canary_optimization.tuning_cpu.limit is None
        assert canary_optimization.tuning_cpu.pinned is False

        assert canary_optimization.tuning_memory, "Expected Tuning Memory"
        assert canary_optimization.tuning_memory.value == 134217728
        assert canary_optimization.tuning_memory.value.human_readable() == "128.0Mi"
        assert canary_optimization.tuning_memory.request == 134217728
        assert canary_optimization.tuning_memory.limit is None
        assert canary_optimization.tuning_memory.pinned is False

        assert canary_optimization.tuning_replicas.value == 1
        assert canary_optimization.tuning_replicas.pinned is True

        # Validate Main
        assert canary_optimization.main_cpu, "Expected Main CPU"
        assert canary_optimization.main_cpu.value == 0.125
        assert canary_optimization.main_cpu.request == 0.125
        assert canary_optimization.main_cpu.limit is None
        assert canary_optimization.main_cpu.pinned is True

        assert canary_optimization.main_memory, "Expected Main Memory"
        assert canary_optimization.main_memory.value == 134217728
        assert canary_optimization.main_memory.value.human_readable() == "128.0Mi"
        assert canary_optimization.main_memory.request == 134217728
        assert canary_optimization.main_memory.limit is None
        assert canary_optimization.main_memory.pinned is True

        assert canary_optimization.main_replicas.value == 1
        assert canary_optimization.main_replicas.pinned is True

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements", files=["fiber-http_bursty_memory.yaml"]
    )
    async def test_reading_values_from_bursty_memory_optimization_class(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to read limits instead of requests
        container_config = tuning_config.deployments[0].containers[0]
        container_config.cpu.get = ["limit"]
        container_config.memory.get = ["limit"]
        container_config.memory.max = "3.0GiB"  # Raise max so we validate

        # NOTE: Create the optimizations class to bring up the canary
        kubernetes_optimizations = (
            await servo.connectors.kubernetes.KubernetesOptimizations.create(
                tuning_config
            )
        )
        canary_optimization = kubernetes_optimizations.optimizations[0]

        # Validate Tuning
        assert canary_optimization.tuning_cpu, "Expected Tuning CPU"
        assert canary_optimization.tuning_cpu.value == 0.25
        assert canary_optimization.tuning_cpu.request == 0.125
        assert canary_optimization.tuning_cpu.limit == 0.25
        assert canary_optimization.tuning_cpu.pinned is False

        assert canary_optimization.tuning_memory, "Expected Tuning Memory"
        assert canary_optimization.tuning_memory.value == 2147483648
        assert canary_optimization.tuning_memory.value.human_readable() == "2.0Gi"
        assert canary_optimization.tuning_memory.request == 134217728
        assert canary_optimization.tuning_memory.limit == 2147483648
        assert canary_optimization.tuning_memory.pinned is False

        assert canary_optimization.tuning_replicas.value == 1
        assert canary_optimization.tuning_replicas.pinned is True

        # Validate Main
        assert canary_optimization.main_cpu, "Expected Main CPU"
        assert canary_optimization.main_cpu.value == 0.25
        assert canary_optimization.main_cpu.request == 0.125
        assert canary_optimization.main_cpu.limit == 0.25
        assert canary_optimization.main_cpu.pinned is True

        assert canary_optimization.main_memory, "Expected Main Memory"
        assert canary_optimization.main_memory.value == 2147483648
        assert canary_optimization.main_memory.value.human_readable() == "2.0Gi"
        assert canary_optimization.main_memory.request == 134217728
        assert canary_optimization.main_memory.limit == 2147483648
        assert canary_optimization.main_memory.pinned is True

        assert canary_optimization.main_replicas.value == 2
        assert canary_optimization.main_replicas.pinned is True

    @pytest.mark.applymanifests(
        "../manifests/resource_requirements", files=["fiber-http_bursty_memory.yaml"]
    )
    async def test_preflight_cycle(
        self, kube, tuning_config: KubernetesConfiguration
    ) -> None:
        servo.logging.set_level("DEBUG")

        # Setup the config to set a default limit
        tuning_config.deployments[0].containers[0].cpu.get = ["limit"]
        tuning_config.deployments[0].containers[0].memory.max = "2.0GiB"
        tuning_config.deployments[0].containers[0].memory.get = ["limit"]

        connector = KubernetesConnector(config=tuning_config)

        # Describe to get our baseline
        baseline_description = await connector.describe()
        baseline_main_cpu_setting = baseline_description.get_setting(
            "fiber-http/fiber-http.cpu"
        )
        assert baseline_main_cpu_setting
        assert baseline_main_cpu_setting.value == 0.25

        baseline_main_memory_setting = baseline_description.get_setting(
            "fiber-http/fiber-http.mem"
        )
        assert baseline_main_memory_setting
        assert baseline_main_memory_setting.value.human_readable() == "2.0Gi"

        ## Tuning settings
        baseline_tuning_cpu_setting = baseline_description.get_setting(
            "fiber-http/fiber-http-tuning.cpu"
        )
        assert baseline_tuning_cpu_setting
        assert baseline_tuning_cpu_setting.value == 0.25

        baseline_tuning_memory_setting = baseline_description.get_setting(
            "fiber-http/fiber-http-tuning.mem"
        )
        assert baseline_tuning_memory_setting
        assert baseline_tuning_memory_setting.value.human_readable() == "2.0Gi"

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

        adjusted_description = await connector.adjust(
            [cpu_adjustment, memory_adjustment]
        )
        assert adjusted_description is not None

        ## Main settings
        adjusted_main_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http.cpu"
        )
        assert adjusted_main_cpu_setting
        assert adjusted_main_cpu_setting.value == 0.25

        adjusted_main_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http.mem"
        )
        assert adjusted_main_mem_setting
        assert adjusted_main_mem_setting.value.human_readable() == "2.0Gi"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.cpu"
        )
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 0.5

        adjusted_tuning_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.mem"
        )
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0Gi"

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        ## Main settings
        adjusted_main_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http.cpu"
        )
        assert adjusted_main_cpu_setting
        assert adjusted_main_cpu_setting.value == 0.25

        adjusted_main_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http.mem"
        )
        assert adjusted_main_mem_setting
        assert adjusted_main_mem_setting.value.human_readable() == "2.0Gi"

        ## Tuning settings
        adjusted_tuning_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.cpu"
        )
        assert adjusted_tuning_cpu_setting
        assert adjusted_tuning_cpu_setting.value == 0.5

        adjusted_tuning_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.mem"
        )
        assert adjusted_tuning_mem_setting
        assert adjusted_tuning_mem_setting.value.human_readable() == "1.0Gi"

        ## Read the Main Pod and check resources
        main_deployment = await DeploymentHelper.read(
            "fiber-http", tuning_config.namespace
        )
        main_pods = await DeploymentHelper.get_latest_pods(main_deployment)
        main_pod_container = find_container(main_pods[0], "fiber-http")

        ## CPU is set to 500m on both requirements
        assert ContainerHelper.get_resource_requirements(main_pod_container, "cpu") == {
            servo.connectors.kubernetes.ResourceRequirement.request: "125m",
            servo.connectors.kubernetes.ResourceRequirement.limit: "250m",
        }

        ## Read the Tuning Pod and check resources
        tuning_pod = await PodHelper.read("fiber-http-tuning", tuning_config.namespace)
        tuning_pod_container = find_container(tuning_pod, "fiber-http")

        ## CPU is set to 500m on both requirements
        assert ContainerHelper.get_resource_requirements(
            tuning_pod_container, "cpu"
        ) == {
            servo.connectors.kubernetes.ResourceRequirement.request: "500m",
            servo.connectors.kubernetes.ResourceRequirement.limit: "500m",
        }

        ## Memory is set to 1Gi on both requirements
        assert ContainerHelper.get_resource_requirements(
            tuning_pod_container, "memory"
        ) == {
            servo.connectors.kubernetes.ResourceRequirement.request: "1Gi",
            servo.connectors.kubernetes.ResourceRequirement.limit: "1Gi",
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

        adjusted_description = await connector.adjust(
            [cpu_adjustment, memory_adjustment]
        )
        assert adjusted_description is not None

        adjusted_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.cpu"
        )
        assert adjusted_cpu_setting
        assert adjusted_cpu_setting.value == 0.25

        adjusted_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.mem"
        )
        assert adjusted_mem_setting
        assert adjusted_mem_setting.value.human_readable() == "2.0Gi"

        ## Run another describe
        adjusted_description = await connector.describe()
        assert adjusted_description is not None

        adjusted_cpu_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.cpu"
        )
        assert adjusted_cpu_setting
        assert adjusted_cpu_setting.value == 0.25

        adjusted_mem_setting = adjusted_description.get_setting(
            "fiber-http/fiber-http-tuning.mem"
        )
        assert adjusted_mem_setting
        assert adjusted_mem_setting.value.human_readable() == "2.0Gi"


# TODO: test_inject_by_source_port_int, test_inject_by_source_port_name

##
# Sidecar injection tests

ENVOY_SIDECAR_IMAGE_TAG = "opsani/envoy-proxy:servox-v0.9.0"


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestSidecarInjection:
    @pytest.fixture(autouse=True)
    async def _wait_for_manifests(self, kube: kubetest.client.TestClient, config):
        kube.wait_for_registered()
        config.timeout = "5m"

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @pytest.mark.applymanifests(
        "../manifests/sidecar_injection", files=["fiber-http_single_port.yaml"]
    )
    @pytest.mark.parametrize(
        "port, service_name",
        [
            (None, "fiber-http"),
            (80, "fiber-http"),
            ("http", "fiber-http"),
        ],
    )
    async def test_inject_single_port_deployment(
        self, namespace: str, service_name: str, port: Union[str, int]
    ) -> None:
        deployment = await DeploymentHelper.read("fiber-http", namespace)
        assert len(get_containers(deployment)) == 1, "expected a single container"
        service = await ServiceHelper.read(service_name, namespace)
        assert len(service.spec.ports) == 1
        port_obj: V1ServicePort = service.spec.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 8480

        await DeploymentHelper.inject_sidecar(
            deployment,
            "opsani-envoy",
            ENVOY_SIDECAR_IMAGE_TAG,
            service=service_name,
            port=port,
        )

        # Examine new sidecar
        deployment = await DeploymentHelper.read("fiber-http", namespace)
        containers = get_containers(deployment)
        assert len(containers) == 2, "expected an injected container"
        sidecar_container = containers[1]
        assert sidecar_container.name == "opsani-envoy"

        # Check ports and env
        assert sidecar_container.ports == [
            V1ContainerPort(
                container_port=9980,
                host_ip=None,
                host_port=None,
                name="opsani-proxy",
                protocol="TCP",
            ),
            V1ContainerPort(
                container_port=9901,
                host_ip=None,
                host_port=None,
                name="opsani-metrics",
                protocol="TCP",
            ),
        ]
        assert sidecar_container.env == [
            V1EnvVar(
                name="OPSANI_ENVOY_PROXY_SERVICE_PORT", value="9980", value_from=None
            ),
            V1EnvVar(
                name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT",
                value="8480",
                value_from=None,
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name="OPSANI_ENVOY_PROXY_METRICS_PORT", value="9901", value_from=None
            ),
        ]

    @pytest.mark.applymanifests(
        "../manifests/sidecar_injection", files=["fiber-http_multiple_ports.yaml"]
    )
    @pytest.mark.parametrize(
        "port, service_name, error",
        [
            (
                None,
                "fiber-http",
                ValueError(
                    "Target Service 'fiber-http' exposes multiple ports -- target port must be specified"
                ),
            ),
            (80, "fiber-http", None),
            ("http", "fiber-http", None),
        ],
    )
    async def test_inject_multiport_deployment(
        self,
        namespace: str,
        service_name: str,
        port: Union[str, int],
        error: Optional[Exception],
    ) -> None:
        deployment = await DeploymentHelper.read("fiber-http", namespace)
        assert len(get_containers(deployment)) == 1, "expected a single container"
        service = await ServiceHelper.read(service_name, namespace)
        assert len(service.spec.ports) == 2
        port_obj: V1ServicePort = service.spec.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == 8480

        try:
            await DeploymentHelper.inject_sidecar(
                deployment,
                "opsani-envoy",
                ENVOY_SIDECAR_IMAGE_TAG,
                service=service_name,
                port=port,
            )
        except Exception as e:
            assert repr(e) == repr(error)

        # Examine new sidecar (if success is expected)
        if error is None:
            deployment = await DeploymentHelper.read("fiber-http", namespace)
            containers = get_containers(deployment)
            assert len(containers) == 2, "expected an injected container"
            sidecar_container = containers[1]
            assert sidecar_container.name == "opsani-envoy"

            # Check ports and env
            assert sidecar_container.ports == [
                kubernetes_asyncio.client.V1ContainerPort(
                    container_port=9980,
                    host_ip=None,
                    host_port=None,
                    name="opsani-proxy",
                    protocol="TCP",
                ),
                kubernetes_asyncio.client.V1ContainerPort(
                    container_port=9901,
                    host_ip=None,
                    host_port=None,
                    name="opsani-metrics",
                    protocol="TCP",
                ),
            ]
            assert sidecar_container.env == [
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXY_SERVICE_PORT",
                    value="9980",
                    value_from=None,
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT",
                    value="8480",
                    value_from=None,
                ),
                kubernetes_asyncio.client.V1EnvVar(
                    name="OPSANI_ENVOY_PROXY_METRICS_PORT",
                    value="9901",
                    value_from=None,
                ),
            ]

    @pytest.mark.applymanifests(
        "../manifests/sidecar_injection",
        files=["fiber-http_multiple_ports_symbolic_targets.yaml"],
    )
    @pytest.mark.parametrize(
        "port, service_name",
        [
            (None, "fiber-http"),
            (80, "fiber-http"),
            ("http", "fiber-http"),
        ],
    )
    async def test_inject_symbolic_target_port(
        self, namespace: str, service_name: str, port: Union[str, int]
    ) -> None:
        """test_inject_by_source_port_name_with_symbolic_target_port"""
        deployment = await DeploymentHelper.read("fiber-http", namespace)
        assert len(get_containers(deployment)) == 1, "expected a single container"
        service = await ServiceHelper.read(service_name, namespace)
        assert len(service.spec.ports) == 1
        port_obj: V1ServicePort = service.spec.ports[0]

        if isinstance(port, int):
            assert port_obj.port == port
        elif isinstance(port, str):
            assert port_obj.name == port
        assert port_obj.target_port == "collector"

        await DeploymentHelper.inject_sidecar(
            deployment,
            "opsani-envoy",
            ENVOY_SIDECAR_IMAGE_TAG,
            service=service_name,
            port=port,
        )

        # Examine new sidecar
        deployment = await DeploymentHelper.read("fiber-http", namespace)
        containers = get_containers(deployment)
        assert len(containers) == 2, "expected an injected container"
        sidecar_container = containers[1]
        assert sidecar_container.name == "opsani-envoy"

        # Check ports and env
        assert sidecar_container.ports == [
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9980,
                host_ip=None,
                host_port=None,
                name="opsani-proxy",
                protocol="TCP",
            ),
            kubernetes_asyncio.client.V1ContainerPort(
                container_port=9901,
                host_ip=None,
                host_port=None,
                name="opsani-metrics",
                protocol="TCP",
            ),
        ]
        assert sidecar_container.env == [
            kubernetes_asyncio.client.V1EnvVar(
                name="OPSANI_ENVOY_PROXY_SERVICE_PORT", value="9980", value_from=None
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name="OPSANI_ENVOY_PROXIED_CONTAINER_PORT",
                value="8480",
                value_from=None,
            ),
            kubernetes_asyncio.client.V1EnvVar(
                name="OPSANI_ENVOY_PROXY_METRICS_PORT", value="9901", value_from=None
            ),
        ]


@pytest.mark.integration
@pytest.mark.usefixtures("kubernetes_asyncio_config")
class TestKubernetesClusterConnectorIntegration:
    """Tests not requiring manifests setup, just an active cluster"""

    @pytest.fixture
    def namespace(self, kube: kubetest.client.TestClient) -> str:
        return kube.namespace

    @respx.mock
    async def test_telemetry_hello(
        self,
        namespace: str,
        config: KubernetesConfiguration,
        servo_runner: servo.runner.Runner,
    ) -> None:
        async with ApiClient() as api:
            v1 = VersionApi(api)
            version_obj: VersionInfo = await v1.get_code()

        expected = (
            f'"telemetry": {{"servox.version": "{servo.__version__}", "servox.platform": "{platform.platform()}", '
            f'"kubernetes.namespace": "{namespace}", "kubernetes.version": "{version_obj.major}.{version_obj.minor}", "kubernetes.platform": "{version_obj.platform}"}}'
        )

        connector = KubernetesConnector(
            config=config, telemetry=servo_runner.servo.telemetry
        )
        # attach connector
        await servo_runner.servo.add_connector("kubernetes", connector)

        request = respx.post(
            "https://api.opsani.com/accounts/generated-id.test/applications/generated/servo"
        ).mock(
            return_value=httpx.Response(
                200, text=f'{{"status": "{servo.api.OptimizerStatuses.ok}"}}'
            )
        )

        await servo_runner.servo.post_event(
            servo.api.Events.hello,
            dict(
                agent=servo.api.user_agent(),
                telemetry=servo_runner.servo.telemetry.values,
            ),
        )

        assert request.called
        print(request.calls.last.request.content.decode())
        assert expected in request.calls.last.request.content.decode()
