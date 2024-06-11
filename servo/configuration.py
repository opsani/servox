# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import base64
import enum
import inspect
import json
import os
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union
import typing
import httpx
import pydantic.json
import pydantic_core
from typing_extensions import Annotated, TypeAlias

import backoff
import pydantic
import pydantic_settings
import yaml

import servo.logging
import servo.types
from servo import types
from pydantic import Field, StringConstraints, ConfigDict

__all__ = [
    "AbstractBaseConfiguration",
    "AppdynamicsOptimizer",
    "BaseConfiguration",
    "BaseServoConfiguration",
    "OpsaniOptimizer",
    "OptimizerTypes",
    "CommonConfiguration",
]


ORGANIZATION_REGEX = r"([A-Za-z0-9-.]{5,50})"
# Organization regex constraint to enforce that:
# * Cannot contain a forward slash (/)
# * Cannot solely consist of a single period (.) or double periods (..)
# * Cannot match the regular expression: __.*__
# * Cannot start with dash (-)
# * Must be between at least 5 characters long and no longer than 50
# * Must match domain names but also allow non-domain names and names including no period (.)

NAME_REGEX = r"[a-zA-Z\_\-\.0-9]{1,64}"
OPTIMIZER_ID_REGEX = f"^{ORGANIZATION_REGEX}/{NAME_REGEX}$"


class SidecarConnectionFile(pydantic.BaseModel):
    Authorization: pydantic.SecretStr
    Endpoint: pydantic.AnyHttpUrl
    TenantId: str


class AppdynamicsOptimizer(pydantic_settings.BaseSettings):
    optimizer_id: str
    tenant_id: Optional[str] = None
    base_url: Optional[pydantic.AnyHttpUrl] = None
    # static config properties
    client_id: Optional[str] = None
    client_secret: Optional[pydantic.SecretStr] = None
    # dynamic config properties
    connection_file: Optional[str] = None
    token: Optional[pydantic.SecretStr] = None
    # override properties
    url: Optional[pydantic.AnyHttpUrl] = None
    token_url: Optional[pydantic.AnyHttpUrl] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.connection_file:
            # workaround to prevent race condition with sidecar. Only relevant on init
            init_backoff = backoff.on_exception(
                backoff.expo, FileNotFoundError, max_time=60
            )(self.load_connection_file)
            init_backoff()
        elif (
            self.client_id is None
            or self.client_secret is None
            or self.tenant_id is None
            or self.base_url is None
        ):
            raise ValueError(
                f"{self.__class__.__name__} must be configured with a connection file or specify base_url, client_id, client_secret, and tenant_id"
            )

        if not self.url:
            self.url = (
                f"{self.base_url}/rest/optimize/co/v1/optimizers/{self.optimizer_id}/"
            )
        if not self.token_url:
            self.token_url = (
                f"{self.base_url}/auth/{self.tenant_id}/default/oauth2/token"
            )

    def load_connection_file(self) -> None:
        """In place update of properties based on the current state of the configured connection file"""
        if not self.connection_file:
            raise ValueError("Unable to load connection file, no file specified")
        with open(self.connection_file) as connection_file_stream:
            content = yaml.safe_load(connection_file_stream)

        validated_content = SidecarConnectionFile.parse_obj(content)
        self.token = validated_content.Authorization
        self.base_url = validated_content.Endpoint.rstrip("/")
        self.tenant_id = validated_content.TenantId

    @pydantic.field_validator("base_url")
    def _rstrip_slash(cls, url: pydantic_core.Url) -> str:
        if url:
            return str(url).rstrip("/")
        return url

    @property
    def id(self) -> str:
        return f"{self.tenant_id} - {self.optimizer_id}"

    @property
    def name(self) -> str:
        return f"{self.optimizer_id}"

    # TODO[pydantic]: The following keys were removed: `fields`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = pydantic_settings.SettingsConfigDict(
        extra="forbid",
        validate_assignment=True,
        env_prefix="appd_",
    )


class OpsaniOptimizer(pydantic_settings.BaseSettings):
    """
    An Optimizer models an Opsani optimization engines that the Servo can connect to
    in order to access the Opsani machine learning technology for optimizing system infrastructure
    and application workloads.

    Attributes:
        id: A friendly identifier formed by joining the `organization` and the `name` with a slash character
            of the form `example.com/my-app` or `another.com/app-2`.
        token: An opaque access token for interacting with the Optimizer via HTTP Bearer Token authentication.
        base_url: The base URL for accessing the Opsani API. This field is typically only useful to Opsani developers or in the context
            of deployments with specific contractual, firewall, or security mandates that preclude access to the primary API.
        url: An optional URL that overrides the computed URL for accessing the Opsani API. This option is utilized during development
            and automated testing to bind the servo to a fixed URL.
    """

    id: Annotated[str, StringConstraints(pattern=OPTIMIZER_ID_REGEX)] = pydantic.Field(
        alias="opsani_optimizer"
    )
    token: pydantic.SecretStr
    base_url: pydantic.AnyHttpUrl = "https://api.opsani.com"
    url: Optional[pydantic.AnyHttpUrl] = None
    _organization: str
    _name: str

    def __init__(self, *args, **kwargs) -> None:
        # kwargs
        if not kwargs.get("token") and (
            token_file := os.environ.get("OPSANI_TOKEN_FILE")
        ):
            kwargs["token"] = pathlib.Path(token_file).read_text().strip()

        # alias will supress attempts to init with acutal property names. support this manually
        if (id_arg := kwargs.pop("id", None)) is not None:
            kwargs["opsani_optimizer"] = id_arg
        super().__init__(*args, **kwargs)

        organization, name = self.id.split("/")
        self._organization = organization
        self._name = name
        if not self.url:
            self.url = self.default_url

    @pydantic.field_validator("base_url")
    def _rstrip_slash(cls, url: pydantic_core.Url) -> str:
        return str(url).rstrip("/")

    @pydantic.field_serializer("token")
    def token_value(self, tok: pydantic.SecretStr, _) -> str:
        return tok.get_secret_value()

    @property
    def organization(self) -> str:
        """Returns the organization component of the optimizer ID.

        The domain name of the Organization tha the optimizer belongs to.

        For example, a domain name of `awesome.com` might belong to Awesome, Inc and all optimizers would be
        deployed under this domain name umbrella for easy access and autocompletion ergonomics.

        """
        return self._organization

    @property
    def name(self) -> str:
        """Returns the name component of the optimizer ID.

        The symbolic name of the application or service under optimization in a string of URL-safe characters
        between 1 and 64 characters in length.
        """
        return self._name

    @property
    def default_url(self) -> str:
        return f"{self.base_url}/accounts/{self.organization}/applications/{self.name}/"

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="OPSANI_", extra="forbid", validate_assignment=True
    )


OptimizerTypes: TypeAlias = Union[AppdynamicsOptimizer, OpsaniOptimizer]


DEFAULT_TITLE = "Base Connector Configuration Schema"


class AbstractBaseConfiguration(pydantic_settings.BaseSettings, servo.logging.Mixin):
    """
    AbstractBaseConfiguration is the root of the servo configuration class hierarchy.
    It does not define any concrete configuration model fields but provides a number
    of shared behaviors common and functionality common across all servo connectors.

    Typically connector configuration classes will inherit from the concrete subclass
    `BaseConfiguration` rather than `AbstractBaseConfiguration`. Direct subclasses of
    `AbstractBaseConfiguration` are utilized when you wish to make use of Pydantic's
    Custom Root Type support (see https://pydantic-docs.helpmanual.io/usage/models/#custom-root-types).
    Custom Roots require that no other model fields are declared on the model when the
    `__root__` field is defined. Custom roots effectively inline the target attribute
    from the model, unwrapping a layer of object containment from the config file and
    JSON Schema perspective. This is especially useful when the connector models a
    collection of independent elements such as webhooks or notifications.
    """

    @classmethod
    def parse_file(
        cls, file: pathlib.Path, *, key: Optional[str] = None
    ) -> List["AbstractBaseConfiguration"]:
        """
        Parse a YAML configuration file and return a list of configuration objects with the contents.

        If the file does not contain a valid configuration, a `ValidationError` will be raised.
        """
        configs = yaml.load_all(file.read_text(), Loader=yaml.FullLoader)
        config_objs = []

        for config in configs:
            if key:
                try:
                    config = config[key]
                except KeyError as error:
                    raise KeyError(f"invalid key '{key}'") from error
            config_objs.append(cls.parse_obj(config))

        return config_objs

    @classmethod
    def generate(cls, **kwargs) -> "AbstractBaseConfiguration":
        """
        Return a set of default settings for a new configuration.

        Implementations should build a complete, validated Pydantic model and return it.

        This is an abstract method that needs to be implemented in subclasses in order to support config generation.
        """
        return cls(**kwargs)

    # Automatically uppercase env names upon subclassing
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Schema title
        base_name = cls.__name__.replace("Configuration", "")
        if cls.model_config.get("title", None) == DEFAULT_TITLE:
            cls.model_config["title"] = f"{base_name} Connector Configuration Schema"

        # Default prefix
        # prefix = cls.model_config.get("env_prefix", None)
        # if prefix == "":
        #     prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).upper() + "_"

        # pydantic auto configures env vars so no need for this TODO remove when verified
        # for name, field in cls.__fields__.items():
        #     if (env_override := field.field_info.extra.get("env")) and not isinstance(
        #         env_override, list
        #     ):
        #         field.field_info.extra["env_names"] = {env_override}
        #     else:
        #         field.field_info.extra["env_names"] = {f"{prefix}{name}".upper()}

    def yaml(
        self,
        *,
        include: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        exclude: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **dumps_kwargs: Any,
    ) -> str:
        """
        Generate a YAML representation of the configuration.

        Arguments are passed through to the Pydantic `BaseModel.json` method.
        """
        # NOTE: We have to serialize through JSON first (not all fields serialize directly to YAML)
        config_json = self.model_dump_json(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            **dumps_kwargs,
        )
        return yaml.dump(json.loads(config_json), sort_keys=False)

    model_config = pydantic_settings.SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
        title=DEFAULT_TITLE,
        env_nested_delimiter="_",
    )


class BaseConfiguration(AbstractBaseConfiguration):
    """
    BaseConfiguration is the base configuration class for Opsani Servo Connectors.

    BaseConfiguration subclasses are typically paired 1:1 with a Connector class
    that inherits from `servo.connector.Connector` and implements the business logic
    of the connector. Configuration classes are connector specific and designed
    to be initialized from commandline arguments, environment variables, and defaults.
    Connectors are initialized with a valid settings instance capable of providing necessary
    configuration for the connector to function.

    An optional textual description of the configuration stanza useful for differentiating
    between configurations within assemblies.
    """

    description: Optional[str] = pydantic.Field(
        None, description="An optional description of the configuration."
    )


# Handled by pydantic by default TODO remove when verified
# Uppercase handling for non-subclassed settings models. Should be pushed into Pydantic as a PR
# env_names = BaseConfiguration.__fields__["description"].field_info.extra.get(
#     "env_names", set()
# )
# BaseConfiguration.__fields__["description"].field_info.extra["env_names"] = set(
#     map(str.upper, env_names)
# )


class BackoffSettings(AbstractBaseConfiguration):
    """
    BackoffSettings objects model configuration of backoff and retry policies.

    See https://github.com/litl/backoff
    """

    max_time: Optional[servo.types.Duration]
    """
    The maximum amount of time to retry before giving up.
    """

    @pydantic.field_serializer("max_time")
    def serialize_courses_in_order(max_time: servo.types.Duration):
        return str(max_time)

    max_tries: Optional[int]
    """
    The maximum number of retry attempts to make before giving up.
    """


class Timeouts(AbstractBaseConfiguration):
    """Timeouts models the configuration of timeouts for the HTTPX library, which provides HTTP networking capabilities to the
    servo.

    See https://www.python-httpx.org/advanced/#timeout-configuration
    """

    connect: Optional[servo.types.Duration]
    """Specifies the maximum amount of time to wait until a connection to the requested host is established. If HTTPX is unable
    to connect within this time frame, a ConnectTimeout exception is raised.
    """

    read: Optional[servo.types.Duration]
    """Specifies the maximum duration to wait for a chunk of data to be received (for example, a chunk of the response body).
    If HTTPX is unable to receive data within this time frame, a ReadTimeout exception is raised.
    """

    write: Optional[servo.types.Duration]
    """Specifies the maximum duration to wait for a chunk of data to be sent (for example, a chunk of the request body).
    If HTTPX is unable to send data within this time frame, a WriteTimeout exception is raised.
    """

    pool: Optional[servo.types.Duration]
    """Specifies the maximum duration to wait for acquiring a connection from the connection pool. If HTTPX is unable to
    acquire a connection within this time frame, a PoolTimeout exception is raised. A related configuration here is the maximum
    number of allowable connections in the connection pool, which is configured by the pool_limits.
    """

    def __init__(
        self,
        timeout: Optional[Union[str, int, float, servo.types.Duration]] = None,
        **kwargs,
    ) -> None:  # noqa: D107
        for attr in ("connect", "read", "write", "pool"):
            if not attr in kwargs:
                kwargs[attr] = timeout
        super().__init__(**kwargs)


ProxyKey = Annotated[str, StringConstraints(pattern=r"^(https?|all)://")]


class BackoffContexts(enum.StrEnum):
    """An enumeration that defines the default set of backoff contexts."""

    default = "__default__"
    connect = "connect"


class BackoffConfigurations(pydantic.RootModel):
    """A mapping of named backoff configurations."""

    root: Dict[str, BackoffSettings]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, context: str) -> BackoffSettings:
        return self.root[context]

    def get(self, context: str, default: Any = None) -> BackoffSettings:
        return self.root.get(context, default)

    def max_time(
        self, context: str = BackoffContexts.default
    ) -> Optional[servo.types.Duration]:
        """Return the maximum amount of time to wait before giving up."""
        return (
            self.get(context, None) or self.get(BackoffContexts.default)
        ).max_time.total_seconds()

    def max_tries(self, context: str = BackoffContexts.default) -> Optional[int]:
        """Return the maximum number of calls to attempt to the target before
        giving up."""
        return (self.get(context, None) or self.get(BackoffContexts.default)).max_tries


def ensure_valid_url_and_str(v: Any):
    if isinstance(v, str):
        return str(pydantic.AnyHttpUrl(v))

    return v


# Ensure URL is valid but keep url as string for downstream libraries
ValidProxy = typing.Annotated[
    Union[str, httpx.Proxy],
    pydantic.functional_validators.AfterValidator(ensure_valid_url_and_str),
]


class CommonConfiguration(AbstractBaseConfiguration):
    """CommonConfiguration models configuration for the Servo connector and establishes default
    settings for shared services such as networking and logging.
    """

    backoff: BackoffConfigurations = pydantic.Field(
        {
            BackoffContexts.default: {"max_time": "10m", "max_tries": None},
            BackoffContexts.connect: {"max_time": "1h", "max_tries": None},
        }
    )
    """A mapping of named operations to settings for the backoff library, which provides backoff
    and retry capabilities to the servo.

    See https://github.com/litl/backoff
    """

    proxies: Union[None, ProxyKey, Dict[ProxyKey, Optional[ValidProxy]]] = None
    """Proxy configuration for the HTTPX library, which provides HTTP networking capabilities to the
    servo.

    See https://www.python-httpx.org/advanced/#http-proxying
    """

    timeouts: Optional[Timeouts] = None
    """Timeout configuration for the HTTPX library, which provides HTTP networking capabilities to the
    servo.
    """

    ssl_verify: Union[None, bool, pydantic.FilePath] = None
    """SSL verification settings for the HTTPX library, which provides HTTP networking capabilities to the
    servo.

    Used to provide a certificate bundle for interacting with HTTPS web services with certificates that
    do not verify with the standard bundle (self-signed, private PKI, etc).

    Setting a value of `False` disables SSL verification and is strongly discouraged due to the significant
    security implications.

    See https://www.python-httpx.org/advanced/#ssl-certificates
    """

    @pydantic.field_validator("timeouts", mode="before")
    def parse_timeouts(cls, v):
        if isinstance(v, (str, int, float)):
            return Timeouts(v)
        return v

    @classmethod
    def generate(cls, **kwargs) -> Optional["CommonConfiguration"]:
        return None


class ChecksConfiguration(AbstractBaseConfiguration):
    """ChecksConfiguration models configuration for behavior of the checks flow, such as
    whether to automatically apply remedies.
    """

    connectors: Optional[list[str]] = pydantic.Field(
        None,
        description="Connectors to check",
    )
    name: Optional[list[str]] = pydantic.Field(
        None,
        description="Filter by name",
    )

    id: Optional[list[str]] = pydantic.Field(
        None,
        description="Filter by ID",
    )

    tag: Optional[list[str]] = pydantic.Field(
        None,
        description="Filter by tag",
    )

    quiet: bool = pydantic.Field(
        default=False, description="Do not echo generated output to stdout"
    )

    verbose: bool = pydantic.Field(default=False, description="Display verbose output")

    progressive: bool = pydantic.Field(
        default=True, description="Execute checks and emit output progressively"
    )

    wait: str = pydantic.Field(default="30m", description="Wait for checks to pass")

    delay: str = pydantic.Field(
        default="expo", description="Delay duration. Requires --wait"
    )

    halt_on: servo.types.ErrorSeverity = pydantic.Field(
        default=servo.types.ErrorSeverity.critical,
        description="Halt running on failure severity",
    )

    remedy: bool = pydantic.Field(
        default=True,
        description="Automatically apply remedies to failed checks if detected",
    )

    check_halting: bool = pydantic.Field(
        default=False, description="Halt to wait for each checks success"
    )

    @classmethod
    def generate(cls, **kwargs) -> Optional["ChecksConfiguration"]:
        return None


class BaseServoConfiguration(
    AbstractBaseConfiguration, abc.ABC, pydantic_settings.BaseSettings
):
    """
    Abstract base class for Servo instances.

    Note that the concrete BaseServoConfiguration class is built dynamically at runtime
    based on the available connectors and configuration in effect.

    See `Assembly` for details on how the concrete model is built.

    NOTE: Inherits from AbstractBaseConfiguration because of optimizer property
    """

    name: Optional[str] = None
    description: Optional[str] = None
    servo_uid: Union[str, None] = pydantic.Field(default=None, alias="SERVO_UID")
    optimizer: OptimizerTypes = {}
    connectors: list[str] | dict[str, str] | None = pydantic.Field(
        None,
        description=(
            "An optional, explicit configuration of the active connectors.\n"
            "\nConfigurable as either an array of connector identifiers (names or class) or\n"
            "a dictionary where the keys specify the key path to the connectors configuration\n"
            "and the values identify the connector (by name or class name)."
        ),
        examples=[
            ["kubernetes", "prometheus"],
            {"staging_prom": "prometheus", "gateway_prom": "prometheus"},
        ],
    )
    """
    An optional list of connector names or a mapping of connector names to connector class names
    """

    no_diagnostics: bool = pydantic.Field(
        default=True, description="Do not poll the Opsani API for diagnostics"
    )

    settings: Optional[CommonConfiguration] = pydantic.Field(
        default_factory=lambda: CommonConfiguration(),
        description="Configuration of the Servo connector",
    )
    """Configuration of the Servo itself.

    Servo settings are applied as defaults for other connectors whenever possible.
    """

    checks: Optional[ChecksConfiguration] = pydantic.Field(
        default_factory=lambda: ChecksConfiguration(),
        description="Configuration of Checks behavior",
    )

    @classmethod
    def generate(
        cls: Type["BaseServoConfiguration"], **kwargs
    ) -> Optional["BaseServoConfiguration"]:
        """
        Generates configuration for the servo assembly.
        """
        for name, field in cls.model_fields.items():
            if (
                name not in kwargs
                and inspect.isclass(field.annotation)
                and issubclass(field.annotation, AbstractBaseConfiguration)
            ):
                if inspect.isgeneratorfunction(field.annotation.generate):
                    for name, config in field.annotation.generate():
                        kwargs[name] = config
                else:
                    if config := field.annotation.generate():
                        kwargs[name] = config

        if "optimizer" not in kwargs:
            kwargs["optimizer"] = {
                "id": "generated-id.test/generated",
                "token": "generated-token",
            }

        return cls(**kwargs)

    @pydantic.field_validator("connectors", mode="before")
    @classmethod
    def validate_connectors(
        cls, connectors
    ) -> Optional[Union[Dict[str, str], List[str]]]:
        if isinstance(connectors, str):
            # NOTE: Special case. When we are invoked with a string it is typically an env var
            try:
                decoded_value = json.loads(connectors)
            except ValueError as e:
                raise ValueError(f'error parsing JSON for "{connectors}"') from e

            # Prevent infinite recursion
            if isinstance(decoded_value, str):
                raise ValueError(
                    f'JSON string values for `connectors` cannot parse into strings: "{connectors}"'
                )

            connectors = decoded_value

        # import late until dependencies are untangled
        from servo.connector import (
            _normalize_connectors,
            _routes_for_connectors_descriptor,
        )

        connectors = _normalize_connectors(connectors)
        # NOTE: Will raise if descriptor is invalid, failing validation
        _routes_for_connectors_descriptor(connectors)

        return connectors

    model_config = pydantic_settings.SettingsConfigDict(
        extra="forbid",
        validate_assignment=False,
        title="Abstract Servo Configuration Schema",
        env_prefix="servo_",
    )


class FastFailConfiguration(pydantic_settings.BaseSettings):
    """Configuration providing support for fast fail behavior which returns early
    from long running connector operations when SLO violations are observed"""

    disabled: Annotated[int, Field(ge=0, le=1, multiple_of=1)] = 0
    """Toggle fast-fail behavior on or off"""

    period: servo.types.Duration = "60s"
    """How often to check the SLO metrics"""

    span: servo.types.Duration = pydantic.Field(None, validate_default=True)
    """The span or window of time that SLO metrics are gathered for"""

    skip: servo.types.Duration = 0
    """How long to wait before querying SLO metrics for potential violations"""

    treat_zero_as_missing: bool = False
    """Whether or not to treat zero values as missing per certain metric systems"""
    model_config = ConfigDict(
        case_sensitive=True,
        extra="forbid",
        validate_assignment=True,
        env_prefix="SERVO_",
    )

    @pydantic.model_validator(mode="before")
    def span_defaults_to_period(cls, values):
        if values.get("span", None) is None:
            if "period" in values:
                values["span"] = values["period"]
            else:
                values["span"] = "60s"
        return values
