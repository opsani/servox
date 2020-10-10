from __future__ import annotations

import abc
import inspect
import json
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pydantic
import yaml

import servo.logging
import servo.types
from servo import types

__all__ = [
    "AbstractBaseConfiguration",
    "BaseConfiguration",
    "BaseAssemblyConfiguration",
    "Optimizer",
    "ServoConfiguration",
]


class Optimizer(pydantic.BaseSettings):
    """
    An Optimizer models an Opsani optimization engines that the Servo can connect to
    in order to access the Opsani machine learning technology for optimizing system infrastructure
    and application workloads.
    """

    org_domain: pydantic.constr(
        regex=r"(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))"
    )
    """
    The domain name of the Organization tha the optimizer belongs to.

    For example, a domain name of `awesome.com` might belong to Awesome, Inc and all optimizers would be
    deployed under this domain name umbrella for easy access and autocompletion ergonomics.
    """

    app_name: pydantic.constr(regex=r"^[a-z\-\.0-9]{3,64}$")
    """
    The symbolic name of the application or service under optimization in a string of URL-safe characters between 3 and 64
    characters in length.
    """

    token: str
    """
    An opaque access token for interacting with the Optimizer via HTTP Bearer Token authentication.
    """

    base_url: pydantic.AnyHttpUrl = "https://api.opsani.com/"
    """
    The base URL for accessing the Opsani API. This option is typically only useful to Opsani developers or in the context
    of deployments with specific contractual, firewall, or security mandates that preclude access to the primary API.
    """

    url: Optional[pydantic.AnyHttpUrl]
    """An optional URL that overrides the computed URL for accessing the Opsani API. This option is utilized during development
    and automated testing to bind the servo to a fixed URL.
    """

    def __init__(self, id: str = None, **kwargs):
        if isinstance(id, str):
            org_domain, app_name = id.split("/")
        else:
            org_domain = kwargs.pop("org_domain", None)
            app_name = kwargs.pop("app_name", None)
        super().__init__(org_domain=org_domain, app_name=app_name, **kwargs)

    @property
    def id(self) -> str:
        """
        Returns the primary identifier of the optimizer.

        A friendly identifier formed by joining the `org_domain` and the `app_name` with a slash character
        of the form `example.com/my-app` or `another.com/app-2`.
        """
        return f"{self.org_domain}/{self.app_name}"

    @property
    def api_url(self) -> str:
        """
        Returns a complete URL for interacting with the optimizer API.
        """
        return (
            self.url
            or f"{self.base_url}accounts/{self.org_domain}/applications/{self.app_name}/"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = pydantic.Extra.forbid
        fields = {
            "token": {
                "env": "OPSANI_TOKEN",
            },
            "base_url": {
                "env": "OPSANI_BASE_URL",
            },
        }


DEFAULT_TITLE = "Base Connector Configuration Schema"


class AbstractBaseConfiguration(pydantic.BaseSettings, servo.logging.Mixin):
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
    ) -> "AbstractBaseConfiguration":
        """
        Parse a YAML configuration file and return a configuration object with the contents.

        If the file does not contain a valid configuration, a `ValidationError` will be raised.
        """
        config = yaml.load(file.read_text(), Loader=yaml.FullLoader)
        if key:
            try:
                config = config[key]
            except KeyError as error:
                raise KeyError(f"invalid key '{key}'") from error
        return cls.parse_obj(config)

    @classmethod
    def generate(cls, **kwargs) -> "AbstractBaseConfiguration":
        """
        Return a set of default settings for a new configuration.

        Implementations should build a complete, validated Pydantic model and return it.

        This is an abstract method that needs to be implemented in subclasses in order to support config generation.
        """
        return cls()

    # Automatically uppercase env names upon subclassing
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Schema title
        base_name = cls.__name__.replace("Configuration", "")
        if cls.__config__.title == DEFAULT_TITLE:
            cls.__config__.title = f"{base_name} Connector Configuration Schema"

        # Default prefix
        prefix = cls.__config__.env_prefix
        if prefix == "":
            prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).upper() + "_"

        for name, field in cls.__fields__.items():
            field.field_info.extra["env_names"] = {f"{prefix}{name}".upper()}

    def yaml(
        self,
        *,
        include: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        exclude: Union[pydantic.AbstractSetIntStr, pydantic.MappingIntStrAny] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        encoder: Optional[Callable[[Any], Any]] = None,
        **dumps_kwargs: Any,
    ) -> str:
        """
        Generate a YAML representation of the configuration.

        Arguments are passed through to the Pydantic `BaseModel.json` method.
        """
        # NOTE: We have to serialize through JSON first (not all fields serialize directly to YAML)
        config_json = self.json(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            encoder=encoder,
            **dumps_kwargs,
        )
        return yaml.dump(json.loads(config_json))

    @staticmethod
    def json_encoders(
        encoders: Dict[Type[Any], Callable[..., Any]] = {}
    ) -> Dict[Type[Any], Callable[..., Any]]:
        """
        Returns a dict mapping servo types to callable JSON encoders for use in Pydantic Config classes
        when `json_encoders` need to be customized. Encoders provided in the encoders argument
        are merged into the returned dict and take precedence over the defaults.
        """
        from servo.types import DEFAULT_JSON_ENCODERS

        return {**DEFAULT_JSON_ENCODERS, **encoders}

    class Config(servo.types.BaseModelConfig):
        env_file = ".env"
        case_sensitive = True
        extra = pydantic.Extra.forbid
        title = DEFAULT_TITLE


class BaseConfiguration(AbstractBaseConfiguration):
    """
    BaseConfiguration is the base configuration class for Opsani Servo Connectors.

    BaseConfiguration subclasses are typically paired 1:1 with a Connector class
    that inherits from `servo.connector.Connector` and implements the business logic
    of the connector. Configuration classes are connector specific and designed
    to be initialized from commandline arguments, environment variables, and defaults.
    Connectors are initialized with a valid settings instance capable of providing necessary
    configuration for the connector to function.
    """

    description: Optional[str] = pydantic.Field(
        None, description="An optional annotation describing the configuration."
    )
    """An optional textual description of the configuration stanza useful for differentiating
    between configurations within assemblies.
    """


# Uppercase handling for non-subclassed settings models. Should be pushed into Pydantic as a PR
env_names = BaseConfiguration.__fields__["description"].field_info.extra.get(
    "env_names", set()
)
BaseConfiguration.__fields__["description"].field_info.extra["env_names"] = set(
    map(str.upper, env_names)
)


class BackoffSettings(BaseConfiguration):
    """
    BackoffSettings objects model configuration of backoff and retry policies.

    See https://github.com/litl/backoff
    """

    max_time: Optional[servo.types.Duration]
    """
    The maximum amount of time to retry before giving up.
    """

    max_tries: Optional[int]
    """
    The maximum number of retry attempts to make before giving up.
    """


class Timeouts(BaseConfiguration):
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
    ) -> None:
        for attr in ("connect", "read", "write", "pool"):
            if not attr in kwargs:
                kwargs[attr] = timeout
        super().__init__(**kwargs)


ProxyKey = pydantic.constr(regex=r"^(https?|all)://")


class ServoConfiguration(BaseConfiguration):
    """ServoConfiguration models configuration for the Servo connector and establishes default
    settings for shared services such as networking and logging.
    """

    backoff: Dict[str, BackoffSettings] = pydantic.Field(
        {
            "__default__": {"max_time": "10m", "max_tries": None},
            "connect": {"max_time": "1h", "max_tries": None},
        }
    )
    """A mapping of named operations to settings for the backoff library, which provides backoff
    and retry capabilities to the servo.

    See https://github.com/litl/backoff
    """

    proxies: Union[None, ProxyKey, Dict[ProxyKey, Optional[pydantic.AnyHttpUrl]]]
    """Proxy configuration for the HTTPX library, which provides HTTP networking capabilities to the
    servo.

    See https://www.python-httpx.org/advanced/#http-proxying
    """

    timeouts: Optional[Timeouts]
    """Timeout configuration for the HTTPX library, which provides HTTP networking capabilities to the
    servo.
    """

    ssl_verify: Union[None, bool, pydantic.FilePath]
    """SSL verification settings for the HTTPX library, which provides HTTP networking capabilities to the
    servo.

    Used to provide a certificate bundle for interacting with HTTPS web services with certificates that
    do not verify with the standard bundle (self-signed, private PKI, etc).

    Setting a value of `False` disables SSL verification and is strongly discouraged due to the significant
    security implications.

    See https://www.python-httpx.org/advanced/#ssl-certificates
    """

    @pydantic.validator("timeouts", pre=True)
    def parse_timeouts(cls, v):
        if isinstance(v, (str, int, float)):
            return Timeouts(v)
        return v

    @classmethod
    def generate(cls, **kwargs) -> Optional["ServoConfiguration"]:
        return None

    class Config(servo.types.BaseModelConfig):
        validate_assignment = True


class BaseAssemblyConfiguration(BaseConfiguration, abc.ABC):
    """
    Abstract base class for Servo assembly settings.

    Note that the concrete BaseAssemblyConfiguration class is built dynamically at runtime
    based on the avilable connectors and configuration in effect.

    See `Assembly` for details on how the concrete model is built.
    """

    connectors: Optional[Union[List[str], Dict[str, str]]] = pydantic.Field(
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

    servo: Optional[ServoConfiguration] = pydantic.Field(
        None, description="Configuration of the Servo connector"
    )
    """Configuration of the Servo itself.

    Servo settings are applied as defaults for other connectors whenever possible.
    """

    @classmethod
    def generate(
        cls: Type["BaseAssemblyConfiguration"], **kwargs
    ) -> Optional["BaseAssemblyConfiguration"]:
        """
        Generates configuration for the servo assembly.
        """
        for name, field in cls.__fields__.items():
            if (
                name not in kwargs
                and inspect.isclass(field.type_)
                and issubclass(field.type_, AbstractBaseConfiguration)
            ):
                if inspect.isgeneratorfunction(field.type_.generate):
                    for name, config in field.type_.generate():
                        kwargs[name] = config
                else:
                    if config := field.type_.generate():
                        kwargs[name] = config

        return cls(**kwargs)

    @pydantic.validator("connectors", pre=True)
    @classmethod
    def validate_connectors(
        cls, connectors
    ) -> Optional[Union[Dict[str, str], List[str]]]:
        if isinstance(connectors, str):
            # NOTE: Special case. When we are invoked with a string it is typically an env var
            try:
                decoded_value = BaseAssemblyConfiguration.__config__.json_loads(connectors)  # type: ignore
            except ValueError as e:
                raise ValueError(f'error parsing JSON for "{connectors}"') from e

            # Prevent infinite recursion
            if isinstance(decoded_value, str):
                raise ValueError(
                    f'JSON string values for `connectors` cannot parse into strings: "{connectors}"'
                )

            connectors = decoded_value

        # import late until dependencies are untangled
        from servo.connector import _normalize_connectors, _routes_for_connectors_descriptor

        connectors = _normalize_connectors(connectors)
        # NOTE: Will raise if descriptor is invalid, failing validation
        _routes_for_connectors_descriptor(connectors)

        return connectors

    class Config(types.BaseModelConfig):
        extra = pydantic.Extra.forbid
        title = "Abstract Servo Configuration Schema"
        env_prefix = "SERVO_"
