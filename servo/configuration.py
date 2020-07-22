import re
import json
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    HttpUrl,
    constr,
    root_validator,
    validator,
)
from servo.types import Duration


class Optimizer(BaseSettings):
    """
    An Optimizer models an Opsani optimization engines that the Servo can connect to
    in order to access the Opsani machine learning technology for optimizing system infrastructure
    and application workloads.
    """

    org_domain: constr(
        regex=r"(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))"
    )
    """
    The domain name of the Organization tha the optimizer belongs to.

    For example, a domain name of `awesome.com` might belong to Awesome, Inc and all optimizers would be
    deployed under this domain name umbrella for easy access and autocompletion ergonomics.
    """

    app_name: constr(regex=r"^[a-z\-\.0-9]{3,64}$")
    """
    The symbolic name of the application or servoce under optimization in a string of URL-safe characters between 3 and 64
    characters in length 
    """

    token: str
    """
    An opaque access token for interacting with the Optimizer via HTTP Bearer Token authentication.
    """

    base_url: HttpUrl = "https://api.opsani.com/"
    """
    The base URL for accessing the Opsani API. This optiion is typically only useful for Opsani developers or in the context
    of deployments with specific contractual, firewall, or security mandates that preclude access to the primary API.
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
            f"{self.base_url}accounts/{self.org_domain}/applications/{self.app_name}/"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = Extra.forbid
        fields = {
            "token": {"env": "OPSANI_TOKEN",},
            "base_url": {"env": "OPSANI_BASE_URL",},
        }


DEFAULT_TITLE = "Connector Configuration Schema"
DEFAULT_JSON_ENCODERS = {
    # Serialize Duration as Golang duration strings (treated as a timedelta otherwise)
    Duration: lambda d: f"{d}"
}


class BaseConfiguration(BaseSettings):
    """
    BaseConfiguration is the base configuration class for Opsani Servo Connectors.

    BaseConfiguration instances are typically paired 1:1 with a Connector class
    that inherits from `servo.connector.Connector` and provides the business logic
    of the connector. Configuration classes are connector specific and designed
    to be initialized from commandline arguments, environment variables, and defaults.
    Connectors are initialized with a valid settings instance capable of providing necessary
    configuration for the connector to function.
    """

    description: Optional[str] = Field(
        None, description="An optional annotation describing the configuration."
    )
    """An optional textual description of the configuration stanza useful for differentiating
    between configurations within assemblies.
    """

    @classmethod
    def parse_file(
        cls, file: Path, *, key: Optional[str] = None
    ) -> "BaseConfiguration":
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
    def generate(cls, **kwargs) -> "BaseConfiguration":
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
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
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
        return {**DEFAULT_JSON_ENCODERS, **encoders}

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = Extra.forbid
        title = DEFAULT_TITLE
        json_encoders = DEFAULT_JSON_ENCODERS


# Uppercase handling for non-subclassed settings models. Should be pushed into Pydantic as a PR
env_names = BaseConfiguration.__fields__["description"].field_info.extra.get(
    "env_names", set()
)
BaseConfiguration.__fields__["description"].field_info.extra["env_names"] = set(
    map(str.upper, env_names)
)
