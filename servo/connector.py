import abc
import json
import re
from enum import Enum
from pathlib import Path
from typing import (
    Any, 
    ClassVar, 
    Generator, 
    Optional, 
    Set, 
    Type, 
    get_type_hints, 
    TypeVar, 
    Callable,
    Dict
)

import httpx
import semver
import typer
import yaml
import logging
import loguru
from pkg_resources import EntryPoint, iter_entry_points
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    HttpUrl,
    ValidationError,
    constr,
    root_validator,
    validator,
)
from pydantic.fields import ModelField


USER_AGENT = "github.com/opsani/servox"


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

    app_name: constr(regex=r"^[a-z\-]{3,64}$")
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

    class Config:
        env_prefix = "SERVO_OPTIMIZER_"
        env_file = ".env"
        case_sensitive = False  # TODO: Normalize the env vars
        extra = Extra.forbid
        fields = {
            "token": {
                "env": "SERVO_OPTIMIZER_TOKEN",
                "env_names": {"SERVO_OPTIMIZER_TOKEN"},
            }
        }


class ConnectorSettings(BaseSettings):
    """
    ConnectorSettings is the base configuration class for Opsani Servo Connectors.

    ConnectorSettings instances are typically paired 1:1 with a Connector class
    that inherits from `servo.connector.Connector` and provides the business logic
    of the connector. Settings classes are configuration specific specific and designed
    to be initialized from commandline arguments, environment variables, and defaults.
    Connectors are initialized with a valid settings instance capable of providing necessary
    configuration for the connector to function.
    """

    description: Optional[str]
    """An optional textual description of the configyuration stanza useful for differentiating
    between configurations within assemblies.
    """

    @classmethod
    def parse_file(cls, file: Path) -> 'ConnectorSettings':
        config = yaml.load(file.read_text(), Loader=yaml.FullLoader)
        return cls.parse_obj(config)

    @classmethod
    def generate(cls) -> 'ConnectorSettings':
        """
        Return a set of default settings for a new configuration.

        Implementations should build a complete, validated Pydantic model and return it.

        This is an abstract method that needs to be implemented in subclasses.
        """
        raise NotImplementedError(f"Generated settings must be implemented in the ConnectorSettings subclass '{cls.__qualname__}'")

    # Automatically uppercase env names upon subclassing
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Set default environment variable names
        # TODO: we can probably just use env_name
        for name, field in cls.__fields__.items():
            field.field_info.extra["env_names"] = {f"SERVO_{name}".upper()}
        

    class Config:
        env_prefix = "SERVO_"
        env_file = ".env"
        case_sensitive = True
        extra = Extra.forbid
        fields = {
            "description": {
                "env": "SERVO_DESCRIPTION",
                "env_names": {"SERVO_DESCRIPTION"},
            }
        }


EventFunctionType = TypeVar("EventFunctionType", bound=Callable[..., Any])

class EventResult(BaseModel):
    """
    Encapsulates the result of a dispatched Connector event
    """
    connector: 'Connector'
    event: str
    value: Any

from pydantic.main import ModelMetaclass

# NOTE: Boolean flag to know if we can safely reference Connector from the metaclass
_is_base_connector_class_defined = False

class ConnectorMetaclass(ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Decorate the class with an event registry, inheriting from our parent connectors
        events: Dict[str, 'EventDescriptor'] = {}

        for base in reversed(bases):
            if _is_base_connector_class_defined and issubclass(base, Connector) and base is not Connector:
                events.update(base.__events__)

        new_namespace = {
            '__events__': events,
            **{n: v for n, v in namespace.items()},
        }
        cls = super().__new__(mcs, name, bases, new_namespace, **kwargs)
        return cls

class Connector(BaseModel, abc.ABC, metaclass=ConnectorMetaclass):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """

    # Connector metadata
    name: ClassVar[str] = None
    """Name of the connector, by default derived from the class name.
    """

    version: ClassVar["Version"] = None
    """Semantic Versioning string of the connector.
    """

    description: ClassVar[Optional[str]] = None
    """Optional textual description of the connector.
    """

    homepage: ClassVar[Optional[HttpUrl]] = None
    """Link to the homepage of the connector.
    """

    license: ClassVar[Optional["License"]] = None
    """An enumerated value that identifies the license that the connector is distributed under.
    """

    maturity: ClassVar[Optional["Maturity"]] = None
    """An enumerated value that identifies the self-selected maturity level of the connector, provided for
    advisory purposes.
    """

    # Instance configuration

    settings: ConnectorSettings
    """Settings for the connector set explicitly or loaded from a config file.
    """

    optimizer: Optional[Optimizer]
    """Name of the command for interacting with the connector instance via the CLI.

    Note that optimizers are attached as configuration to Connector instance because
    the settings are not managed as part of the assembly config files and are always
    provided via environment variablesm, commandline arguments, or secrets management.
    """

    config_key_path: str
    """Key-path to the root of the connector's configuration.
    """

    command_name: constr(regex=r"^[a-z\-\_]{3,32}$")
    """Name of the command for interacting with the connector instance via the CLI.
    """

    @classmethod
    def all(cls) -> Set[Type["Connector"]]:
        """Return a set of all Connector subclasses"""
        return cls.__connectors__
    
    ##
    # Configuration

    @root_validator(pre=True)
    @classmethod
    def validate_metadata(cls, v):
        assert cls.name is not None, "name must be provided"
        assert cls.version is not None, "version must be provided"
        if isinstance(cls.version, str):
            # Attempt to parse
            cls.version = Version.parse(cls.version)
        assert isinstance(
            cls.version, (Version, semver.VersionInfo)
        ), "version is not a semantic versioning descriptor"
        return v

    @validator("config_key_path")
    @classmethod
    def validate_config_key_path(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/\\.]{3,128}$", v)
        ), "key paths may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
        return v

    @classmethod
    def settings_model(cls) -> Type["Settings"]:
        """Return the settings model backing the connector. 
        
        The effective type of the setting instance is defined by the type hint definitions of the 
        `settings_model` and `settings` level attributes closest in definition to the target class.
        """
        hints = get_type_hints(cls)
        settings_cls = hints["settings"]
        return settings_cls
    
    ##
    # Events

    @classmethod
    def responds_to_event(cls, event: str) -> bool:
        """
        Returns True if the Connector processes the specified event.
        """
        return bool(cls.__events__.get(event, False))

    def process_event(self, event: str, *args, **kwargs) -> Optional[EventResult]:
        """
        Process an event and return the result.
        Return None if the connector does not respond to the event.
        """
        if not self.responds_to_event(event):
            return None
        
        event_fn = getattr(self, event, None)
        if not callable(event_fn):
            raise ValueError("Encountered a non-callable handler for event '{event}'")
        
        value = event_fn(*args, **kwargs)
        return EventResult(
            connector=self,
            event=event,
            value=value
        )
    

    # subclass registry of connectors
    __connectors__: Set[Type["Connector"]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.__connectors__.add(cls)
        cls.__key_path__ = _key_path_for_connector_class(cls)
        
        cls.name = cls.__name__.replace("Connector", " Connector")
        cls.version = semver.VersionInfo.parse("0.0.0")                
        
        # Register events for all annotated methods (see `event` decorator)
        for key, value in cls.__dict__.items():                
            if v := getattr(value, '__connector_event__', None):
                if not isinstance(v, EventDescriptor):
                    raise TypeError(f"Unexpected event descriptor of type '{v.__class__}'")

                if cls.__events__.get(key, None):
                    raise ValueError(f"Duplicate event handler registered for event '{key}'")

                cls.__events__[key] = v
        
    def __init__(
        self,
        settings: ConnectorSettings,
        *,
        config_key_path: Optional[str] = None,
        command_name: Optional[str] = None,
        **kwargs,
    ):
        config_key_path = (
            config_key_path if config_key_path is not None else self.__class__.__key_path__
        )
        command_name = (
            command_name
            if command_name is not None
            else _command_name_from_config_key_path(config_key_path)
        )
        super().__init__(
            settings=settings,
            config_key_path=config_key_path,
            command_name=command_name,
            **kwargs,
        )

    ##
    # Subclass services    

    def api_client(self) -> httpx.Client:
        """Yields an httpx.AsyncClient instance configured to talk to Opsani API"""
        base_url = f"{self.optimizer.base_url}accounts/{self.optimizer.org_domain}/applications/{self.optimizer.app_name}/"
        headers = {
            "Authorization": f"Bearer {self.optimizer.token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
        return httpx.Client(base_url=base_url, headers=headers)

    @property
    def logger(self) -> logging.Logger:
        """Returns the logger"""
        return loguru.logger

    # TODO: Do we need this?
    @property
    def cli(self) -> Optional[typer.Typer]:
        """Returns a Typer CLI for the connector"""
        return None

_is_base_connector_class_defined = True
EventResult.update_forward_refs()

def _key_path_for_connector_class(cls: Type[Connector]) -> str:
    name = re.sub(r"Connector$", "", cls.__name__)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

def _command_name_from_config_key_path(key_path: str) -> str:
    # foo.bar.this_key => this-key
    return key_path.split('.', 1)[-1].replace('_', '-').lower()

class License(Enum):
    """Defined licenses"""

    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self):
        return self.value


class Maturity(Enum):
    """Connector maturity level"""

    EXPERIMENTAL = "Experimental"
    STABLE = "Stable"
    ROBUST = "Robust"

    @classmethod
    def from_str(cls, identifier: str) -> "Maturity":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No maturity level identified by "{identifier}".')

    def __str__(self):
        return self.value


class Version(semver.VersionInfo):
    pass


def metadata(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[semver.VersionInfo] = None,
    homepage: Optional[HttpUrl] = None,
    license: Optional[License] = None,
    maturity: Optional[Maturity] = None,
):
    """Decorate a Connector class with metadata"""

    def decorator(cls):
        if not issubclass(cls, Connector):
            raise TypeError("Metadata can only be attached to Connector subclasses")

        if name:
            cls.name = name
        if description:
            cls.description = description
        if version:
            cls.version = (
                version
                if isinstance(version, semver.VersionInfo)
                else Version.parse(version)
            )
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = license
        if maturity:
            cls.maturity = maturity
        return cls

    return decorator

class EventDescriptor(BaseModel):
    name: str
    kwargs: Dict[str, Any]
        
def event(**kwargs):
    """
    Registers an event on the Connector
    """
    def decorator(fn: EventFunctionType) -> EventFunctionType:
        # Annotate the function for processing later, see Connector.__init_subclass__
        fn.__connector_event__ = EventDescriptor(
            name=fn.__name__,
            kwargs=kwargs
        )
        return fn

    return decorator

#####

ENTRY_POINT_GROUP = "servo.connectors"


class ConnectorLoader:
    """
    Dynamically discovers and loads connectors via Python setuptools entry points
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self.group = group

    def iter_entry_points(self) -> Generator[EntryPoint, None, None]:
        yield from iter_entry_points(group=self.group, name=None)

    def load(self) -> Generator[Any, None, None]:
        for entry_point in self.iter_entry_points():
            yield entry_point.resolve()
