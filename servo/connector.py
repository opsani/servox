import abc
import json
import logging
import re
from inspect import Parameter, Signature
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Type,
    Union,
    get_type_hints,
)
from weakref import WeakKeyDictionary

import httpx
import loguru
import yaml
from pkg_resources import EntryPoint, iter_entry_points
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
from pydantic.main import ModelMetaclass

from servo.events import (
    CancelEventError,
    Event,
    EventCallable,
    EventError,
    EventHandler,
    EventResult,
    Preposition,
    get_event,
)
from servo.types import Duration, License, Maturity, Version
from servo.utilities import join_to_series

OPSANI_API_BASE_URL = "https://api.opsani.com/"
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

    base_url: HttpUrl = OPSANI_API_BASE_URL
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

# Global registries
_connector_subclasses: Set[Type["Connector"]] = set()
_connector_event_bus = WeakKeyDictionary()

# NOTE: Boolean flag to know if we can safely reference Connector from the metaclass
_is_base_connector_class_defined = False


class ConnectorMetaclass(ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Decorate the class with an event registry, inheriting from our parent connectors
        event_handlers: List[EventDescriptor] = []

        for base in reversed(bases):
            if (
                _is_base_connector_class_defined
                and issubclass(base, Connector)
                and base is not Connector
            ):
                event_handlers.extend(base.__event_handlers__)

        new_namespace = {
            "__event_handlers__": event_handlers,
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

    full_name: ClassVar[str] = None
    """The full name of the connector for referencing it unambiguously.
    """

    version: ClassVar[Version] = None
    """Semantic Versioning string of the connector.
    """

    description: ClassVar[Optional[str]] = None
    """Optional textual description of the connector.
    """

    homepage: ClassVar[Optional[HttpUrl]] = None
    """Link to the homepage of the connector.
    """

    license: ClassVar[Optional[License]] = None
    """An enumerated value that identifies the license that the connector is distributed under.
    """

    maturity: ClassVar[Optional[Maturity]] = None
    """An enumerated value that identifies the self-selected maturity level of the connector, provided for
    advisory purposes.
    """

    ##
    # Instance configuration

    optimizer: Optional[Optimizer]
    """Name of the command for interacting with the connector instance via the CLI.

    Note that optimizers are attached as configuration to Connector instance because
    the settings are not managed as part of the assembly config files and are always
    provided via environment variablesm, commandline arguments, or secrets management.
    """

    config: BaseConfiguration
    """Configuration for the connector set explicitly or loaded from a config file.
    """

    config_key: Optional[str] = None
    """Key-path to the root of the connector's configuration.
    """

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
            cls.version, Version
        ), "version is not a semantic versioning descriptor"
        return v

    @validator("config_key")
    @classmethod
    def validate_config_key(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/\\.]{3,128}$", v)
        ), "key paths may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
        return v

    @classmethod
    def config_model(cls) -> Type["BaseConfiguration"]:
        """
        Return the configuration model backing the connector. 
        
        The model is determined by the type hint of the `configuration` attribute
        nearest in definition to the target class in the inheritance hierarchy.
        """
        hints = get_type_hints(cls)
        config_cls = hints["config"]
        return config_cls

    ##
    # Events

    @classmethod
    def responds_to_event(cls, event: Union[Event, str]) -> bool:
        """
        Returns True if the Connector processes the specified event (before, on, or after).
        """
        if isinstance(event, str):
            event = get_event(event)

        handlers = list(
            filter(lambda handler: handler.event == event, cls.__event_handlers__)
        )
        return len(handlers) > 0

    @classmethod
    def get_event_handlers(
        cls, event: Union[Event, str], preposition: Preposition = Preposition.ON
    ) -> List[EventHandler]:
        """
        Retrieves the event handlers for the given event and preposition.
        """
        if isinstance(event, str):
            event = get_event(event, None)

        return list(
            filter(
                lambda handler: handler.event == event
                and handler.preposition == preposition,
                cls.__event_handlers__,
            )
        )

    ##
    # Event processing

    def dispatch_event(
        self,
        event: Union[Event, str],
        *args,
        first: bool = False,
        include: Optional[List["Connector"]] = None,
        exclude: Optional[List["Connector"]] = None,
        prepositions: Preposition = (
            Preposition.BEFORE | Preposition.ON | Preposition.AFTER
        ),
        **kwargs,
    ) -> Union[EventResult, List[EventResult]]:
        """
        Dispatches an event to active connectors for processing and returns the results.

        Eventing can be used to notify other connectors of activities and state changes
        driven by one connector or to facilitate loosely coupled cross-connector RPC 
        communication.

        :param first: When True, halt dispatch and return the result from the first connector that responds.
        :param include: A list of specific connectors to dispatch the event to.
        :param exclude: A list of specific connectors to exclude from event dispatch.
        """
        results: List[EventResult] = []
        connectors = include if include is not None else self.__connectors__
        event = get_event(event) if isinstance(event, str) else event

        if exclude:
            # NOTE: We filter by key-paths to avoid recursive hell in Pydantic
            # TODO: Is this necessary/correct? Should just be `config_key` ?
            excluded_keypaths = list(map(lambda c: c.__config_key__, exclude))
            connectors = list(
                filter(lambda c: c.__config_key__ not in excluded_keypaths, connectors)
            )

        # Invoke the before event handlers
        if prepositions & Preposition.BEFORE:
            try:
                for connector in connectors:
                    connector.process_event(event, Preposition.BEFORE, *args, **kwargs)
            except CancelEventError as error:
                # Cancelled by a before event handler. Unpack the result and return it
                return [error.result]

        # Invoke the on event handlers and gather results
        if prepositions & Preposition.ON:
            for connector in connectors:
                connector_results = connector.process_event(
                    event, Preposition.ON, *args, **kwargs
                )
                if connector_results is not None:
                    results.extend(connector_results)
                    if first:
                        break

        # Invoke the after event handlers
        if prepositions & Preposition.AFTER:
            after_args = list(args)
            after_args.insert(0, results)
            for connector in connectors:
                connector.process_event(
                    event, Preposition.AFTER, results, *args, **kwargs
                )

        if first:
            return results[0] if results else None

        return results

    def process_event(
        self, event: Event, preposition: Preposition, *args, **kwargs
    ) -> Optional[List[EventResult]]:
        """
        Process an event and return the results.
        Returns None if the connector does not respond to the event.
        """
        event_handlers = self.get_event_handlers(event, preposition)
        if len(event_handlers) == 0:
            return None

        results: List[EventResult] = []
        for event_handler in event_handlers:
            # NOTE: Explicit kwargs take precendence over those defined during handler declaration
            handler_kwargs = event_handler.kwargs.copy()
            handler_kwargs.update(kwargs)
            try:
                value = event_handler.handler(self, *args, **kwargs)
            except CancelEventError as error:
                if preposition != Preposition.BEFORE:
                    raise TypeError(
                        f"Cannot cancel an event from an {preposition} handler"
                    ) from error

                # Annotate the exception and reraise to halt execution
                error.result = EventResult(
                    connector=self,
                    event=event,
                    preposition=preposition,
                    handler=event_handler,
                    value=error,
                )
                raise error
            except EventError as error:
                value = error

            result = EventResult(
                connector=self,
                event=event,
                preposition=preposition,
                handler=event_handler,
                value=value,
            )
            results.append(result)

        return results

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        _connector_subclasses.add(cls)
        # TODO: Do I need this? I could just put it as a class var also!
        cls.__config_key__ = _config_key_for_connector_class(cls)

        cls.name = cls.__name__.replace("Connector", "")
        cls.full_name = cls.__name__.replace("Connector", " Connector")
        cls.version = Version.parse("0.0.0")

        # Register events handlers for all annotated methods (see `event_handler` decorator)
        for key, value in cls.__dict__.items():
            if handler := getattr(value, "__event_handler__", None):
                if not isinstance(handler, EventHandler):
                    raise TypeError(
                        f"Unexpected event descriptor of type '{handler.__class__}'"
                    )

                handler.connector_type = cls
                cls.__event_handlers__.append(handler)

    def __init__(
        self,
        *,
        config_key: Optional[str] = None,
        __connectors__: List["Connector"] = None,
        **kwargs,
    ):
        config_key = (
            config_key if config_key is not None else self.__class__.__config_key__
        )
        super().__init__(
            config_key=config_key, **kwargs,
        )

        # NOTE: Connector references are held off the model so
        # that Pydantic doesn't see additional attributes
        __connectors__ = __connectors__ if __connectors__ is not None else [self]
        _connector_event_bus[self] = __connectors__

    def __hash__(self):
        return hash((self.name, self.config_key, id(self),))

    @property
    def __connectors__(self) -> List["Connector"]:
        return _connector_event_bus[self]

    ##
    # Subclass services

    def api_client(self) -> httpx.Client:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
        headers = {
            "Authorization": f"Bearer {self.optimizer.token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
        return httpx.Client(base_url=self.optimizer.api_url, headers=headers)

    @property
    def logger(self) -> logging.Logger:
        """Returns the logger"""
        return loguru.logger


_is_base_connector_class_defined = True
EventResult.update_forward_refs(Connector=Connector)
EventHandler.update_forward_refs(Connector=Connector)


def _config_key_for_connector_class(cls: Type[Connector]) -> str:
    name = re.sub(r"Connector$", "", cls.__name__)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def metadata(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[Version] = None,
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
                version if isinstance(version, Version) else Version.parse(version)
            )
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = license
        if maturity:
            cls.maturity = maturity
        return cls

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
