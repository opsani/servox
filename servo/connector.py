import abc
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
)
from servo.types import License, Maturity, Version
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

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = Extra.forbid
        fields = {
            "token": {"env": "OPSANI_TOKEN",},
            "base_url": {"env": "OPSANI_BASE_URL",},
        }


DEFAULT_TITLE = "Connector Configuration Schema"


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

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = Extra.forbid
        title = DEFAULT_TITLE


# Uppercase handling for non-subclassed settings models. Should be pushed into Pydantic as a PR
env_names = BaseConfiguration.__fields__["description"].field_info.extra.get(
    "env_names", set()
)
BaseConfiguration.__fields__["description"].field_info.extra["env_names"] = set(
    map(str.upper, env_names)
)


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

    configuration: BaseConfiguration
    """Configuration for the connector set explicitly or loaded from a config file.
    """

    config_key_path: str
    """Key-path to the root of the connector's configuration.
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
            cls.version, Version
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
    def config_model(cls) -> Type["BaseConfiguration"]:
        """
        Return the configuration model backing the connector. 
        
        The effective type of the configuration instance is defined by the type hint definitions of the 
        `config_model` and `configuration` level attributes closest in definition to the target class.
        """
        hints = get_type_hints(cls)
        config_cls = hints["configuration"]
        return config_cls

    ##
    # Events

    @classmethod
    def create_event(cls, name: str, signature: Union[Callable, Signature]) -> Event:
        if cls.__events__.get(name, None):
            raise ValueError(f"Event '{name}' has already been created")

        signature = (
            signature
            if isinstance(signature, Signature)
            else Signature.from_callable(signature)
        )
        if list(
            filter(
                lambda param: param.kind == Parameter.VAR_POSITIONAL,
                signature.parameters.values(),
            )
        ):
            raise TypeError(
                f"Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
            )

        event = Event(name=name, signature=signature)
        cls.__events__[name] = event
        return event

    @classmethod
    def responds_to_event(cls, event: Union[Event, str]) -> bool:
        """
        Returns True if the Connector processes the specified event (before, on, or after).
        """
        if isinstance(event, str):
            event = cls.__events__.get(event)

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
            event = cls.__events__.get(event)

        return list(
            filter(
                lambda handler: handler.event == event
                and handler.preposition == preposition,
                cls.__event_handlers__,
            )
        )

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

    # subclass registry of connectors
    __connectors__: Set[Type["Connector"]] = set()
    __events__: Dict[str, Event] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.__connectors__.add(cls)
        cls.__key_path__ = _key_path_for_connector_class(cls)

        cls.name = cls.__name__.replace("Connector", " Connector")
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
        self, *, config_key_path: Optional[str] = None, **kwargs,
    ):
        config_key_path = (
            config_key_path
            if config_key_path is not None
            else self.__class__.__key_path__
        )
        super().__init__(
            config_key_path=config_key_path, **kwargs,
        )

    ##
    # Subclass services

    def api_client(self) -> httpx.Client:
        """Yields an httpx.Client instance configured to talk to Opsani API"""
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


_is_base_connector_class_defined = True
EventResult.update_forward_refs(Connector=Connector)
EventHandler.update_forward_refs(Connector=Connector)


def _key_path_for_connector_class(cls: Type[Connector]) -> str:
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


def event(
    name: Optional[str] = None, *, handler: bool = False
) -> Callable[[EventCallable], EventCallable]:
    """
    Creates a new event using the signature of the decorated function.

    Events must be defined before handlers can be registered using before_event, on_event, after_event, or
    event_handler.

    :param handler: When True, the decorated function implementation is registered as an on event handler.
    """

    def decorator(fn: EventCallable) -> EventCallable:
        event_name = name if name else fn.__name__
        Connector.create_event(event_name, fn)

        if handler:
            decorator = on_event(event_name)
            return decorator(fn)
        else:
            return fn

    return decorator


def before_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run before the specified event.

    Before event handlers require no arguments positional or keyword arguments and return `None`. Any arguments
    provided via the `kwargs` parameter are passed through at invocation time. Before event handlers 
    can cancel event propagation by raising `CancelEventError`. Canceled events are reported to the
    event originator by attaching the `CancelEventError` instance to the `EventResult`.

    :param event: The event or name of the event to run the handler before.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.BEFORE, **kwargs)


def on_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run on the specified event.

    :param event: The event or name of the event to run the handler on.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.ON, **kwargs)


def after_event(
    event: Optional[str] = None, **kwargs
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler to run after the specified event.

    After event handlers are invoked with the event results as their first argument (type `List[EventResult]`)
    and return `None`.

    :param event: The event or name of the event to run the handler after.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """
    return event_handler(event, Preposition.AFTER, **kwargs)


def event_handler(
    event_name: Optional[str] = None,
    preposition: Preposition = Preposition.ON,
    **kwargs,
) -> Callable[[EventCallable], EventCallable]:
    """
    Registers the decorated function as an event handler.

    Event handlers are the foundational mechanism for connectors to provide functionality
    to the servo assembly. As events occur during operation, handlers are invoked and the
    results are aggregated and evalauted by the servo. Event handlers are registered for
    a specific event preposition, enabling them to execute before, after, or during an event.

    :param event: Specifies the event name. If not given, inferred from the name of the decorated handler function.
    :param preposition: Specifies the sequencing of a handler in relation to the event.
    :param kwargs: An optional dictionary of supplemental arguments to be passed when the handler is called.
    """

    def decorator(fn: EventCallable) -> EventCallable:
        name = event_name if event_name else fn.__name__
        event = Connector.__events__.get(name, None)
        if event is None:
            raise ValueError(f"Unknown event '{name}'")

        if preposition != Preposition.ON:
            name = f"{preposition} {name}"
        handler_signature = Signature.from_callable(fn)

        if preposition == Preposition.BEFORE:
            before_handler_signature = Signature.from_callable(__before_handler)
            _validate_handler_signature(
                handler_signature,
                event_signature=before_handler_signature,
                handler_name=name,
            )
        elif preposition == Preposition.ON:
            _validate_handler_signature(
                handler_signature, event_signature=event.signature, handler_name=name
            )
        elif preposition == Preposition.AFTER:
            after_handler_signature = Signature.from_callable(__after_handler)
            _validate_handler_signature(
                handler_signature,
                event_signature=after_handler_signature,
                handler_name=name,
            )
        else:
            assert "Undefined preposition value"

        # Annotate the function for processing later, see Connector.__init_subclass__
        fn.__event_handler__ = EventHandler(
            event=event, preposition=preposition, handler=fn, kwargs=kwargs
        )
        return fn

    return decorator


def __before_handler(self) -> None:
    pass


def __after_handler(self, results: List[EventResult]) -> None:
    pass


def _validate_handler_signature(
    handler_signature: Signature, *, event_signature: Signature, handler_name: str
) -> None:
    """
    Validates that the given handler signature is compatible with the event signature. Validation
    checks the parameter and return value types using annotations. The intent is to immediately
    expose errors in event handlers rather than encountering them at runtime (which may take 
    an arbitrary amount of time to trigger a given event). Raises a TypeError when an incompatibility 
    is encountered.

    :param handler_signature: The event handler signature to validate.
    :param event_signature: The reference event signature to validate against.
    :param handler_name: The name of the handler for inclusion in error messages & logs.
    """

    # Skip the work if the signatures are identical
    if handler_signature == event_signature:
        return

    handler_parameters: Mapping[str, Parameter] = handler_signature.parameters
    handler_positional_parameters = list(
        filter(
            lambda param: param.kind
            in [Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL],
            handler_parameters.values(),
        )
    )
    handler_keyword_parameters = dict(
        filter(
            lambda item: item[1].kind
            in [
                Parameter.KEYWORD_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_KEYWORD,
            ],
            handler_parameters.items(),
        )
    )

    event_parameters: Mapping[str, Parameter] = event_signature.parameters
    event_positional_parameters = list(
        filter(
            lambda param: param.kind
            in [Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL],
            event_parameters.values(),
        )
    )
    event_keyword_parameters = dict(
        filter(
            lambda item: item[1].kind
            in [
                Parameter.KEYWORD_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
                Parameter.VAR_KEYWORD,
            ],
            event_parameters.items(),
        )
    )

    # We assume instance methods
    args = list(handler_parameters.keys())
    first_arg = args.pop(0) if args else None
    if first_arg != "self":
        raise TypeError(
            f"Invalid signature for '{handler_name}' event handler: {handler_signature}, \"self\" must be the first argument"
        )

    # Check return type annotation
    if handler_signature.return_annotation != event_signature.return_annotation:
        raise TypeError(
            f"Invalid return type annotation for '{handler_name}' event handler: expected {event_signature.return_annotation}, but found {handler_signature.return_annotation}"
        )

    # Check for extraneous positional parameters on the handler
    handler_positional_only = list(
        filter(
            lambda param: param.kind == Parameter.POSITIONAL_ONLY,
            handler_positional_parameters,
        )
    )
    event_positional_only = list(
        filter(
            lambda param: param.kind == Parameter.POSITIONAL_ONLY,
            event_positional_parameters,
        )
    )
    if len(handler_positional_only) > len(event_positional_only):
        extra_param_names = sorted(
            list(
                set(map(lambda p: p.name, handler_positional_only))
                - set(map(lambda p: p.name, event_positional_only))
            )
        )
        raise TypeError(
            f"Invalid type annotation for '{handler_name}' event handler: encountered extra positional parameters ({join_to_series(extra_param_names)})"
        )

    # Check for extraneous keyword parameters on the handler
    handler_keyword_nonvar = dict(
        filter(
            lambda item: item[1].kind != Parameter.VAR_KEYWORD,
            handler_keyword_parameters.items(),
        )
    )
    event_keyword_nonvar = dict(
        filter(
            lambda item: item[1].kind != Parameter.VAR_KEYWORD,
            event_keyword_parameters.items(),
        )
    )
    extraneous_keywords = sorted(
        list(set(handler_keyword_nonvar.keys()) - set(event_keyword_nonvar.keys()))
    )
    if extraneous_keywords:
        raise TypeError(
            f"Invalid type annotation for '{handler_name}' event handler: encountered extra parameters ({join_to_series(extraneous_keywords)})"
        )

    # Iterate the event signature parameters and see if the handler's signature satisfies each one
    for index, (parameter_name, event_parameter) in enumerate(event_parameters.items()):
        if event_parameter.kind == Parameter.POSITIONAL_ONLY:
            if index > len(handler_positional_parameters) - 1:
                if handler_positional_parameters[-1].kind != Parameter.VAR_POSITIONAL:
                    raise TypeError(
                        f"Missing required positional parameter: '{parameter_name}'"
                    )

            handler_parameter = handler_positional_parameters[index]
            if handler_parameter != Parameter.VAR_POSITIONAL:
                # Compare types
                if handler_parameter.annotation != event_parameter.annotation:
                    raise TypeError(
                        f"Incorrect type annotation for positional parameter '{parameter_name}': expected {event_parameter.annotation}, but found {handler_parameter.annotation}"
                    )

                if (
                    handler_parameter.return_annotation
                    != event_parameter.return_annotation
                ):
                    raise TypeError(
                        f"Incorrect return type annotation for positional parameter '{parameter_name}': expected {event_parameter.return_annotation}, but found {handler_parameter.return_annotation}"
                    )

        elif event_parameter.kind == Parameter.VAR_POSITIONAL:
            # NOTE: This should never happen
            raise TypeError(
                "Invalid signature: events cannot declare variable positional arguments (e.g. *args)"
            )

        elif event_parameter.kind in [
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ]:
            if handler_parameter := handler_keyword_parameters.get(
                parameter_name, None
            ):
                # We have the keyword arg, check the types
                if handler_parameter.annotation != event_parameter.annotation:
                    raise TypeError(
                        f"Incorrect type annotation for parameter '{parameter_name}': expected {event_parameter.annotation}, but found {handler_parameter.annotation}"
                    )
            else:
                # Check if the last parameter is a VAR_KEYWORD
                if (
                    list(handler_keyword_parameters.values())[-1].kind
                    != Parameter.VAR_KEYWORD
                ):
                    raise TypeError(
                        f"Missing required parameter: '{parameter_name}': expected signature: {event_signature}"
                    )

        else:
            assert event_parameter.kind == Parameter.VAR_KEYWORD, event_parameter.kind


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
