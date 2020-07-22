import abc
import importlib
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Optional,
    Set,
    Type,
    get_type_hints,
)

from pkg_resources import EntryPoint, iter_entry_points
from pydantic import (
    BaseModel,
    HttpUrl,
    root_validator,
    validator,
)


from servo import api, events, logging, repeating
from servo.configuration import BaseConfiguration, Optimizer
from servo.events import (
    CancelEventError,
    Event,
    EventCallable,
    EventContext,
    EventError,
    EventHandler,
    EventResult,
    Preposition,
    get_event,
)
from servo.types import *
from servo.utilities import join_to_series


_connector_subclasses: Set[Type["Connector"]] = set()


# NOTE: Initialize mixins first to control initialization graph
class Connector(api.Mixin, events.Mixin, logging.Mixin, repeating.Mixin, BaseModel, abc.ABC, metaclass=events.Metaclass):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """
    
    ##
    # Connector metadata

    name: str = None
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

    ##
    # Validators

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

    @validator("name")
    @classmethod
    def validate_name(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/\\.]{3,128}$", v)
        ), "names may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        _connector_subclasses.add(cls)

        cls.name = cls.__name__.replace("Connector", "")
        cls.full_name = cls.__name__.replace("Connector", " Connector")
        cls.version = Version.parse("0.0.0")
        cls.__default_name__ = _name_for_connector_class(cls)

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        **kwargs,
    ):
        name = (
            name if name is not None else self.__class__.__default_name__
        )
        super().__init__(
            *args, name=name, **kwargs,
        )

    def __hash__(self):
        return hash((self.name, id(self),))


EventResult.update_forward_refs(Connector=Connector)
EventHandler.update_forward_refs(Connector=Connector)


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

##
# Utility functions

def _name_for_connector_class(cls: Type[Connector]) -> str:
    name = re.sub(r"Connector$", "", cls.__name__)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _connector_class_from_string(connector: str) -> Optional[Type[Connector]]:
    if not isinstance(connector, str):
        return None

    # Check for an existing class in the namespace
    # FIXME: This symbol lookup doesn't seem solid
    connector_class = globals().get(connector, None)
    try:
        connector_class = (
            eval(connector) if connector_class is None else connector_class
        )
    except Exception:
        pass
    
    if _validate_class(connector_class):
        return connector_class

    # Check if the string is an identifier for a connector
    for connector_class in _connector_subclasses:
        if connector == connector_class.__default_name__ or connector in [
            connector_class.__name__,
            connector_class.__qualname__,
        ]:
            return connector_class

    # Try to load it as a module path
    if "." in connector:
        module_path, class_name = connector.rsplit(".", 1)
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            connector_class = getattr(module, class_name)
            if _validate_class(connector_class):
                return connector_class

    return None

def _validate_class(connector: type) -> bool:
    if connector is None or not isinstance(connector, type):
        return False

    if not issubclass(connector, Connector):
        raise TypeError(f"{connector.__name__} is not a Connector subclass")

    return True


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
