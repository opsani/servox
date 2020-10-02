from __future__ import annotations
import abc
import importlib
import re
from pkg_resources import EntryPoint, iter_entry_points
from typing import (
    Any,
    ClassVar,
    IO,
    Iterable,
    Generator,
    Optional,
    Set,
    Type,
    Tuple,
    get_type_hints,
)

import loguru
from pydantic import (
    BaseModel,
    HttpUrl,
    root_validator,
    validator,
)


from servo import api, events, logging, repeating
from servo.configuration import BaseConfiguration, Optimizer
from servo.events import EventHandler, EventResult
from servo.types import *
from servo.utilities import associations


_connector_subclasses: Set[Type["BaseConnector"]] = set()


# NOTE: Initialize mixins first to control initialization graph
class BaseConnector(associations.Mixin, api.Mixin, events.Mixin, logging.Mixin, repeating.Mixin, BaseModel, abc.ABC, metaclass=events.Metaclass):
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
        
        if not cls.__default_name__:
            if name := _name_for_connector_class(cls):
                cls.__default_name__ = name
            else:
                raise ValueError(f"A default connector name could not be constructed for class '{cls}'")
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

    def __init_subclass__(cls: Type['BaseConnector'], **kwargs):
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
    
    @property
    def api_client_options(self) -> Dict[str, Any]:
        return self.__dict__.get("api_client_options", super().api_client_options)
    
    @property
    def logger(self) -> loguru.Logger:
        """Returns a contextualized logger"""
        # NOTE: We support the explicit connector ref and the context var so
        # that logging is attributable outside of an event whenever possible
        return super().logger.bind(connector=self)


EventResult.update_forward_refs(BaseConnector=BaseConnector)
EventHandler.update_forward_refs(BaseConnector=BaseConnector)


def metadata(
    name: Optional[Union[str, Tuple[str, str]]] = None,
    description: Optional[str] = None,
    version: Optional[Union[str, Version]] = None,
    *,
    homepage: Optional[Union[str, HttpUrl]] = None,
    license: Optional[Union[str, License]] = None,
    maturity: Optional[Union[str, Maturity]] = None,
):
    """Decorate a Connector class with metadata"""

    def decorator(cls):
        if not issubclass(cls, BaseConnector):
            raise TypeError("Metadata can only be attached to Connector subclasses")

        if name:
            if isinstance(name, tuple):
                if len(name) != 2:
                    raise ValueError(f"Connector names given as tuples must contain exactly 2 elements: full name and alias")
                cls.name, cls.__default_name__ = name                
            else:
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
            cls.license = license if isinstance(license, License) else License.from_str(license)
        if maturity:
            cls.maturity = maturity if isinstance(maturity, Maturity) else Maturity.from_str(maturity)
        return cls

    return decorator

##
# Utility functions

def _name_for_connector_class(cls: Type[BaseConnector]) -> Optional[str]:
    for name in (cls.name, cls.__name__):
        if not name:
            continue
        name = re.sub(r"Connector$", "", name)
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        if name != "":
            return name
    return None


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


def _normalize_connectors(connectors: Optional[Iterable]) -> Optional[Iterable]:
    if connectors is None:
        return connectors
    elif isinstance(connectors, str):
        if _connector_class_from_string(connectors) is None:
            raise ValueError(f"Invalid connectors value: {connectors}")
        return connectors
    elif isinstance(connectors, type) and issubclass(connectors, BaseConnector):
        return connectors.__name__
    elif isinstance(connectors, (list, tuple, set)):
        connectors_list: List[str] = []
        for connector in connectors:
            connectors_list.append(_normalize_connectors(connector))
        return connectors_list
    elif isinstance(connectors, dict):
        normalized_dict: Dict[str, str] = {}
        for key, value in connectors.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Connector descriptor keys must be strings (invalid value '{key}'"
                )
            normalized_dict[key] = _normalize_connectors(value)

        return normalized_dict
    else:
        raise ValueError(f"Invalid connectors value: {connectors}")

def _routes_for_connectors_descriptor(connectors) -> Dict[str, "BaseConnector"]:
    if connectors is None:
        # None indicates that all available connectors should be activated
        return None

    elif isinstance(connectors, str):
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

        return _routes_for_connectors_descriptor(decoded_value)

    elif isinstance(connectors, (list, tuple, set)):
        connector_routes: Dict[str, str] = {}
        for connector in connectors:
            if _validate_class(connector):
                connector_routes[connector.__default_name__] = connector
            elif connector_class := _connector_class_from_string(connector):
                connector_routes[connector_class.__default_name__] = connector_class
            else:
                raise ValueError(f"Missing validation for value {connector}")

        return connector_routes

    elif isinstance(connectors, dict):
        connector_routes = {}
        for name, value in connectors.items():
            if not isinstance(name, str):
                raise TypeError(f'Connector names must be strings: "{name}"')

            # Validate the name
            try:
                BaseConnector.validate_name(name)
            except AssertionError as e:
                raise ValueError(f'"{name}" is not a valid connector name: {e}') from e

            # Resolve the connector class
            if isinstance(value, type):
                connector_class = value
            elif isinstance(value, str):
                connector_class = _connector_class_from_string(value)
            else:
                raise ValueError(f'"{value}" is not a string or type')

            # Check for key reservations
            if name in _reserved_keys():
                if c := _default_routes().get(name, None):
                    if connector_class != c:
                        raise ValueError(
                            f'Name "{name}" is reserved by `{c.__name__}`'
                        )
                else:
                    raise ValueError(f'Name "{name}" is reserved')

            connector_routes[name] = connector_class

        return connector_routes

    else:
        raise ValueError(
            f"Unexpected type `{type(connectors).__qualname__}`` encountered (connectors: {connectors})"
        )

def _connector_class_from_string(connector: str) -> Optional[Type["BaseConnector"]]:
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

    if not issubclass(connector, BaseConnector):
        raise TypeError(f"{connector.__name__} is not a Connector subclass")

    return True


RESERVED_KEYS = ["connectors", "control", "measure", "adjust", "optimization"]


def _reserved_keys() -> List[str]:
    reserved_keys = list(_default_routes().keys())
    reserved_keys.extend(RESERVED_KEYS)
    return reserved_keys


def _default_routes() -> Dict[str, Type[BaseConnector]]:
    from servo.servo import Servo
    routes = {}
    for connector in _connector_subclasses:
        if connector is not Servo:
            routes[connector.__default_name__] = connector
    return routes
