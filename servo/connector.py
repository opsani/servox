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

"""Provides an extensible, event-driven interface for connecting Servo assemblies with external systems.

Connectors are the foundational unit of functionality within the Servo. Connectors emit and respond to events
such as measure and adjust in order to drive optimization activities. Because there are so many sources of metrics
data and ways to orchestrate cloud infrastructure, the servo exposes a flexible plugin interface that enables
integration with arbitrary systems via the connector module.
"""
from __future__ import annotations

import abc
import contextlib
import contextvars
import importlib
import re
from typing import (
    Any,
    ClassVar,
    Generator,
    Iterable,
    Optional,
    Set,
    Tuple,
    Type,
    get_type_hints,
)

import loguru
import importlib.metadata
import pydantic

import servo.api
import servo.configuration
import servo.events
import servo.logging
import servo.pubsub
import servo.repeating
import servo.telemetry
import servo.utilities.associations
from servo.types import *

__all__ = [
    "BaseConnector",
    "current_connector",
    "metadata",
]


_current_context_var = contextvars.ContextVar("servox.current_connector", default=None)


def current_connector() -> Optional["BaseConnector"]:
    """Return the active connector for the current execution context.

    The value is managed by a contextvar and is concurrency safe.
    """
    return _current_context_var.get(None)


_connector_subclasses: Set[Type["BaseConnector"]] = set()


# NOTE: Initialize mixins first to control initialization graph
class BaseConnector(
    servo.utilities.associations.Mixin,
    servo.events.Mixin,
    servo.logging.Mixin,
    servo.pubsub.Mixin,
    servo.repeating.Mixin,
    pydantic.BaseModel,
    abc.ABC,
    metaclass=servo.events.Metaclass,
):
    """Connectors expose functionality to Servo assemblies by connecting external services and resources."""

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

    cryptonym: ClassVar[Optional[str]] = None
    """Optional code name of the version.
    """

    description: ClassVar[Optional[str]] = None
    """Optional textual description of the connector.
    """

    homepage: ClassVar[Optional[pydantic.HttpUrl]] = None
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

    config: servo.configuration.BaseConfiguration
    """Configuration for the connector set explicitly or loaded from a config file."""

    # TODO: needs better name... BaseCommonConfiguration? attr can be _base_config or __base_config__
    # NOTE: __shared__ maybe?
    _global_config: servo.configuration.CommonConfiguration = pydantic.PrivateAttr(
        default_factory=lambda: servo.configuration.CommonConfiguration()
    )
    """Shared configuration from our parent Servo instance."""

    _optimizer: Optional[servo.configuration.OptimizerTypes] = pydantic.PrivateAttr(
        default=None
    )
    """Shared optimizer from our parent Servo instance."""

    @property
    def optimizer(
        self,
    ) -> Optional[servo.configuration.OptimizerTypes]:
        """The optimizer for the connector."""
        return self._optimizer

    ##
    # Shared telemetry metadata
    telemetry: servo.telemetry.Telemetry = pydantic.Field(
        default_factory=servo.telemetry.Telemetry
    )

    ##
    # Validators

    @pydantic.root_validator(pre=True)
    @classmethod
    def _validate_metadata(cls, v):
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
                raise ValueError(
                    f"A default connector name could not be constructed for class '{cls}'"
                )
        return v

    @pydantic.validator("name")
    @classmethod
    def _validate_name(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/\\.]{3,128}$", v)
        ), "names may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
        return v

    @classmethod
    def config_model(cls) -> Type[servo.configuration.BaseConfiguration]:
        """
        Return the configuration model backing the connector.

        The model is determined by the type hint of the `configuration` attribute
        nearest in definition to the target class in the inheritance hierarchy.
        """
        hints = get_type_hints(cls)
        config_cls = hints["config"]
        return config_cls

    @classmethod
    def version_summary(cls) -> str:
        cryptonym_ = f' "{cls.cryptonym}"' if cls.cryptonym else ""
        return f"{cls.full_name} v{cls.version}{cryptonym_}"

    @classmethod
    def summary(cls) -> str:
        cryptonym_ = f' "{cls.cryptonym}"' if cls.cryptonym else ""
        return (
            f"{cls.full_name} v{cls.version}{cryptonym_} ({cls.maturity})\n"
            f"{cls.description}\n"
            f"{cls.homepage}\n"
            f"Licensed under the terms of {cls.license}"
        )

    def __init_subclass__(cls: Type["BaseConnector"], **kwargs) -> None:  # noqa: D105
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
    ) -> None:  # noqa: D107
        name = name if name is not None else self.__class__.__default_name__
        super().__init__(
            *args,
            name=name,
            **kwargs,
        )

    def __hash__(self):  # noqa: D105
        return hash(
            (
                self.name,
                id(self),
            )
        )

    @property
    def logger(self) -> "loguru.Logger":
        """Return a logger object bound to the connector."""
        # NOTE: We support the explicit connector ref and the context var so
        # that logging is attributable outside of an event whenever possible
        return super().logger.bind(connector=self)

    @contextlib.contextmanager
    def current(self):
        """A context manager that sets the current connector context."""
        try:
            token = _current_context_var.set(self)
            yield self

        finally:
            _current_context_var.reset(token)


def metadata(
    name: Optional[Union[str, Tuple[str, str]]] = None,
    description: Optional[str] = None,
    version: Optional[Union[str, Version]] = None,
    *,
    homepage: Optional[Union[str, pydantic.HttpUrl]] = None,
    license: Optional[Union[str, License]] = None,
    maturity: Optional[Union[str, Maturity]] = None,
    cryptonym: Optional[str] = None,
):
    """Decorate a Connector class with metadata."""

    def decorator(cls):
        if not issubclass(cls, BaseConnector):
            raise TypeError("Metadata can only be attached to Connector subclasses")

        if name:
            if isinstance(name, tuple):
                if len(name) != 2:
                    raise ValueError(
                        f"Connector names given as tuples must contain exactly 2 elements: full name and alias"
                    )
                cls.name, cls.__default_name__ = name
            else:
                cls.name = name
        if description:
            cls.description = description
        if version:
            cls.version = (
                version if isinstance(version, Version) else Version.parse(version)
            )
        cls.cryptonym = cryptonym
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = (
                license if isinstance(license, License) else License.from_str(license)
            )
        if maturity:
            cls.maturity = (
                maturity
                if isinstance(maturity, Maturity)
                else Maturity.from_str(maturity)
            )
        return cls

    return decorator


##
# Utility functions


def _name_for_connector_class(cls: Type[BaseConnector]) -> Optional[str]:
    for name in (cls.name, cls.__name__):
        if not name:
            continue
        name = re.sub(r"Connector$", "", name)
        if re.match(r"^[A-Z]+$", name):
            # Handle case where the name is an acronym (e.g. 'OLAS') => 'olas'
            name = name.lower()
        else:
            # Handle case where the name is CamelCase (e.g., 'DataDog') => 'data_dog'
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        if name != "":
            return name
    return None


#####

ENTRY_POINT_GROUP = "servo.connectors"


class ConnectorLoader:
    """Discover and load connectors via Python setuptools entry points."""

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:  # noqa: D107
        self.group = group

    def iter_entry_points(self) -> tuple[importlib.metadata.EntryPoint]:
        return importlib.metadata.entry_points()[self.group]

    def load(self) -> Generator[Any, None, None]:
        for entry_point in self.iter_entry_points():
            yield entry_point.load()


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
            decoded_value = servo.configuration.BaseServoConfiguration.__config__.json_loads(connectors)  # type: ignore
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
                raise ValueError(f'no connector found for the identifier "{connector}"')

        return connector_routes

    elif isinstance(connectors, dict):
        connector_routes = {}
        for name, value in connectors.items():
            if not isinstance(name, str):
                raise TypeError(f'Connector names must be strings: "{name}"')

            # Validate the name
            try:
                BaseConnector._validate_name(name)
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
                        raise ValueError(f'Name "{name}" is reserved by `{c.__name__}`')
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
