import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import typer
import yaml
from pydantic import BaseModel, Extra, create_model, validator
from pydantic.json import pydantic_encoder
from pydantic.schema import schema as pydantic_schema

from servo.connector import (
    Connector,
    ConnectorLoader,
    ConnectorSettings,
    License,
    Maturity,
    Optimizer,
    metadata,
)


class BaseServoSettings(ConnectorSettings):
    """
    Abstract base class for Servo settings

    Note that the concrete BaseServoSettings class is built dynamically at runtime
    based on the avilable connectors and configuration in effect.

    See `ServoAssembly` for details on how the concrete model is built.
    """

    optimizer: Optimizer
    """The Opsani optimizer the Servo is attached to"""

    connectors: Optional[Dict[str, str]] = None
    """A map of connector key-paths to fully qualified class names"""

    @validator("connectors", pre=True)
    @classmethod
    def validate_connectors(cls, connectors) -> Optional[Dict[str, str]]:
        if routes := _routes_for_connectors_descriptor(connectors):
            path_to_modules = {}
            for path, connector_class in routes.items():
                path_to_modules[path] = _module_path(connector_class)
            return path_to_modules

    class Config:
        # We are the base root of pluggable configuration
        # so we ignore any extra fields so you can turn connectors on and off
        extra = Extra.ignore

        title = "Servo"


# class ConnectorResponse(BaseModel):
#     event: str
#     connector: Connector
#     data: Any


@metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
)
class Servo(Connector):
    """The Servo"""

    settings: BaseServoSettings
    connectors: List[Connector] = []

    ##
    # Event processing

    def send_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Dispatch an event"""

    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Handle an event"""

    ##
    # Misc

    def cli(self) -> typer.Typer:
        # TODO: Return the root CLI?
        pass


class ServoAssembly(BaseModel):
    """
    A Servo assembly models the runtime configuration of a Servo.

    Connectors are dynamically loaded via setuptools entry points
    (see https://packaging.python.org/specifications/entry-points/)
    and the settings class for a Servo instance must be created at
    runtime because the servo.yaml configuration file includes settings
    for an arbitrary number of connectors mounted onto arbitrary keys 
    in the config file.

    The ServoAssembly class is responsible for handling the connector
    loading and creating a concrete BaseServoSettings model that supports
    the connectors available and activated in the assembly. An assembly
    is the combination of configuration and associated code artifacts
    in an executable environment (e.g. a Docker image or a Python virtualenv
    running on your workstation.

    NOTE: The ServoAssembly class overrides the Pydantic base class implementations
    of the schema family of methods. See the method docstrings for specific details.
    """

    ## Configuration
    config_file: Path
    optimizer: Optimizer

    ## Assembled settings & Servo
    settings_model: Type[BaseServoSettings]
    servo: Servo

    @classmethod
    def assemble(
        cls,
        *,
        config_file: Path,
        optimizer: Optimizer,
        env: Optional[Dict[str, str]] = os.environ,
        **kwargs,
    ) -> ("ServoAssembly", Servo, Type[BaseServoSettings]):
        """Assembles a Servo by processing configuration and building a dynamic settings model"""

        _discover_connectors()
        ServoSettings, connector_type_routes = _create_settings_model(
            config_file=config_file, env=env
        )

        # Build our Servo settings instance from the config file + environment
        if config_file.exists():
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            if not (config is None or isinstance(config, dict)):
                raise ValueError(
                    f'error: config file "{config_file}" parsed to an unexpected value of type "{config.__class__}"'
                )
            config = {} if config is None else config
            config["optimizer"] = optimizer.dict()
            settings = ServoSettings.parse_obj(config)
        else:
            # If we do not have a config file, build a minimal configuration
            # NOTE: This configuration is likely incomplete/invalid due to required
            # settings on the connectors not being fully configured
            args = kwargs.copy()
            for c in cls.all_connectors():
                args[c.default_path()] = c.settings_model().construct()
            settings = ServoSettings(optimizer=optimizer, **args)

        # Initialize all active connectors
        connectors: List[Connector] = []
        for key_path, connector_type in connector_type_routes.items():
            connector_settings = getattr(settings, key_path)
            if connector_settings:
                # NOTE: If the command is routed but doesn't define a settings class this will raise
                connector = connector_type(connector_settings)
                connectors.append(connector)

        # Build the servo object
        servo = Servo(settings, connectors=connectors)
        assembly = ServoAssembly(
            config_file=config_file,
            optimizer=optimizer,
            settings_model=ServoSettings,
            servo=servo,
        )

        # TODO: The dynamic settings class needs to map key-paths to connector classes

        return assembly, servo, ServoSettings

    ##
    # Utility functions

    def parse_file(self, config_file: Path = None) -> BaseServoSettings:
        config_file = self.config_file if config_file is None else config_file
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config["optimizer"] = self.optimizer.dict()
        return self.settings_model.parse_obj(config)

    @classmethod
    def default_routes(cls) -> Dict[str, Type[Connector]]:
        routes = {}
        for connector in cls.all_connectors():
            mounts[connector.default_path()] = connector
        return routes

    @classmethod
    def all_connectors(cls) -> Set[Type[Connector]]:
        """Return a set of all connectors in the assembly excluding the Servo"""
        connectors = set()
        for c in Connector.all():
            if c != Servo:
                connectors.add(c)
        return connectors

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        """Returns a schema that only includes connector model definitions"""
        connectors = self.all_connectors() if all else self.servo.connectors
        settings_models = list(map(lambda c: c.settings_model(), connectors))
        return pydantic_schema(settings_models, title="Servo Schema")

    def top_level_schema_json(self, *, all: bool = False) -> str:
        """Return a JSON string representation of the top level schema"""
        return json.dumps(
            self.top_level_schema(all=all), indent=2, default=pydantic_encoder
        )

    # # TODO: Override the schema functions to decouple CLI from settings model


def _module_path(cls: Type) -> str:
    if cls.__module__:
        return ".".join([cls.__module__, cls.__name__])
    else:
        return cls.__name__


def _discover_connectors() -> Set[Type[Connector]]:
    """
    Discover available connectors that are registered via setuptools
    entry points.

    See ConnectorLoader for details.
    """
    connectors = set()
    loader = ConnectorLoader()
    for connector in loader.load():
        connectors.add(connector)
    return connectors


def _create_settings_model(
    *, config_file: Path, env: Optional[Dict[str, str]] = os.environ
) -> (Type[BaseServoSettings], Dict[str, Type[Connector]]):
    # map of config key in YAML to settings class for target connector
    setting_fields: Optional[Dict[str, Type[ConnectorSettings]]] = None
    key_paths_to_settings_type_names: Dict[str, str] = {}
    key_paths_to_connector_types: Dict[str, Type[Connector]] = {}

    # NOTE: If `connectors` key is present in config file, require the keys to be present
    if config_file.exists():
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if isinstance(config, dict):  # Config file could be blank or malformed
            connectors_value = config.get("connectors", None)
            if connectors_value:
                routes = _routes_for_connectors_descriptor(connectors_value)
                setting_fields = {}
                for path, connector_class in routes.items():
                    base_settings_model = connector_class.settings_model()
                    settings_model = create_model(
                        f"{path}_{base_settings_model.__qualname__}",
                        __base__=base_settings_model,
                    )

                    # Traverse across all the fields and update the env vars
                    for name, field in settings_model.__fields__.items():
                        field.field_info.extra["env_names"] = {
                            f"SERVO_{path}_{name}".upper()
                        }

                    setting_fields[path] = (settings_model, ...)
                    # TODO: Bundle this into class...
                    key_paths_to_settings_type_names[path] = _module_path(
                        settings_model
                    )
                    key_paths_to_connector_types[path] = connector_class

    # If we don't have any target connectors, add all available as optional fields
    if setting_fields is None:
        setting_fields = {}
        for c in Connector.all():
            if c is not Servo:
                setting_fields[c.default_path()] = (c.settings_model(), None)
                key_paths_to_settings_type_names[c.default_path()] = _module_path(
                    c.settings_model()
                )  # RENAME: module types
                key_paths_to_connector_types[c.default_path()] = c

    # Create our model
    servo_settings_model = create_model(
        "ServoSettings",
        __base__=BaseServoSettings,
        optimizer=(Optimizer, ...),
        # connectors=key_paths_to_settings_type_names,
        **setting_fields,
    )

    return servo_settings_model, key_paths_to_connector_types


def _connector_class_from_string(connector: str) -> Optional[Type[Connector]]:
    if not isinstance(connector, str):
        return None

    # Check fo an existing class in the namespace
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
    for connector_class in ServoAssembly.all_connectors():
        if connector == connector_class.default_path() or connector in [
            connector_class.__name__,
            connector_class.__qualname__,
        ]:
            return connector_class

    # Try to load it as a module path
    if "." in connector:
        module_path, class_name = e.split(":", 2)
        module = importlib.import_module(module_path)
        connector_class = getattr(module, class_name)
        if _validate_class(connector_class):
            return connector_class

    raise ValueError(f"{connector} does not identify a Connector class")


def _validate_class(connector: type) -> bool:
    if not isinstance(connector, type):
        return False

    if not issubclass(connector, Connector):
        raise TypeError(f"{connector.__name__} is not a Connector subclass")

    return True


def _routes_for_connectors_descriptor(connectors):
    if connectors is None:
        # None indicates that all available connectors should be activated
        return None
    elif isinstance(connectors, str):
        # NOTE: Special case. When we are invoked with a string it is typically an env var
        try:
            decoded_value = BaseServoSettings.__config__.json_loads(connectors)  # type: ignore
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
                connector_routes[connector.default_path()] = connector
            elif connector_class := _connector_class_from_string(connector):
                connector_routes[connector_class.default_path()] = connector_class
            else:
                raise ValueError(f"Missing validation for value {connector}")

        return connector_routes

    elif isinstance(connectors, dict):
        connector_map = _default_routes()
        reserved_keys = _reserved_keys()

        connector_routes = {}
        for config_path, value in connectors.items():
            if not isinstance(config_path, str):
                raise ValueError(f'Key "{config_path}" is not a string')

            # Validate the key format
            try:
                Connector.validate_config_path(config_path)
            except AssertionError as e:
                raise ValueError(f'Key "{config_path}" is not valid: {e}') from e

            # Resolve the connector class
            if isinstance(value, type):
                connector_class = value
            elif isinstance(value, str):
                connector_class = _connector_class_from_string(value)

            # Check for key reservations
            if config_path in reserved_keys:
                if c := connector_map.get(config_path, None):
                    if connector_class != c:
                        raise ValueError(
                            f'Key "{config_path}" is reserved by `{c.__name__}`'
                        )
                else:
                    raise ValueError(f'Key "{config_path}" is reserved')

            connector_routes[config_path] = connector_class

        return connector_routes

    else:
        raise ValueError(
            f"Unexpected type `{type(connectors).__qualname__}`` encountered (connectors: {connectors})"
        )


def _reserved_keys() -> List[str]:
    reserved_keys = list(_default_routes().keys())
    reserved_keys.extend(
        ["connectors", "control", "measure", "adjust"]
    )  # TODO: make this a constant...
    return reserved_keys


def _default_routes() -> Dict[str, Type[Connector]]:
    routes = {}
    for connector in Connector.all():
        if connector is not Servo:
            routes[connector.default_path()] = connector
    return routes
