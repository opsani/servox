from __future__ import annotations

import contextvars
import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import pydantic
import pydantic.json
import yaml

import servo.configuration
import servo.connector
import servo.servo

__all__ = ["Assembly", "_assembly_context_var"]


_assembly_context_var = contextvars.ContextVar("servo.Assembly.current", default=None)


class Assembly(pydantic.BaseModel):
    """
    An Assembly models the environment and runtime configuration of a Servo.

    Connectors are dynamically loaded via setuptools entry points
    (see https://packaging.python.org/specifications/entry-points/)
    and the settings class for a Servo instance must be created at
    runtime because the servo.yaml configuration file includes settings
    for an arbitrary number of connectors mounted onto arbitrary keys
    in the config file.

    The Assembly class is responsible for handling the connector
    loading and creating a concrete BaseAssemblyConfiguration model that supports
    the connectors available and activated in the assembly. An assembly
    is the combination of configuration and associated code artifacts
    in an executable environment (e.g. a Docker image or a Python virtualenv
    running on your workstation.

    NOTE: The Assembly class overrides the Pydantic base class implementations
    of the schema family of methods. See the method docstrings for specific details.
    """

    ## Configuration
    config_file: pathlib.Path
    optimizer: servo.configuration.Optimizer

    ## Assembled settings & Servo
    config_model: Type[servo.configuration.BaseAssemblyConfiguration]
    servo: servo.servo.Servo

    @staticmethod
    def current() -> "Assembly":
        """
        Returns the active assembly for the current execution context.

        The value is managed by a contextvar and is concurrency safe.
        """
        return _assembly_context_var.get()

    @classmethod
    def assemble(
        cls,
        *,
        config_file: pathlib.Path,
        optimizer: servo.configuration.Optimizer,
        env: Optional[Dict[str, str]] = os.environ,
        **kwargs,
    ) -> Tuple["Assembly", servo.Servo, Type[servo.configuration.BaseAssemblyConfiguration]]:
        """Assembles a Servo by processing configuration and building a dynamic settings model"""

        _discover_connectors()

        # Build our Servo configuration from the config file + environment
        if not config_file.exists():
            raise FileNotFoundError(f"config file '{config_file}' does not exist")

        AssemblyConfiguration, routes = _create_config_model(
            config_file=config_file, env=env
        )

        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if not (config is None or isinstance(config, dict)):
            raise ValueError(
                f'error: config file "{config_file}" parsed to an unexpected value of type "{config.__class__}"'
            )
        config = {} if config is None else config
        assembly_config = AssemblyConfiguration.parse_obj(config)

        # Initialize all active connectors
        connectors: List[servo.connector.BaseConnector] = []
        for name, connector_type in routes.items():
            connector_config = getattr(assembly_config, name)
            if connector_config:
                # NOTE: If the command is routed but doesn't define a settings class this will raise
                connector = connector_type(
                    name=name,
                    config=connector_config,
                    optimizer=optimizer,
                    __connectors__=connectors,
                )
                connectors.append(connector)

        # Build the servo object
        servo_ = servo.servo.Servo(
            config=assembly_config,
            connectors=connectors.copy(),  # Avoid self-referential reference to servo
            optimizer=optimizer,
            __connectors__=connectors,
        )
        connectors.append(servo_)
        assembly = Assembly(
            config_file=config_file,
            optimizer=optimizer,
            config_model=AssemblyConfiguration,
            servo=servo_,
        )

        # Set the context vars
        _assembly_context_var.set(assembly)
        servo.servo.Servo.set_current(servo_)

        return assembly, servo_, AssemblyConfiguration

    def __init__(self, *args, servo: servo.Servo, **kwargs) -> None:
        super().__init__(*args, servo=servo, **kwargs)

        # Ensure object is shared by identity
        self.servo = servo

    ##
    # Utility functions

    @classmethod
    def all_connector_types(cls) -> Set[Type[servo.connector.BaseConnector]]:
        """Returns a set of all connector types in the assembly excluding the Servo"""
        return servo.connector._connector_subclasses.copy()

    @property
    def connectors(self) -> List[servo.connector.BaseConnector]:
        """
        Returns a list of all active connectors in the assembly including the Servo.
        """
        return [self.servo, *self.servo.connectors]

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        """Returns a schema that only includes connector model definitions"""
        connectors = self.all_connector_types() if all else self.servo.connectors
        config_models = list(map(lambda c: c.config_model(), connectors))
        return pydantic.schema.schema(config_models, title="Servo Schema")

    def top_level_schema_json(self, *, all: bool = False) -> str:
        """Return a JSON string representation of the top level schema"""
        return json.dumps(
            self.top_level_schema(all=all),
            indent=2,
            default=pydantic.json.pydantic_encoder,
        )


def _module_path(cls: Type) -> str:
    if cls.__module__:
        return ".".join([cls.__module__, cls.__name__])
    else:
        return cls.__name__


def _discover_connectors() -> Set[Type[servo.connector.BaseConnector]]:
    """
    Discovers available connectors that are registered via setuptools entry points.

    See ConnectorLoader for details.
    """
    connectors = set()
    loader = servo.connector.ConnectorLoader()
    for connector in loader.load():
        connectors.add(connector)
    return connectors


def _create_config_model_from_routes(
    routes=Dict[str, Type[servo.connector.BaseConnector]],
    *,
    require_fields: bool = True,
) -> Type[servo.configuration.BaseAssemblyConfiguration]:
    # Create Pydantic fields for each active route
    connector_versions: Dict[
        Type[servo.connector.BaseConnector], str
    ] = {}  # use dict for uniquing and ordering
    setting_fields: Dict[
        str, Tuple[Type[servo.configuration.BaseConfiguration], Any]
    ] = {}
    default_value = (
        ... if require_fields else None
    )  # Pydantic uses ... for flagging fields required

    for name, connector_class in routes.items():
        config_model = _derive_config_model_for_route(name, connector_class)
        config_model.__config__.title = (
            f"{connector_class.full_name} Settings (named {name})"
        )
        setting_fields[name] = (config_model, default_value)
        connector_versions[
            connector_class
        ] = f"{connector_class.full_name} v{connector_class.version}"

    # Create our model
    servo_config_model = pydantic.create_model(
        "ServoConfiguration",
        __base__=servo.configuration.BaseAssemblyConfiguration,
        **setting_fields,
    )

    connectors_series = servo.utilities.join_to_series(
        list(connector_versions.values())
    )
    servo_config_model.__config__.title = "Servo Configuration Schema"
    servo_config_model.__config__.schema_extra = {
        "description": f"Schema for configuration of Servo v{servo.Servo.version} with {connectors_series}"
    }

    return servo_config_model


def _create_config_model(
    *,
    config_file: pathlib.Path,
    routes: Dict[str, Type[servo.connector.BaseConnector]] = None,
    env: Optional[Dict[str, str]] = os.environ,
) -> Tuple[
    Type[servo.configuration.BaseAssemblyConfiguration],
    Dict[str, Type[servo.connector.BaseConnector]],
]:
    # map of config key in YAML to settings class for target connector
    if routes is None:
        routes = servo.connector._default_routes()
    require_fields: bool = False

    # NOTE: If `connectors` key is present in config file, require the keys to be present
    if config_file.exists():
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if isinstance(config, dict):  # Config file could be blank or malformed
            connectors_value = config.get("connectors", None)
            if connectors_value:
                routes = servo.connector._routes_for_connectors_descriptor(
                    connectors_value
                )
                require_fields = True

    servo_config_model = _create_config_model_from_routes(
        routes, require_fields=require_fields
    )
    return servo_config_model, routes


def _normalize_name(name: str) -> str:
    """
    Normalizes the given name.
    """
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", name)


class SettingModelCacheEntry:
    connector_type: Type[servo.connector.BaseConnector]
    connector_name: str
    config_model: Type[servo.configuration.BaseConfiguration]

    def __init__(
        self,
        connector_type: Type[servo.connector.BaseConnector],
        connector_name: str,
        config_model: Type[servo.configuration.BaseConfiguration],
    ) -> None:
        self.connector_type = connector_type
        self.connector_name = connector_name
        self.config_model = config_model

    def __str__(self):
        return f"{self.config_model} for {self.connector_type.__name__} named '{self.connector_name}'"


__config_models_cache__: List[SettingModelCacheEntry] = []


def _derive_config_model_for_route(
    name: str, model: Type[servo.connector.BaseConnector]
) -> Type[servo.configuration.BaseConfiguration]:
    """Inherits a new Pydantic model from the given settings and set up nested environment variables"""
    # NOTE: It is important to produce a new model name to disambiguate the models within Pydantic
    # because key-paths are guanranteed to be unique, we can utilize it as a cache key
    base_config_model = model.config_model()

    # See if we already have a matching model
    config_model: Optional[Type[servo.configuration.BaseConfiguration]] = None
    model_names = set()
    for cache_entry in __config_models_cache__:
        model_names.add(cache_entry.config_model.__name__)
        if cache_entry.connector_type is model and cache_entry.connector_name == name:
            config_model = cache_entry.config_model
            break

    if config_model is None:
        if base_config_model == servo.configuration.BaseConfiguration:
            # Connector hasn't defined a settings class or is reusing one, use the connector class name as base name
            # This is essential for preventing `KeyError` exceptions in Pydantic schema generation
            # due to naming conflicts.
            model_name = f"{model.__name__}Settings"
        elif (
            name == model.__default_name__
            and not f"{base_config_model.__qualname__}" in model_names
        ):
            # Connector is mounted at the default route, use default name
            model_name = f"{base_config_model.__qualname__}"
        else:
            model_name = _normalize_name(f"{base_config_model.__qualname__}__{name}")

        config_model = pydantic.create_model(model_name, __base__=base_config_model)

    # Cache it for reuse
    cache_entry = SettingModelCacheEntry(
        connector_type=model, connector_name=name, config_model=config_model
    )
    __config_models_cache__.append(cache_entry)

    # Traverse across all the fields and update the env vars
    for field_name, field in config_model.__fields__.items():
        field.field_info.extra.pop("env", None)
        field.field_info.extra["env_names"] = {f"SERVO_{name}_{field_name}".upper()}

    return config_model
