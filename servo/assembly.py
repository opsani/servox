from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import os
import pathlib
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import pydantic
import pydantic.json
import yaml

import servo.configuration
import servo.connector
import servo.servo

__all__ = ["Assembly"]


_assembly_context_var = contextvars.ContextVar("servo.Assembly.current", default=None)


class Assembly(pydantic.BaseModel):
    """
    An Assembly models the environment and runtime configuration of a collection of Servos.

    Connectors are dynamically loaded via setuptools entry points
    (see https://packaging.python.org/specifications/entry-points/)
    and the settings class for a Servo instance must be created at
    runtime because the servo.yaml configuration file includes settings
    for an arbitrary number of connectors mounted onto arbitrary keys
    in the config file.

    The Assembly class is responsible for handling the connector
    loading and creating a concrete BaseServoConfiguration model that supports
    the connectors available and activated in the assembly. An assembly
    is the combination of configuration and associated code artifacts
    in an executable environment (e.g. a Docker image or a Python virtualenv
    running on your workstation.

    NOTE: The Assembly class overrides the Pydantic base class implementations
    of the schema family of methods. See the method docstrings for specific details.
    """

    config_file: pathlib.Path
    servos: List[servo.servo.Servo]

    @staticmethod
    def current() -> "Assembly":
        """
        Return the active assembly for the current execution context.

        The value is managed by a contextvar and is concurrency safe.
        """
        return _assembly_context_var.get()

    @classmethod
    def assemble(
        cls,
        *,
        config_file: pathlib.Path,
        optimizer: Optional[servo.configuration.Optimizer], # TODO: Move this into the config model?
        env: Optional[Dict[str, str]] = os.environ,
        **kwargs,
    ) -> "Assembly":
        """Assemble a Servo by processing configuration and building a dynamic settings model"""

        _discover_connectors()

        # Build our Servo configuration from the config file + environment
        if not config_file.exists():
            raise FileNotFoundError(f"config file '{config_file}' does not exist")

        configs = yaml.load_all(open(config_file), Loader=yaml.FullLoader)
        if not isinstance(configs, Iterator):
            raise ValueError(
                f'error: config file "{config_file}" parsed to an unexpected value of type "{configs.__class__}"'
            )

        # if len(configs) > 1 and optimizer is not None:
        #     raise ValueError("cannot configure a multi-servo assembly with a single optimizer")

        servos: List[servo.servo.Servo] = []
        for config in configs:
            # TODO: Needs to be public / have a better name
            # TODO: We need to index the env vars here for multi-servo
            servo_config_model, routes = _create_config_model(
                config=config, env=env
            )
            servo_config = servo_config_model.parse_obj(config)
            # TODO: Can probably dump the optimizer option
            # servo_config.optimizer = optimizer

            # Initialize all active connectors
            connectors: List[servo.connector.BaseConnector] = []
            for name, connector_type in routes.items():
                connector_config = getattr(servo_config, name)
                if connector_config is not None:
                    connector = connector_type(
                        name=name,
                        config=connector_config,
                        optimizer=servo_config.optimizer,
                        __connectors__=connectors,
                    )
                    connectors.append(connector)

            # Build the servo object
            servo_ = servo.servo.Servo(
                config=servo_config,
                connectors=connectors.copy(),  # Avoid self-referential reference to servo
                optimizer=servo_config.optimizer,
                __connectors__=connectors,
            )
            connectors.append(servo_)
            servos.append(servo_)

        assembly = Assembly(
            config_file=config_file,
            servos=servos,
        )

        # Set the context vars
        _assembly_context_var.set(assembly)

        # Activate the servo if we are in the common case single player mode
        if len(servos) == 1:
            servo.servo.Servo.set_current(servos[0])
        
        return assembly

    def __init__(self, *args, servos: List[servo.Servo], **kwargs) -> None: # noqa: D107
        super().__init__(*args, servos=servos, **kwargs)

        # Ensure object is shared by identity
        self.servos = servos

    ##
    # Utility functions

    async def dispatch_event(
        self,
        event: Union[servo.events.Event, str],
        *args,
        first: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        prepositions: servo.events.Preposition = (
            servo.events.Preposition.BEFORE | servo.events.Preposition.ON | servo.events.Preposition.AFTER
        ),
        return_exceptions: bool = False,
        **kwargs,
    ) -> Union[Optional[servo.events.EventResult], List[servo.events.EventResult]]:
        """Dispatch an event to all servos active in the assembly."""

        group = asyncio.gather(
            *list(
                map(
                    lambda s: s.dispatch_event(
                        event,
                        *args,
                        first=first,
                        include=include,
                        exclude=exclude,
                        prepositions=prepositions,
                        **kwargs
                    ),
                    self.servos,
                )
            ),
            return_exceptions=return_exceptions,
        )
        results = await group
        if results:
            results = functools.reduce(lambda x, y: x + y, results)

        # TODO: This needs to be tested in multi-servo
        if first:
            return results[0] if results else None

        return results

    @classmethod
    def all_connector_types(cls) -> Set[Type[servo.connector.BaseConnector]]:
        """Return a set of all connector types in the assembly excluding the Servo"""
        return servo.connector._connector_subclasses.copy()

    @property
    def connectors(self) -> List[servo.connector.BaseConnector]:
        """
        Return a list of all active connectors in the assembly including the Servo.
        """
        return [self.servo, *self.servo.connectors]

    @property
    def servo(self) -> servo.servo.Servo:
        return self.servos[0]

    async def add_servo(self, servo_: servo.servo.Servo) -> None:
        """Add a servo to the assembly.

        Once added, the servo is sent the startup event to prepare for execution.

        Args:
            servo_: The servo to add to the assembly.
        """
        self.servos.append(servo_)
        await servo.startup()

    async def remove_servo(self, servo_: servo.servo.Servo) -> None:
        """Remove a servo from the assembly.

        Before removal, the servo is sent the shutdown event to prepare for
        eviction from the assembly.

        Args:
            servo_: The servo to remove from the assembly.
        """
        await servo.shutdown()
        self.servos.remove(servo_)

    async def startup(self):
        """Notify all servos that the assembly is starting up.
        """
        await asyncio.gather(
                *list(
                    map(
                        lambda s: s.startup(),
                        self.servos,
                    )
                )
            )

    async def shutdown(self):
        """Notify all servos that the assembly is shutting down.
        """
        await asyncio.gather(
                *list(
                    map(
                        lambda s: s.shutdown(),
                        self.servos,
                    )
                )
            )

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        """Return a schema that only includes connector model definitions"""
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
    Discover available connectors that are registered via setuptools entry points.

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
) -> Type[servo.configuration.BaseServoConfiguration]:
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
        __base__=servo.configuration.BaseServoConfiguration,
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

# TODO: This needs a public API and better name. Prolly moves to configuration module
# TODO: Is this actually different from _create_config_model_from_routes?
def _create_config_model(
    *,
    config: Dict[str, Any], # TODO: Could be optional?
    routes: Dict[str, Type[servo.connector.BaseConnector]] = None,
    env: Optional[Dict[str, str]] = os.environ,
) -> Tuple[
    Type[servo.configuration.BaseServoConfiguration],
    Dict[str, Type[servo.connector.BaseConnector]],
]:
    require_fields: bool = False
    # map of config key in YAML to settings class for target connector
    routes = servo.connector._default_routes() if routes is None else routes

    if isinstance(config, dict):
        # NOTE: If `connectors` key is present in the config, require the keys to be present
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

# TODO: Move to the strings utility module
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
    ) -> None: # noqa: D107
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
