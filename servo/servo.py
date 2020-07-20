import abc
import importlib
import inspect
import json
import os
import re
from contextvars import ContextVar
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Type, Union

import httpx
import yaml
from pydantic import BaseModel, Extra, Field, create_model, validator
from pydantic.json import pydantic_encoder
from pydantic.schema import schema as pydantic_schema

import servo
from servo import connector
from servo.connector import (
    BaseConfiguration,
    Connector,
    ConnectorLoader,
    License,
    Maturity,
    Optimizer,
    _connector_subclasses,
)
from servo.events import Preposition, event, on_event
from servo.types import Check, Control, Description, Measurement, Metric
from servo.utilities import join_to_series


class Events(str, Enum):
    """
    Events is an enumeration of the names of events defined by the servo.
    """

    # Lifecycle events
    STARTUP = "startup"
    SHUTDOWN = "shutdown"

    # Informational events
    METRICS = "metrics"
    COMPONENTS = "components"

    # Operational events
    CHECK = "check"
    DESCRIBE = "describe"
    MEASURE = "measure"
    ADJUST = "adjust"
    PROMOTE = "promote"


class _EventDefinitions:
    """
    Defines the default events. This class is declarative and is never directly referenced.

    The event signature is inferred from the decorated function.
    """

    # Lifecycle events
    @event(Events.STARTUP)
    def startup(self) -> None:
        pass

    @event(Events.SHUTDOWN)
    def shutdown(self) -> None:
        pass

    # Informational events
    @event(Events.METRICS)
    def metrics(self) -> List[Metric]:
        pass

    @event(Events.COMPONENTS)
    def components(self) -> Description:
        pass

    # Operational events
    @event(Events.MEASURE)
    def measure(
        self, *, metrics: List[str] = None, control: Control = Control()
    ) -> Measurement:
        pass

    @event(Events.CHECK)
    def check(self) -> List[Check]:
        pass

    @event(Events.DESCRIBE)
    def describe(self) -> Description:
        pass

    @event(Events.ADJUST)
    def adjust(self, data: dict) -> dict:
        pass

    @event(Events.PROMOTE)
    def promote(self) -> None:
        pass


class BaseServoConfiguration(BaseConfiguration, abc.ABC):
    """
    Abstract base class for Servo settings

    Note that the concrete BaseServoConfiguration class is built dynamically at runtime
    based on the avilable connectors and configuration in effect.

    See `ServoAssembly` for details on how the concrete model is built.
    """

    connectors: Optional[Union[List[str], Dict[str, str]]] = Field(
        None,
        description=(
            "An optional, explicit configuration of the active connectors.\n"
            "\nConfigurable as either an array of connector identifiers (names or class) or\n"
            "a dictionary where the keys specify the key path to the connectors configuration\n"
            "and the values identify the connector (by name or class name)."
        ),
        examples=[
            ["kubernetes", "prometheus"],
            {"staging_prom": "prometheus", "gateway_prom": "prometheus"},
        ],
    )
    """
    An optional list of connector keys or a dict mapping of connector 
    key-paths to connector class names
    """

    @classmethod
    def generate(
        cls: Type["BaseServoConfiguration"], **kwargs
    ) -> "BaseServoConfiguration":
        """
        Generate configuration for the servo settings
        """
        for name, field in cls.__fields__.items():
            if (
                name not in kwargs
                and inspect.isclass(field.type_)
                and issubclass(field.type_, BaseConfiguration)
            ):
                kwargs[name] = field.type_.generate()
        return cls(**kwargs)

    @validator("connectors", pre=True)
    @classmethod
    def validate_connectors(
        cls, connectors
    ) -> Optional[Union[Dict[str, str], List[str]]]:
        if isinstance(connectors, str):
            # NOTE: Special case. When we are invoked with a string it is typically an env var
            try:
                decoded_value = BaseServoConfiguration.__config__.json_loads(connectors)  # type: ignore
            except ValueError as e:
                raise ValueError(f'error parsing JSON for "{connectors}"') from e

            # Prevent infinite recursion
            if isinstance(decoded_value, str):
                raise ValueError(
                    f'JSON string values for `connectors` cannot parse into strings: "{connectors}"'
                )

            connectors = decoded_value

        connectors = _normalize_connectors(connectors)
        # NOTE: Will raise if invalid
        _routes_for_connectors_descriptor(connectors)
        return connectors

    class Config:
        extra = Extra.forbid
        title = "Abstract Servo Configuration Schema"
        env_prefix = "SERVO_"


@connector.metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
    version=servo.__version__,
)
class Servo(Connector):
    """
    The Servo
    """

    config: BaseServoConfiguration
    """Configuration for the Servo.

    Note that the Servo configuration is built dynamically at Servo assembly time.
    The concrete type is built in `ServoAssembly.assemble()` and adds a field
    for each active connector.
    """

    routes: Dict[str, Connector]
    """Routes for active connectors.

    The keys are key-paths that map to the implicit or explicit connector declarations 
    in the configuration. The values are fully configured Connector instances.
    """

    def __init__(self, *args, routes: Dict[str, Connector], **kwargs) -> None:
        super().__init__(*args, routes={}, **kwargs)

        # Ensure the routes refer to the same objects by identity
        self.routes.update(routes)        

    def startup(self):
        return self.broadcast_event(Events.STARTUP, prepositions=Preposition.ON)

    def shutdown(self):
        return self.broadcast_event(Events.SHUTDOWN, prepositions=Preposition.ON)

    @property
    def connectors(self) -> List[Connector]:
        """
        Returns a list of the active connectors.
        """
        return list(self.routes.values())

    ##
    # Event handlers

    @on_event()
    async def check(self) -> List[Check]:
        from servo.servo_runner import APIEvent, APIRequest

        async with self.api_client() as client:
            event_request = APIRequest(event=APIEvent.HELLO)
            response = await client.post("servo", data=event_request.json())
            success = (response.status_code == httpx.codes.OK)
            return [Check(
                name="Opsani API connectivity",
                success=success,
                comment=f"Response status code: {response.status_code}",
            )]


# Context vars for asyncio tasks managed by ServoAssembly
_assembly_context_var = ContextVar('servo.assembly', default=None)
_servo_context_var = ContextVar('servo.servo', default=None)


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
    loading and creating a concrete BaseServoConfiguration model that supports
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
    config_model: Type[BaseServoConfiguration]
    servo: Servo

    @classmethod
    def assemble(
        cls,
        *,
        config_file: Path,
        optimizer: Optimizer,
        env: Optional[Dict[str, str]] = os.environ,
        **kwargs,
    ) -> ("ServoAssembly", Servo, Type[BaseServoConfiguration]):
        """Assembles a Servo by processing configuration and building a dynamic settings model"""

        _discover_connectors()
        ServoConfiguration, routes = _create_config_model(
            config_file=config_file, env=env
        )

        # Build our Servo configuration from the config file + environment
        if config_file.exists():
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            if not (config is None or isinstance(config, dict)):
                raise ValueError(
                    f'error: config file "{config_file}" parsed to an unexpected value of type "{config.__class__}"'
                )
            config = {} if config is None else config
            servo_config = ServoConfiguration.parse_obj(config)
        else:
            # If we do not have a config file, build a minimal configuration
            # NOTE: This configuration is likely incomplete/invalid due to required
            # settings on the connectors not being fully configured
            args = kwargs.copy()
            for name, connector_type in routes.items():
                args[name] = connector_type.config_model().construct()
            servo_config = ServoConfiguration.construct(**args)

        # Initialize all active connectors
        connectors: List[Connector] = []
        servo_routes: Dict[str, Connector] = {}
        for name, connector_type in routes.items():
            connector_config = getattr(servo_config, name)
            if connector_config:
                # NOTE: If the command is routed but doesn't define a settings class this will raise
                connector = connector_type(
                    name=name,
                    config=connector_config,
                    optimizer=optimizer,
                    __connectors__=connectors,
                )
                servo_routes[name] = connector
                connectors.append(connector)

        # Build the servo object
        servo = Servo(
            config=servo_config,
            routes=servo_routes,
            optimizer=optimizer,
            __connectors__=connectors,
        )
        connectors.append(servo)
        assembly = ServoAssembly(
            config_file=config_file,
            optimizer=optimizer,
            config_model=ServoConfiguration,
            servo=servo,
        )

        # Set the context vars
        _assembly_context_var.set(assembly)
        _servo_context_var.set(servo)

        return assembly, servo, ServoConfiguration

    def __init__(self, *args, servo: Servo, **kwargs):
        super().__init__(*args, servo=servo, **kwargs)

        # Ensure object is shared by identity
        self.servo = servo

    ##
    # Utility functions

    @classmethod
    def all_connector_types(cls) -> Set[Type[Connector]]:
        """Return a set of all connector types in the assembly excluding the Servo"""
        return _connector_subclasses

    @property
    def connectors(self) -> List[Connector]:
        """
        Returns a list of all active connectors in the assembly including the Servo.
        """
        return [self.servo, *self.servo.connectors]

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        """Returns a schema that only includes connector model definitions"""
        connectors = self.all_connector_types() if all else self.servo.connectors
        config_models = list(map(lambda c: c.config_model(), connectors))
        return pydantic_schema(config_models, title="Servo Schema")

    def top_level_schema_json(self, *, all: bool = False) -> str:
        """Return a JSON string representation of the top level schema"""
        return json.dumps(
            self.top_level_schema(all=all), indent=2, default=pydantic_encoder
        )


def _module_path(cls: Type) -> str:
    if cls.__module__:
        return ".".join([cls.__module__, cls.__name__])
    else:
        return cls.__name__


def _discover_connectors() -> Set[Type[Connector]]:
    """
    Discover available connectors that are registered via setuptools entry points.

    See ConnectorLoader for details.
    """
    connectors = set()
    loader = ConnectorLoader()
    for connector in loader.load():
        connectors.add(connector)
    return connectors


def _create_config_model_from_routes(
    routes=Dict[str, Type[Connector]], *, require_fields: bool = True,
) -> Type[BaseServoConfiguration]:
    # Create Pydantic fields for each active route
    connector_versions: Dict[
        Type[Connector], str
    ] = {}  # use dict for uniquing and ordering
    setting_fields: Dict[str, Tuple[Type[BaseConfiguration], Any]] = {}
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
    servo_config_model = create_model(
        "ServoConfiguration", __base__=BaseServoConfiguration, **setting_fields,
    )

    connectors_series = join_to_series(list(connector_versions.values()))
    servo_config_model.__config__.title = "Servo Configuration Schema"
    servo_config_model.__config__.schema_extra = {
        "description": f"Schema for configuration of Servo v{Servo.version} with {connectors_series}"
    }

    return servo_config_model


def _create_config_model(
    *,
    config_file: Path,
    routes: Dict[str, Type[Connector]] = None,
    env: Optional[Dict[str, str]] = os.environ,
) -> (Type[BaseServoConfiguration], Dict[str, Type[Connector]]):
    # map of config key in YAML to settings class for target connector
    if routes is None:
        routes = _default_routes()
    require_fields: bool = False

    # NOTE: If `connectors` key is present in config file, require the keys to be present
    if config_file.exists():
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if isinstance(config, dict):  # Config file could be blank or malformed
            connectors_value = config.get("connectors", None)
            if connectors_value:
                routes = _routes_for_connectors_descriptor(connectors_value)
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
    connector_type: Type[Connector]
    connector_name: str
    config_model: Type[BaseConfiguration]

    def __init__(
        self,
        connector_type: Type[Connector],
        connector_name: str,
        config_model: Type[BaseConfiguration],
    ) -> None:
        self.connector_type = connector_type
        self.connector_name = connector_name
        self.config_model = config_model

    def __str__(self):
        return f"{self.config_model} for {self.connector_type.__name__} named '{self.connector_name}'"


__config_models_cache__: List[SettingModelCacheEntry] = []


def _derive_config_model_for_route(
    name: str, model: Type[Connector]
) -> Type[BaseConfiguration]:
    """Inherits a new Pydantic model from the given settings and set up nested environment variables"""
    # NOTE: It is important to produce a new model name to disambiguate the models within Pydantic
    # because key-paths are guanranteed to be unique, we can utilize it as a cache key
    base_config_model = model.config_model()

    # See if we already have a matching model
    config_model: Optional[Type[BaseConfiguration]] = None
    model_names = set()
    for cache_entry in __config_models_cache__:
        model_names.add(cache_entry.config_model.__name__)
        if cache_entry.connector_type is model and cache_entry.connector_name == name:
            config_model = cache_entry.config_model
            break

    if config_model is None:
        if base_config_model == BaseConfiguration:
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
            model_name = _normalize_name(
                f"{base_config_model.__qualname__}__{name}"
            )

        config_model = create_model(model_name, __base__=base_config_model)

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


def _connector_class_from_string(connector: str) -> Optional[Type[Connector]]:
    if not isinstance(connector, str):
        return None

    # Check for an existing class in the namespace
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
    for connector_class in ServoAssembly.all_connector_types():
        if connector == connector_class.__default_name__ or connector in [
            connector_class.__name__,
            connector_class.__qualname__,
        ]:
            return connector_class

    # Try to load it as a module path
    if "." in connector:
        module_path, class_name = connector.rsplit(".", 1)
        module = importlib.import_module(module_path)
        connector_class = getattr(module, class_name)
        if _validate_class(connector_class):
            return connector_class

    return None


def _validate_class(connector: type) -> bool:
    if not isinstance(connector, type):
        return False

    if not issubclass(connector, Connector):
        raise TypeError(f"{connector.__name__} is not a Connector subclass")

    return True


def _normalize_connectors(connectors: Optional[Iterable]) -> Optional[Iterable]:
    if connectors is None:
        return connectors
    elif isinstance(connectors, str):
        if _connector_class_from_string(connectors) is None:
            raise ValueError(f"Invalid connectors value: {connectors}")
        return connectors
    elif isinstance(connectors, type) and issubclass(connectors, Connector):
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


def _routes_for_connectors_descriptor(connectors):
    if connectors is None:
        # None indicates that all available connectors should be activated
        return None

    elif isinstance(connectors, str):
        # NOTE: Special case. When we are invoked with a string it is typically an env var
        try:
            decoded_value = BaseServoConfiguration.__config__.json_loads(connectors)  # type: ignore
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
        connector_map = _default_routes()
        reserved_keys = _reserved_keys()

        connector_routes = {}
        for name, value in connectors.items():
            if not isinstance(name, str):
                raise TypeError(f'Connector names must be strings: "{key}"')

            # Validate the name
            try:
                Connector.validate_name(name)
            except AssertionError as e:
                raise ValueError(f'"{name}" is not a valid connector name: {e}') from e

            # Resolve the connector class
            if isinstance(value, type):
                connector_class = value
            elif isinstance(value, str):
                connector_class = _connector_class_from_string(value)

            # Check for key reservations
            if name in reserved_keys:
                if c := connector_map.get(name, None):
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


RESERVED_KEYS = ["connectors", "control", "measure", "adjust", "optimization"]


def _reserved_keys() -> List[str]:
    reserved_keys = list(_default_routes().keys())
    reserved_keys.extend(RESERVED_KEYS)
    return reserved_keys


def _default_routes() -> Dict[str, Type[Connector]]:
    routes = {}
    for connector in _connector_subclasses:
        if connector is not Servo:
            routes[connector.__default_name__] = connector
    return routes
