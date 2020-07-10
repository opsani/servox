import importlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Iterable

import typer
import httpx
import yaml

from pydantic import BaseModel, Extra, Field, create_model, validator
from pydantic.json import pydantic_encoder
from pydantic.schema import schema as pydantic_schema
from servo.connector import (
    Connector,
    ConnectorLoader,
    ConnectorSettings,
    EventResult,
    License,
    Maturity,
    Optimizer,
    metadata,
    event,
)
from servo.types import Event, EventRequest, CheckResult
from servo.utilities import join_to_series
import inspect

class Events(str, Enum):
    """
    Defines the standard Servo events.
    """

    CHECK = "check"
    DESCRIBE = "describe"
    MEASURE = "measure"
    ADJUST = "adjust"
    PROMOTE = "promote"


# TODO: Make abstract , abc.ABC
class BaseServoSettings(ConnectorSettings):
    """
    Abstract base class for Servo settings

    Note that the concrete BaseServoSettings class is built dynamically at runtime
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
    def generate(cls: Type["BaseServoSettings"], **kwargs) -> "BaseServoSettings":
        """
        Generate configuration for the servo settings
        """
        for name, field in cls.__fields__.items():
            if name not in kwargs and inspect.isclass(field.type_) and issubclass(field.type_, ConnectorSettings):
                kwargs[name] = field.type_.generate()
        return cls(**kwargs)

    @validator("connectors", pre=True)
    @classmethod
    def validate_connectors(cls, connectors) -> Optional[Union[Dict[str, str], List[str]]]:
        if isinstance(connectors, str):
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
            
            connectors = decoded_value

        connectors = _normalize_connectors(connectors)
        # NOTE: Will raise if invalid
        _routes_for_connectors_descriptor(connectors)
        return connectors


    class Config:
        # We are the base root of pluggable configuration
        # so we ignore any extra fields so you can turn connectors on and off
        extra = Extra.ignore
        title = "Abstract Servo Configuration Schema"
        env_prefix = "SERVO_"


@metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
)
class Servo(Connector):
    """
    The Servo
    """

    settings: BaseServoSettings
    """Settings for the Servo.

    Note that the Servo settings are dynamically built at Servo assembly time.
    The concrete 
    """

    connectors: List[Connector] = []
    """The active connectors within the Servo.
    """

    def __init__(
        self, 
        *args, 
        connectors: List[Connector] = [],
        **kwargs
    ) -> None:
        super().__init__(*args, connectors=connectors, **kwargs)

        # NOTE: The Servo itself is an event processor
        self.connectors.append(self)

    @event()
    def check(self) -> CheckResult:
        from servo.types import Event, EventRequest
        with self.api_client() as client:
            event_request = EventRequest(event=Event.HELLO)
            response = client.post("servo", data=event_request.json())
            if response.status_code != httpx.codes.OK:
                return CheckResult(name="Check Opsani API connectivity", success=False, comment=f"Encountered an unexpected status code of {response.status_code} when connecting to Opsani")

        return CheckResult(name="Check Servo", success=True, comment="All checks passed successfully.")

    ##
    # Event processing

    def dispatch_event(
        self,
        event: str,
        *args,
        first: bool = False,
        all: bool = False,
        include: Optional[List[Connector]] = None,
        exclude: Optional[List[Connector]] = None,
        **kwargs,
    ) -> Union[EventResult, List[EventResult]]:
        """
        Dispatches an event to active connectors for processing and returns the results.

        :param first: When True, halt dispatch and return the result from the first connector that responds.
        :param all: When True, the event is dispatched to all connectors available rather than the active ones.
        :param include: A list of specific connectors to dispatch the event to.
        :param exclude: A list of specific connectors to exclude from event dispatch.
        """
        if all:
            raise RuntimeError("Not yet implemented.")
        results: List[EventResult] = []
        connectors = include if include is not None else self.connectors

        if exclude:
            connectors = list(filter(lambda c: c not in exclude, connectors))
        for connector in connectors:
            result = connector.process_event(event, *args, **kwargs)
            if result is not None:
                if first:
                    return result
                results.append(result)

        return results


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
        ServoSettings, routes = _create_settings_model(config_file=config_file, env=env)

        # Build our Servo settings instance from the config file + environment
        if config_file.exists():
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            if not (config is None or isinstance(config, dict)):
                raise ValueError(
                    f'error: config file "{config_file}" parsed to an unexpected value of type "{config.__class__}"'
                )
            config = {} if config is None else config
            servo_settings = ServoSettings.parse_obj(config)
        else:
            # If we do not have a config file, build a minimal configuration
            # NOTE: This configuration is likely incomplete/invalid due to required
            # settings on the connectors not being fully configured
            args = kwargs.copy()
            for key_path, connector_type in routes.items():
                args[key_path] = connector_type.settings_model().construct()
            servo_settings = ServoSettings.construct(**args)

        # Initialize all active connectors
        connectors: List[Connector] = []
        for key_path, connector_type in routes.items():
            connector_settings = getattr(servo_settings, key_path)
            if connector_settings:
                # NOTE: If the command is routed but doesn't define a settings class this will raise
                connector = connector_type(settings=connector_settings, optimizer=optimizer)
                connectors.append(connector)

        # Build the servo object
        servo = Servo(settings=servo_settings, connectors=connectors, optimizer=optimizer)
        assembly = ServoAssembly(
            config_file=config_file,
            optimizer=optimizer,
            settings_model=ServoSettings,
            servo=servo,
        )

        return assembly, servo, ServoSettings

    ##
    # Utility functions

    @classmethod
    def default_routes(cls) -> Dict[str, Type[Connector]]:
        routes = {}
        for connector in cls.all_connectors():
            mounts[connector.__key_path__] = connector
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

def _create_settings_model_from_routes(
    routes = Dict[str, Type[Connector]],
    *,
    require_fields:bool = True,
) -> Type[BaseServoSettings]:
    # Create Pydantic fields for each active route
    connector_versions: Dict[
        Type[Connector], str
    ] = {}  # use dict for uniquing and ordering
    setting_fields: Dict[str, Tuple[Type[ConnectorSettings], Any]] = {}
    default_value = (
        ... if require_fields else None
    )  # Pydantic uses ... for flagging fields required

    for key_path, connector_class in routes.items():
        settings_model = _derive_settings_model_for_route(key_path, connector_class)
        settings_model.__config__.title = (
            f"{connector_class.name} Settings (at key-path {key_path})"
        )
        setting_fields[key_path] = (settings_model, default_value)
        connector_versions[
            connector_class
        ] = f"{connector_class.name} v{connector_class.version}"

    # Create our model
    servo_settings_model = create_model(
        "ServoSettings", __base__=BaseServoSettings, **setting_fields,
    )

    connectors_series = join_to_series(list(connector_versions.values()))
    servo_settings_model.__config__.title = "Servo Configuration Schema"
    servo_settings_model.__config__.schema_extra = {
        "description": f"Schema for configuration of Servo v{Servo.version} with {connectors_series}"
    }

    return servo_settings_model

def _create_settings_model(
    *, config_file: Path, env: Optional[Dict[str, str]] = os.environ
) -> (Type[BaseServoSettings], Dict[str, Type[Connector]]):
    # map of config key in YAML to settings class for target connector
    routes: Dict[str, Type[Connector]] = _default_routes()
    require_fields: bool = False

    # NOTE: If `connectors` key is present in config file, require the keys to be present
    if config_file.exists():
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        if isinstance(config, dict):  # Config file could be blank or malformed
            connectors_value = config.get("connectors", None)
            if connectors_value:
                routes = _routes_for_connectors_descriptor(connectors_value)
                require_fields = True

    servo_settings_model = _create_settings_model_from_routes(routes, require_fields=require_fields)    
    return servo_settings_model, routes


def _normalize_name(name: str) -> str:
    """
    Normalizes the given name.
    """
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", name)


def _derive_settings_model_for_route(
    key_path: str, model: Type[Connector]
) -> Type[ConnectorSettings]:
    """Inherits a new Pydantic model from the given settings and set up nested environment variables"""
    # NOTE: It is important to produce a new model name to disambiguate the models within Pydantic
    # because key-paths are guanranteed to be unique, we can utilize it as a
    base_setting_model = model.settings_model()

    if base_setting_model == ConnectorSettings:
        # Connector hasn't defined a settings class, use the connector class name as base name
        # This is essential for preventing `KeyError` exceptions in Pydantic schema generation
        # due to naming conflicts.
        model_name = f"{model.__name__}Settings"
    elif key_path == model.__key_path__:
        # Connector is mounted at the default route, use default name
        model_name = f"{base_setting_model.__qualname__}"
    else:
        model_name = _normalize_name(f"{base_setting_model.__qualname__}__{key_path}")

    # TODO: Check if the name has a conflict
    settings_model = create_model(model_name, __base__=base_setting_model,)

    # Traverse across all the fields and update the env vars
    for name, field in settings_model.__fields__.items():
        field.field_info.extra.pop("env", None)
        field.field_info.extra["env_names"] = {f"SERVO_{key_path}_{name}".upper()}

    return settings_model


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
    for connector_class in ServoAssembly.all_connectors():
        if connector == connector_class.__key_path__ or connector in [
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
                raise ValueError(f"Connector descriptor keys must be strings (invalid value '{key}'")
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
                connector_routes[connector.__key_path__] = connector
            elif connector_class := _connector_class_from_string(connector):
                connector_routes[connector_class.__key_path__] = connector_class
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
                Connector.validate_config_key_path(config_path)
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


RESERVED_KEYS = ["connectors", "control", "measure", "adjust"]


def _reserved_keys() -> List[str]:
    reserved_keys = list(_default_routes().keys())
    reserved_keys.extend(RESERVED_KEYS)
    return reserved_keys


def _default_routes() -> Dict[str, Type[Connector]]:
    routes = {}
    for connector in Connector.all():
        if connector is not Servo:
            routes[connector.__key_path__] = connector
    return routes
