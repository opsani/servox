import abc
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_type_hints, Union, Set
import importlib

import durationpy
import httpx
import semver
import typer
import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    FilePath,
    HttpUrl,
    ValidationError,
    constr,
    root_validator,
    validator,
)
from pydantic.schema import schema as pydantic_schema
from pydantic.json import pydantic_encoder


#####

from pkg_resources import EntryPoint, iter_entry_points
from typing import Generator

CONNECTORS_GROUP = "servo.connectors"

class ConnectorLoader:
    def __init__(self, group: str = CONNECTORS_GROUP) -> None:
        self.group = group

    def iter_entry_points(self) -> Generator[EntryPoint, None, None]:
        yield from iter_entry_points(group=self.group, name=None)

    def load(self) -> Generator[Any, None, None]:
        for entry_point in self.iter_entry_points():
            yield entry_point.resolve()

#####

class Optimizer(BaseModel):
    org_domain: constr(
        regex=r"(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))"
    )
    app_name: constr(regex=r"^[a-z\-]{6,32}$")
    token: str
    base_url: HttpUrl = "https://api.opsani.com/"

    def __init__(self, id: str = None, token: str = None, **kwargs):
        org_domain = kwargs.pop("org_domain", None)
        app_name = kwargs.pop("app_name", None)
        if id:
            org_domain, app_name = id.split("/")
        super().__init__(
            org_domain=org_domain, app_name=app_name, token=token, **kwargs
        )

    def id(self) -> str:
        """Returns the optimizer identifier"""
        return f"{self.org_domain}/{self.app_name}"


class Settings(BaseSettings):
    description: Optional[str]

    # Optimizer we are communicating with
    _optimizer: Optimizer

    class Config:
        env_prefix = "SERVO_"
        extra = Extra.forbid


class License(Enum):
    """Defined licenses"""

    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self):
        return self.value


class Maturity(Enum):
    """Connector maturity level"""

    EXPERIMENTAL = "Experimental"
    STABLE = "Stable"
    ROBUST = "Robust"

    @classmethod
    def from_str(cls, identifier: str) -> "Maturity":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No maturity level identified by "{identifier}".')

    def __str__(self):
        return self.value


class Version(semver.VersionInfo):
    pass

CONNECTORS_GROUP = "servo.connectors"
class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """

    # Global registry of all available connectors
    __subclasses: ClassVar[List["Connector"]] = []

    # Connector metadata
    name: ClassVar[str] = None
    version: ClassVar[Version] = None
    description: ClassVar[Optional[str]] = None
    homepage: ClassVar[Optional[HttpUrl]] = None
    license: ClassVar[Optional[License]] = None
    maturity: ClassVar[Optional[Maturity]] = None

    # Instance configuration
    id: str
    settings: Settings
    _logger: logger

    @staticmethod
    def discover() -> Generator[Any, None, None]:
        '''Discover connectors available in the assembl and yield them for configuration'''
        loader = ConnectorLoader()
        yield from loader.load()

    @classmethod
    def all(cls) -> List["Connector"]:
        return cls.__subclasses

    @root_validator(pre=True)
    @classmethod
    def validate_required_metadata(cls, v):
        assert cls.name is not None, "name must be provided"
        assert cls.version is not None, "version must be provided"
        if isinstance(cls.version, str):
            # Attempt to parse
            cls.version = Version.parse(cls.version)
        assert isinstance(
            cls.version, (Version, semver.VersionInfo)
        ), "version is not a semantic versioning descriptor"
        return v

    @validator("id")
    @classmethod
    def key_format_is_valid(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/]{4,64}$", v)
        ), "keys may only contain alphanumeric characters, hyphens, slashes, and underscores"
        return v

    @classmethod
    def settings_class(cls) -> Type['Settings']:
        hints = get_type_hints(cls)
        settings_cls = hints["settings"]
        return settings_cls

    @classmethod
    def default_key(cls) -> str:
        name = cls.__name__.replace("Connector", "")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__.replace("Connector", " Connector")
        cls.version = semver.VersionInfo.parse("0.0.0")
        cls.__subclasses.append(cls)

    def __init__(self, settings: Settings, *, id: Optional[str] = None, **kwargs):
        id = id if id is not None else self.default_key()
        super().__init__(id=id, settings=settings, **kwargs)

    async def api_client(self) -> httpx.AsyncClient:
        """Yields an httpx.AsyncClient instance configured to talk to Opsani API"""
        async with httpx.AsyncClient() as client:
            yield client

    def logger(self) -> logger:
        """Returns the logger"""
        return self._logger

    def cli(self) -> Optional[typer.Typer]:
        """Returns a Typer CLI for the connector"""
        return None


def metadata(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[semver.VersionInfo] = None,
    homepage: Optional[HttpUrl] = None,
    license: Optional[License] = None,
    maturity: Optional[Maturity] = None,
):
    def decorator(cls):
        if name:
            cls.name = name
        if description:
            cls.description = description
        if version:
            cls.version = (
                version
                if isinstance(version, semver.VersionInfo)
                else Version.parse(version)
            )
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = license
        if maturity:
            cls.maturity = maturity
        return cls

    return decorator

def _module_path(cls: Type) -> str:
    return ".".join([cls.__module__, cls.__name__])

class ServoSettings(Settings):
    optimizer: Optimizer
    """The Opsani optimizer the Servo is attached to"""
    
    connectors: Optional[Dict[str, str]] = None

    @validator('connectors', pre=True)
    @classmethod
    def validate_connectors(cls, connectors) -> Dict[str, Type[Connector]]:
        def _validate_class(connector: type) -> bool:
            if not isinstance(connector, type):
                return False

            if not issubclass(connector, Connector):
                raise TypeError(f'{connector.__name__} is not a Connector subclass')

            return True

        def _validate_string(connector: str) -> Optional[Type[Connector]]:
            if not isinstance(connector, str):
                return None

            # Check fo an existing class in the namespace
            connector_class = globals().get(connector, None)
            try:
                connector_class = eval(connector) if connector_class is None else connector_class
            except Exception:
                pass

            if _validate_class(connector_class):
                return connector_class

            # Check if the string is an identifier for a connector
            for connector_class in Servo.all_connectors():
                if connector == connector_class.default_key():
                    return connector_class
            
            # Try to load it as a module path
            if '.' in connector:
                module_path, class_name = e.split(':', 2)
                module = importlib.import_module(module_path)
                connector_class = getattr(module, class_name)
                if _validate_class(connector_class):
                    return connector_class

            raise TypeError(f'{connector} does not identify a Connector class')
        
        # Process our input appropriately
        if connectors is None:
            # None indicates that all available connectors should be activated
            return None
        elif isinstance(connectors, str):
            # NOTE: Special case. When we are invoked with a string it is typically an env var
            try:
                decoded_value = cls.__config__.json_loads(connectors)  # type: ignore
            except ValueError as e:
                raise ValueError(f'error parsing JSON for "{connectors}"') from e

            # Prevent infinite recursion
            if isinstance(decoded_value, str):
                raise ValueError(f'JSON string values for `connectors` cannot parse into strings: "{connectors}"')

            return cls.validate_connectors(decoded_value)

        elif isinstance(connectors, (list, tuple, set)):
            connector_mounts: Dict[str, str] = {}
            for connector in connectors:
                if _validate_class(connector):
                    connector_mounts[connector.default_key()] = _module_path(connector)
                elif connector_class := _validate_string(connector):
                    connector_mounts[connector_class.default_key()] = _module_path(connector_class)
                else:
                    raise ValueError(f"Missing validation for value {connector}")
            
            return connector_mounts

        elif isinstance(connectors, dict):
            connector_map = Servo.default_mounts()
            reserved_keys = Servo.reserved_keys()            

            connector_mounts = {}
            for key, value in connectors.items():
                if not isinstance(key, str):
                    raise ValueError(f'Key "{key}" is not a string')                
                
                # Validate the key format
                try:
                    Connector.key_format_is_valid(key)
                except AssertionError as e:
                    raise ValueError(f'Key "{key}" is not valid: {e}') from e
                
                # Resolve the connector class
                if isinstance(value, type):
                    connector_class = value
                elif isinstance(value, str):
                    connector_class = _validate_string(value)

                # Check for key reservations
                if key in reserved_keys:
                    if c := connector_map.get(key, None):
                        if connector_class != c:
                            raise ValueError(f'Key "{key}" is reserved by `{c.__name__}`')
                    else:
                        raise ValueError(f'Key "{key}" is reserved')
                
                connector_mounts[key] = _module_path(connector_class)
            
            return connector_mounts

        else:
            raise ValueError(f'Unexpected type `{type(connectors).__qualname__}`` encountered (connectors: {connectors})')

    class Config:
        # We are the base root of pluggable configuration
        # so we ignore any extra fields so you can turn connectors on and off
        extra = Extra.ignore

@metadata(
    description="Continuous Optimization Orchestrator",
    homepage="https://opsani.com/",
    maturity=Maturity.ROBUST,
    license=License.APACHE2,
)
class Servo(Connector):
    """The Servo"""

    settings: ServoSettings    
    connectors: Dict[str, Type[Connector]] = {}

    @classmethod
    def default_mounts(cls) -> Dict[str, Type[Connector]]:
        mounts = {}
        for connector in Servo.all_connectors():
            mounts[connector.default_key()] = connector
        return mounts
    
    @classmethod
    def reserved_keys(cls) -> List[str]:
        reserved_keys = list(cls.default_mounts().keys())
        reserved_keys.append('connectors')
        return reserved_keys

    @classmethod
    def all_connectors(cls) -> List[Connector]:
        connectors = []
        for c in Connector.all():
            if c == cls:
                continue
            connectors.append(c)
        return connectors
    
    def active_connectors(self) -> List[Connector]:
        """Return connectors explicitly activated in the configuration"""
        return self.connectors.values()

    def top_level_schema(self, *, all: bool = False) -> Dict[str, Any]:
        '''Returns a schema that only includes connector model definitions'''
        connectors = self.all_connectors() if all else self.active_connectors()
        settings_classes = list(map(lambda c: c.settings_class(), connectors))
        return pydantic_schema(settings_classes, title="Servo Schema")
    
    def top_level_schema_json(self, *, all: bool = False) -> str:
        '''Return a JSON string representation of the top level schema'''
        return json.dumps(self.top_level_schema(all=all), indent=2, default=pydantic_encoder)

    ##
    # Connector management

    def add_connector(self, conn: "Connector") -> None:
        self.connectors.append(conn)

    def remove_connector(self, conn: "Connector") -> None:
        pass

    def load_connectors(self) -> None:
        pass

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


###
### Vegeta


class TargetFormat(str, Enum):
    http = "http"
    json = "json"

    def __str__(self):
        return self.value


class VegetaSettings(Settings):
    """
    Configuration of the Vegeta connector
    """

    rate: str = Field(
        description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.",
    )
    duration: str = Field(
        description="Specifies the amount of time to issue requests to the targets.",
    )
    format: TargetFormat = Field(
        "http",
        description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.",
    )
    target: Optional[str] = Field(
        description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin."
    )
    targets: Optional[FilePath] = Field(
        description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk."
    )
    connections: int = Field(
        10000,
        description="Specifies the maximum number of idle open connections per target host.",
    )
    workers: int = Field(
        10,
        description="Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.",
    )
    max_workers: int = Field(
        18446744073709551615,
        alias="max-workers",
        description="The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.",
        env="",
    )
    max_body: int = Field(
        -1,
        alias="max-body",
        description="Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.",
        env="",
    )
    http2: bool = Field(
        True,
        description="Specifies whether to enable HTTP/2 requests to servers which support it.",
    )
    keepalive: bool = Field(
        True,
        description="Specifies whether to reuse TCP connections between HTTP requests.",
    )
    insecure: bool = Field(
        False,
        description="Specifies whether to ignore invalid server TLS certificates.",
    )

    @root_validator()
    @classmethod
    def validate_target(cls, values):
        target, targets = values.get("target"), values.get("targets")
        if target is None and targets is None:
            raise ValueError("target or targets must be configured")

        if target is not None and targets is not None:
            raise ValueError("target and targets cannot both be configured")

        return values

    @root_validator()
    @classmethod
    def validate_target_format(cls, values):
        target, targets = values.get("target"), values.get("targets")

        # Validate JSON target formats
        if target is not None and values.get("format") == TargetFormat.json:
            try:
                json.loads(target)
            except Exception as e:
                raise ValueError("the target is not valid JSON") from e

        if targets is not None and values.get("format") == TargetFormat.json:
            try:
                json.load(open(targets))
            except Exception as e:
                raise ValueError("the targets file is not valid JSON") from e

        # TODO: Add validation of JSON with JSON Schema (https://github.com/tsenart/vegeta/blob/master/lib/target.schema.json)
        # and HTTP format
        return values

    @validator("rate")
    @classmethod
    def validate_rate(cls, v):
        assert isinstance(
            v, (int, str)
        ), "rate must be an integer or a rate descriptor string"

        # Integer rates
        if isinstance(v, int) or v.isnumeric():
            return str(v)

        # Check for hits/interval
        components = v.split("/")
        assert len(components) == 2, "rate strings are of the form hits/interval"

        hits = components[0]
        duration = components[1]
        assert hits.isnumeric(), "rate must have an integer hits component"

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(duration)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    @validator("duration")
    @classmethod
    def validate_duration(cls, v):
        assert isinstance(
            v, (int, str)
        ), "duration must be an integer or a duration descriptor string"

        if v == "0" or v == 0:
            return v

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(v)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    class Config:
        json_encoders = {TargetFormat: lambda t: t.value()}


class ConnectorCLI(typer.Typer):
    connector: Connector

    def __init__(self, connector: Connector, **kwargs):
        self.connector = connector
        name = kwargs.pop("name", connector.id)
        help = kwargs.pop("help", connector.description)
        add_completion = kwargs.pop("add_completion", False)
        super().__init__(name=name, help=help, add_completion=add_completion, **kwargs)
        self.add_commands()

    def add_commands(self):
        @self.command()
        def schema():
            """
            Display the schema 
            """
            # TODO: Support output formats (dict, json, yaml)...
            typer.echo(self.connector.settings.schema_json(indent=2))

        @self.command()
        def generate():
            """Generate a configuration file"""
            # TODO: support output paths/formats
            # NOTE: We have to serialize through JSON first
            schema = json.loads(json.dumps(self.connector.settings.dict(by_alias=True)))
            output_path = Path.cwd() / f"{self.connector.id}.yaml"
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated {self.connector.id}.yaml")

        @self.command()
        def validate(file: typer.FileText = typer.Argument(...), key: str = ""):
            """
            Validate given file against the JSON Schema
            """
            try:
                config = yaml.load(file, Loader=yaml.FullLoader)
                connector_config = config[key] if key != "" else config
                cls = type(self.connector.settings)
                config = cls.parse_obj(connector_config)
                typer.echo("√ Valid connector configuration")
            except (ValidationError, yaml.scanner.ScannerError) as e:
                typer.echo("X Invalid connector configuration", err=True)
                typer.echo(e, err=True)
                raise typer.Exit(1)

        @self.command()
        def info():
            """
            Display connector info
            """
            typer.echo(
                (
                    f"{self.connector.name} v{self.connector.version} ({self.connector.maturity})\n"
                    f"{self.connector.description}\n"
                    f"{self.connector.homepage}\n"
                    f"Licensed under the terms of {self.connector.license}\n"
                )
            )

        @self.command()
        def version():
            """
            Display version
            """
            typer.echo(f"{self.connector.name} v{self.connector.version}")


@metadata(
    description="Vegeta load testing connector",
    version="0.5.0",
    homepage="https://github.com/opsani/vegeta-connector",
    license=License.APACHE2,
    maturity=Maturity.STABLE,
)
class VegetaConnector(Connector):
    settings: VegetaSettings

    # TODO: Measure

    def cli(self) -> ConnectorCLI:
        """Returns a Typer CLI for interacting with this connector"""
        cli = ConnectorCLI(self, help="Load generation with Vegeta")

        @cli.command()
        def loadgen():
            """
            Run an adhoc load generation
            """

        return cli
    
    # TODO: Message handlers...
    # Model the metrics

    def measure(self):
        pass

    def describe(self):
        pass

class MeasureConnector(Connector):
    pass

class AdjustConnector(Connector):
    pass
