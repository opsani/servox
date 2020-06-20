import typer
import pydantic
import re
from pydantic import BaseModel, Field, ValidationError, Extra, BaseSettings, create_model, HttpUrl, validator, root_validator, FilePath, constr
from pydantic.validators import str_validator
from pydantic.schema import schema
import abc
from typing import ClassVar, Any, Optional, ClassVar, List, Dict, Callable, Union, Literal
from enum import Enum
from pathlib import Path
import yaml
import pyaml
import durationpy
import json
import semver
import httpx
from loguru import logger
from enum import Enum

class Optimizer(BaseModel):
    org_domain: constr(regex=r'(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))')
    app_name: constr(regex=r'^[a-z\-]{6,32}$')
    token: str
    base_url: HttpUrl = "https://api.opsani.com/"

    def __init__(
        self, 
        id: str, 
        token: str,
        **kwargs
    ):
        org_domain, app_name = id.split('/')
        super().__init__(org_domain=org_domain, app_name=app_name, token=token, **kwargs)

# TODO: will be from connector import settings
class ConnectorSettings(BaseSettings):
    description: Optional[str]

    # Optimizer we are communicating with
    _optimizer: Optimizer

    class Config:
        # TODO: Figure out how to uppercase the keys
        env_prefix = 'SERVO_'
        extra = Extra.forbid

class License(Enum):
    """Defined licenses"""
    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> 'License':
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
    def from_str(cls, identifier: str) -> 'Maturity':
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

class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """
    # Global registry of all available connectors
    __subclasses: ClassVar[List['Connector']] = []

    # Connector metadata
    name: ClassVar[str] = None
    version: ClassVar[Version] = None
    description: ClassVar[Optional[str]] = None    
    homepage: ClassVar[Optional[HttpUrl]] = None
    license: ClassVar[Optional[License]] = None
    maturity: ClassVar[Optional[Maturity]] = None

    # Instance configuration
    id: str
    settings: ConnectorSettings
    _logger: logger

    @classmethod
    def all(cls) -> List['Connector']:
        return cls.__subclasses

    @root_validator(pre=True)
    @classmethod
    def validate_required_metadata(cls, v):
        assert cls.name is not None, 'name must be provided'
        assert cls.version is not None, 'version must be provided'
        if isinstance(cls.version, str):
            # Attempt to parse
            cls.version = Version.parse(cls.version)
        assert isinstance(cls.version, (Version, semver.VersionInfo)), 'version is not a semantic versioning descriptor'
        return v

    @validator('id')
    @classmethod
    def id_format_is_valid(cls, v):        
        assert bool(re.match("^[0-9a-z_]{4,16}$", v)), 'id may only contain lowercase alphanumeric characters and underscores'
        return v

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__qualname__.replace('Connector', ' Connector')
        cls.version = semver.VersionInfo.parse("0.0.0")
        cls.__subclasses.append(cls)
    
    def __init__(
        self, 
        settings: ConnectorSettings, 
        *,
        id: Optional[str] = None, 
        **kwargs
    ):
        id = id if id is not None else type(self).__qualname__.replace('Connector', '').lower()
        super().__init__(id=id, settings=settings, **kwargs)
    
    async def api_client(self) -> httpx.AsyncClient:
        """Yields an httpx.AsyncClient instance configured to talk to Opsani API""" 
        async with httpx.AsyncClient() as client:
            yield client

    def logger(self) -> logger:
        """Returns the logger"""
        return self._logger
    
    def cli(self) -> Optional[typer.Typer]:
        '''Returns a Typer CLI for the connector'''
        return None

Connector.update_forward_refs()

# TODO: becomes from servo.connector import Settings (or BaseSettings?)
# TODO: needs to support env vars, loading from file
# TODO: The optimizer probably just folds in here
class ServoSettings(ConnectorSettings):
    connectors: List[str] = []

class Servo(Connector):
    '''The Servo'''

    optimizer: Optimizer # TODO: Replace with settings
    '''The Opsani optimizer the Servo is attached to'''

    connectors: List['Connector'] = []

    def __init__(
        self, 
        optimizer: Optimizer, 
        *,
        id: Optional[str] = None, 
        **kwargs
    ):
        settings = ServoSettings()
        super().__init__(settings=settings, optimizer=optimizer, **kwargs)
        self.optimizer = optimizer
    
    ##
    # Connector management

    def add_connector(self, conn: 'Connector') -> None:
        self.connectors.append(conn)

    def remove_connector(self, conn: 'Connector') -> None:
        pass
        
    def load_connectors(self) -> None:
        pass

    ##
    # Event processing
    
    def send_event(self, event: str, payload: Dict[str, Any]) -> None:
        '''Dispatch an event'''

    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        '''Handle an event'''
    
    ##
    # Lifecycle

    def run(self) -> None:
        pass
    
    ##
    # Misc

    def cli(self) -> typer.Typer:
        # TODO: Get the root CLI and then nest all active connectors
        pass

###
### Vegeta

class TargetFormat(str, Enum):
    http = 'http'
    json = 'json'

    def __str__(self):
        return self.value

class VegetaSettings(ConnectorSettings):
    """
    Configuration of the Vegeta connector
    """
    rate: str = Field(description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.")
    duration: str = Field(description="Specifies the amount of time to issue requests to the targets.")
    format: TargetFormat = Field('http', description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.")
    target: Optional[str] = Field(description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin.")
    targets: Optional[FilePath] = Field(description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk.")
    connections: int = Field(10000, description="Specifies the maximum number of idle open connections per target host.")
    workers: int = Field(10, description="Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.")
    max_workers: int = Field(18446744073709551615, alias="max-workers", description="The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.")
    max_body: int = Field(-1, alias="max-body", description="Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.")
    http2: bool = Field(True, description="Specifies whether to enable HTTP/2 requests to servers which support it.")
    keepalive: bool = Field(True, description="Specifies whether to reuse TCP connections between HTTP requests.")
    insecure: bool = Field(False, description="Specifies whether to ignore invalid server TLS certificates.")

    @root_validator(pre=True)
    @classmethod
    def validate_target(cls, values):
        target, targets = values.get('target'), values.get('targets')
        if target is None and targets is None:
            raise ValueError('target or targets must be configured')

        if target is not None and targets is not None:
            raise ValueError('target and targets cannot both be configured')

        return values
    
    @root_validator()
    @classmethod
    def validate_target_format(cls, values):
        target, targets = values.get('target'), values.get('targets')

        # Validate JSON target formats
        if target is not None and values.get('format') == TargetFormat.json:
            try:
                json.loads(target)
            except Exception as e:
                raise ValueError("the target is not valid JSON") from e
        
        if targets is not None and values.get('format') == TargetFormat.json:
            try:
                json.load(open(targets))
            except Exception as e:
                raise ValueError("the targets file is not valid JSON") from e
        
        # TODO: Add validation of JSON with JSON Schema (https://github.com/tsenart/vegeta/blob/master/lib/target.schema.json)
        # and HTTP format
        return values

    @validator('rate')
    @classmethod
    def validate_rate(cls, v):
        assert isinstance(v, (int, str)), "rate must be an integer or a rate descriptor string"

        # Integer rates
        if isinstance(v, int) or v.isnumeric():
            return str(v)

        # Check for hits/interval
        components = v.split('/')
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
    
    @validator('duration')
    @classmethod
    def validate_duration(cls, v):
        assert isinstance(v, (int, str)), "duration must be an integer or a duration descriptor string"

        if v == '0' or v == 0:
            return v

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(v)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    class Config:
        json_encoders = {
            TargetFormat: lambda t: t.value()
        }

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
            cls.version = version if isinstance(version, semver.VersionInfo) else Version.parse(version)
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = license
        if maturity:
            cls.maturity = maturity
        return cls
    return decorator

# TODO: AdjustMixin, MeasureMixin??

@metadata(
    description='Vegeta load testing connector',
    version='0.5.0',
    homepage='https://github.com/opsani/vegeta-connector',
    license=License.APACHE2,
    maturity=Maturity.STABLE
)
class VegetaConnector(Connector):
    
    def cli(self) -> typer.Typer:
        '''Returns a Typer CLI for interacting with this connector'''
        cli = typer.Typer(name=self.id, help="Vegeta load generator", add_completion=False)

        @cli.command()
        def schema():
            """
            Display the schema 
            """
            # TODO: Support output formats (dict, json, yaml)...
            typer.echo(self.settings.schema_json(indent=2))

        @cli.command()
        def generate():
            '''Generate a configuration file'''
            # TODO: support output paths/formats
            # NOTE: We have to serialize through JSON first
            schema = json.loads(json.dumps(self.settings.dict(by_alias=True)))
            output_path = Path.cwd() / f'{self.id}.yaml'
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated {self.id}.yaml")

        @cli.command()
        def validate(file: typer.FileText = typer.Argument(...), key: str = ""):
            """
            Validate given file against the JSON Schema
            """
            try:
                config = yaml.load(file, Loader=yaml.FullLoader)
                connector_config = config[key] if key != "" else config
                cls = type(self.settings)
                config = cls.parse_obj(connector_config)
                typer.echo("√ Valid connector configuration")
            except (ValidationError, yaml.scanner.ScannerError) as e:
                typer.echo("X Invalid connector configuration", err=True)
                typer.echo(e, err=True)
                raise typer.Exit(1)

        @cli.command()
        def info():
            """
            Display assembly info
            """
            typer.echo((
                f"{self.name} v{self.version} ({self.maturity})\n"
                f"{self.description}\n"
                f"{self.homepage}\n"
                f"Licensed under the terms of {self.license}\n"
            ))

        @cli.command()
        def version():
            """
            Display version
            """
            typer.echo(f'{self.name} v{self.version}')

        @cli.command()
        def loadgen():
            """
            Run an adhoc load generation
            """
            pass

        return cli

# TODO: Moves to cli.py
# app = typer.Typer(callback=callback, add_completion=False)
# app.add_typer(vegeta_app, name="vegeta", help="Vegeta load generator")
# app.add_typer(vegeta_app, name="kubernetes", help="Kubernetes orchestrator")
# app.add_typer(vegeta_app, name="prometheus", help="Prometheus metrics")

# config_descriptor = yaml.load(open("./servo.yaml"), Loader=yaml.FullLoader)
# vegeta = VegetaConnector(config_descriptor=config_descriptor, app="dev.opsani.com/blake", token="sadasdsa")
# servo = Servo(config=config_descriptor, app="dev.opsani.com/blake", token="sadasdsa")
# servo.add_connector(vegeta)

# @app.command()
# def schema():
#     """
#     Display the JSON Schema 
#     """
#     # TODO: Read config file, find all loaded connectors, bundle into a schema...
#     # What you probably have to do is 
#     # print(Servo.schema_json(indent=2))
#     from pydantic.schema import schema
#     from pydantic.json import pydantic_encoder
#     ServoModel = create_model(
#     'ServoModel',
#     servo=(ServoConfig, ...),
#     vegeta=(Config, ...))
#     print(ServoModel.schema_json(indent=2))
#     # top_level_schema = schema([ServoConfig, Config], title='Servo Schema')
#     # print(json.dumps(top_level_schema, indent=2, default=pydantic_encoder))

# @app.command()
# def validate(file: typer.FileText = typer.Argument(...)):
#     """
#     Validate given file against the JSON Schema
#     """
#     ServoModel = create_model(
#     'ServoModel',
#     servo=(ServoConfig, ...),
#     vegeta=(Config, ...))
#     try:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#         config_descriptor = ServoModel.parse_obj(config)
#         typer.echo("√ Valid servo configuration")
#     except ValidationError as e:
#         typer.echo("X Invalid servo configuration")
#         print(e)
#     pyaml.p(config)

# # TODO: Needs to take a list of connectors
