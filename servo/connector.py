import typer
import pydantic
from pydantic import BaseModel, Field, ValidationError, Extra, BaseSettings, create_model, HttpUrl
from pydantic.schema import schema
import abc
from typing import ClassVar, Any, Optional, ClassVar, List, Dict, Callable
from enum import Enum
from pathlib import Path
import yaml
import pyaml
import json
import semver
import httpx
from loguru import logger
from enum import Enum

# TODO: Handles example.com/app
# Add regex validation
class Optimizer(BaseModel):
    org_domain: str
    app_name: str
    token: str
    base_url: HttpUrl = "https://api.opsani.com/"

    # def as_url(self)
    # def as_str(self)

# will be from connector import settings
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

    def __str__(self):
        return self.value

class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """
    # Global registry of all available connectors
    __subclasses: ClassVar[List['Connector']] = []

    # Generic connector metadata
    name: ClassVar[str]
    description: ClassVar[Optional[str]] = None
    version: ClassVar[semver.VersionInfo]
    homepage: ClassVar[Optional[HttpUrl]] = None
    license: ClassVar[Optional[License]] = None
    maturity: ClassVar[Optional[Maturity]] = None

    # Instance configuration
    id: str # TODO: constraints should be lowercase, underscores. Infer id by transforming the name
    settings: ConnectorSettings
    _logger: logger

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__qualname__.replace('Connector', ' Connector')
        cls.__subclasses.append(cls)
    
    def __init__(self, settings: ConnectorSettings, *, id: Optional[str] = None, **kwargs):
        id = id if id is not None else type(self).__qualname__.replace('Connector', '').lower()
        super().__init__(id=id, settings=settings, **kwargs)
    
    async def api_client(self) -> httpx.AsyncClient:
        """Returns an httpx.AsyncClient instance configured to talk to Opsani API""" 
        async with httpx.AsyncClient() as client:
            yield client
    #   r = await client.get('https://www.example.org/')

    def logger(self) -> logger:
        """Returns the logger"""
        return self._logger
    
    def cli(self) -> Optional[typer.Typer]:
        '''Returns a Typer CLI for the connector'''
        pass

Connector.update_forward_refs()

# TODO: becomes from servo.connector import Settings (or BaseSettings?)
class ServoSettings(ConnectorSettings):
    connectors: List[str] = []

# TODO: init with an Optimizer
class Servo(Connector):
    '''The Servo'''

    optimizer: Optimizer
    '''The Opsani optimizer the Servo is attached to'''

    connectors: List['Connector'] = []

    def add_connector(self, conn: 'Connector') -> None:
        self.connectors.append(conn)

    def remove_connector(self, conn: 'Connector') -> None:
        pass
        
    def load_connectors(self) -> None:
        pass
    
    def send_event(self, event: str, payload: Dict[str, Any]) -> None:
        '''Handle an event'''

    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        '''Handle an event'''
    
    def run(self) -> None:
        pass

# TODO: Vegeta specific
class TargetFormat(str, Enum):
    http = 'http'
    json = 'json'

class VegetaSettings(ConnectorSettings):
    """
    Configuration of the Vegeta connector
    """
    # TODO: Validate the rate string (define regexp)
    rate: str = Field(description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.")
    # TODO: Validate the Golang duration string
    duration: str = Field(description="Specifies the amount of time to issue requests to the targets.")
    format: TargetFormat = Field('http', description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.")
    # TODO: Validate the JSON or HTTP format
    target: str = Field(description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin.")
    # TODO: Should be a file
    targets: str = Field("stdin", description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk.")    
    connections: int = Field(10000, description="Specifies the maximum number of idle open connections per target host.")
    workers: int = Field(10, description="Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.")
    max_workers: int = Field(18446744073709551615, alias="max-workers", description="The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.")
    max_body: int = Field(-1, alias="max-body", description="Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.")
    http2: bool = Field(True, description="Specifies whether to enable HTTP/2 requests to servers which support it.")
    keepalive: bool = Field(True, description="Specifies whether to reuse TCP connections between HTTP requests.")
    insecure: bool = Field(False, description="Specifies whether to ignore invalid server TLS certificates.")

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
            cls.version = version
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
    # TODO: Make this a class method? Probably the same with schema...
    @classmethod
    def generate(cls):
        """
        Generate a new default configuration
        """
        pass
    
    # TODO: Not sure if I need schema...
    @classmethod
    def validate(cls, data) -> bool:
        """
        Validate configuration
        """
        return True
    
    # TODO: This may be a class method?
    def cli(self) -> typer.Typer:
        '''Returns a Typer CLI for interacting with this connector'''
        cli = typer.Typer(name=self.id, add_completion=False)

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

        # TODO: file option + key
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
            # pyaml.p({ key: connector_config})

        # TODO: Does this need to be shared?
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
            # Init connector based on input, fire measure
            pass

        return cli

vegeta_app = typer.Typer()
# TODO: We need the basic flags and options

@vegeta_app.command()
def schema():
    """
    Display the JSON Schema 
    """
    print(Config.schema_json(indent=2))

# TODO: file option + key
@vegeta_app.command()
def validate(file: typer.FileText = typer.Argument(...), key: str = ""):
    """
    Validate given file against the JSON Schema
    """
    config = yaml.load(file, Loader=yaml.FullLoader)
    connector_config = config[key] if key != "" else config
    try:
        config = Config.parse_obj(connector_config)
        typer.echo("√ Valid connector configuration")
    except ValidationError as e:
        typer.echo("X Invalid connector configuration")
        print(e)
    pyaml.p({ key: connector_config})

# TODO: Does this need to be shared?
@vegeta_app.command()
def info():
    """
    Display assembly info
    """
    pass

@vegeta_app.command()
def version():
    """
    Display version
    """
    pass

@vegeta_app.command()
def measure():
    """
    Run a measure cycle
    """
    # Init connector based on input, fire measure
    pass

# Use callback to define top-level options
# TODO: Need a way to intelligently opt in or out of this. Maybe a new decorator
def callback(app: str = typer.Option(..., help="Opsani app (format is example.com/app)"), 
             token: str = typer.Option(..., help="Opsani API access token"), 
             base_url: str = typer.Option("http://api.opsani.com/", help="Base URL for connecting to Opsani API")):
    pass
    # TODO: Need to figure out how to pack these values onto a context

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
# # default to using all of them
# @app.command()
# def generate():
#     """
#     Generate a new config file
#     """
#     pass

# # TODO: Does this need to be shared?
# # Docker image?
# @app.command()
# def info():
#     """
#     Display assembly info
#     """
#     pass

# @app.command()
# def version():
#     """
#     Display version
#     """
#     pass

# @app.command()
# def run():
#     """
#     Start the servo
#     """
#     pass

# # group = app.get_group()
# # group.params.append(click_install_param)

# if __name__ == "__main__":
#     app()
