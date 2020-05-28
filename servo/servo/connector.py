import typer
import pydantic
from pydantic import BaseModel, Field, ValidationError, Extra, BaseSettings, create_model
from pydantic.schema import schema
import abc
from typing import ClassVar, Any, Optional, ClassVar, List, Dict
from enum import Enum
import yaml
import pyaml
import json

# TODO: Base config for hyphenate...
class ConnectorConfig(BaseSettings):
    class Config:
        # TODO: Can I use reflection?
        env_prefix = 'SERVO_'

class ServoConfig(ConnectorConfig):
    connectors: List[str] = []

class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting 
    external services and resources.
    """

    # Keys specify the connector, value is the config
    config_descriptor: Dict[str, dict] = None
    config_key: str = None # TODO: Infer from class name
    
    # TODO: All these need env var bindings
    app: str
    token: str
    base_url: str = "https://api.opsani.com/"    
    
    def handle_event(self, event: str):
        """
        Handle an event
        """

Connector.update_forward_refs()

class Servo(Connector):
    connectors: List['Connector'] = []

    def add_connector(self, conn: 'Connector') -> None:
        self.connectors.append(conn)
        
    def load_connectors(self):
        pass

# TODO: Vegeta specific
class Format(str, Enum):
    http = 'http'
    json = 'json'

class Config(ConnectorConfig):
    class Config:
        extra = Extra.forbid

    """
    Configuration of the Vegeta connector
    """
    # TODO: Validate the rate string (define regexp)
    rate: str = Field(description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.")
    # TODO: Validate the Golang duration string
    duration: str = Field(description="Specifies the amount of time to issue requests to the targets.")
    format: Format = Field('http', description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.")
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

class VegetaConnector(Connector):
    config: Config = Config(rate="50/s", duration="30s", target="GET http://localhost:8080")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_key = "vegeta"
    
    # TODO: Return a default config
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

    @classmethod
    def version(cls) -> str:
        """
        Return version connector
        """
        return "0.0.1"

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
app = typer.Typer(callback=callback, add_completion=False)
app.add_typer(vegeta_app, name="vegeta", help="Vegeta load generator")
app.add_typer(vegeta_app, name="kubernetes", help="Kubernetes orchestrator")
app.add_typer(vegeta_app, name="prometheus", help="Prometheus metrics")

config_descriptor = yaml.load(open("./servo.yaml"), Loader=yaml.FullLoader)
vegeta = VegetaConnector(config_descriptor=config_descriptor, app="dev.opsani.com/blake", token="sadasdsa")
servo = Servo(config=config_descriptor, app="dev.opsani.com/blake", token="sadasdsa")
servo.add_connector(vegeta)

@app.command()
def schema():
    """
    Display the JSON Schema 
    """
    # TODO: Read config file, find all loaded connectors, bundle into a schema...
    # What you probably have to do is 
    # print(Servo.schema_json(indent=2))
    from pydantic.schema import schema
    from pydantic.json import pydantic_encoder
    ServoModel = create_model(
    'ServoModel',
    servo=(ServoConfig, ...),
    vegeta=(Config, ...))
    print(ServoModel.schema_json(indent=2))
    # top_level_schema = schema([ServoConfig, Config], title='Servo Schema')
    # print(json.dumps(top_level_schema, indent=2, default=pydantic_encoder))

@app.command()
def validate(file: typer.FileText = typer.Argument(...)):
    """
    Validate given file against the JSON Schema
    """
    typer.echo("Validate" + str(file))

# TODO: Needs to take a list of connectors
# default to using all of them
@app.command()
def generate():
    """
    Generate a new config file
    """
    pass

# TODO: Does this need to be shared?
# Docker image?
@app.command()
def info():
    """
    Display assembly info
    """
    pass

@app.command()
def version():
    """
    Display version
    """
    pass

@app.command()
def run():
    """
    Start the servo
    """
    pass

# group = app.get_group()
# group.params.append(click_install_param)

if __name__ == "__main__":
    app()


# ---------------------------


class Metric:
    # name, unit
    pass

# Models a collection of metrics
class Metrics:
    pass

# Models a su
class ScalarMetric:
    pass

# Models a time-series metrics
class TimeSeriesMetric:
    pass

class Descriptor:
    pass

class Component:
    pass

# {"application": {"components": {"web": {"settings": {"cpu": {"value": 0.25, "min": 0.1, "max": 1.8, "step": 0.1, "type": "range"}, "replicas": {"value": 1, "min": 1, "max": 2, "step": 1, "type": "range"}}}}}, "measurement": {"metrics": {"requests_total": {"unit": "count"}, "throughput": {"unit": "rpm"}, "error_rate": {"unit": "percent"}, "latency_total": {"unit": "milliseconds"}, "latency_mean": {"unit": "milliseconds"}, "latency_50th": {"unit": "milliseconds"}, "latency_90th": {"unit": "milliseconds"}, "latency_95th": {"unit": "milliseconds"}, "latency_99th": {"unit": "milliseconds"}, "latency_max": {"unit": "milliseconds"}, "latency_min": {"unit": "milliseconds"}}}}
class Setting:
    # name, step, type, value, min, max
    pass