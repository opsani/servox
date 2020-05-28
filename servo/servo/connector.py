import typer
from pydantic import BaseModel, Field, ValidationError, Extra
import abc
from typing import ClassVar, Any, Optional, ClassVar, List
from enum import Enum
import yaml
import pyaml

class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting 
    external services and resources.
    """
    config_file: str
    config_key: str = None
    app: str
    token: str
    base_url: str = "https://api.opsani.com/"
    connectors: List['Connector'] = []

    def add_connector(self, conn: 'Connector') -> None:
        self.connectors.append(conn)

Connector.update_forward_refs()

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

class Servo(Connector):
    pass

# TODO: Vegeta specific
class Format(str, Enum):
    http = 'http'
    json = 'json'

class Config(BaseModel):
    class Config:
        extra = Extra.forbid

    """
    Configuration of the Vegeta connector
    """
    # TODO: Validate the rate string (define regexp)
    rate: str = Field(description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.")
    # TODO: Validate the Golang duration string
    duration: str = Field(description="Specifies the amount of time to issue requests to the targets.")
    # TODO: Validate the JSON or HTTP format
    target: str = Field(description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin.")
    # TODO: Should be a file
    targets: str = Field("stdin", description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk.")
    format: Format = Field('http', description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.")
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

vegeta_app = typer.Typer()
# TODO: We need the basic flags and options

@vegeta_app.command()
def schema():
    """
    Display the JSON Schema 
    """
    typer.echo(f"Hello")


@vegeta_app.command()
def validate(file: typer.FileText = typer.Argument(...)):
    """
    Validate given file against the JSON Schema
    """
    typer.echo("Validate")

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

# TODO: Moves to CLI.py
app = typer.Typer()
app.add_typer(vegeta_app, name="vegeta", help="Vegeta load generator")

vegeta = VegetaConnector(config_file="./servo.yaml", app="dev.opsani.com/blake", token="sadasdsa")
servo = Servo(config_file="./servo.yaml", app="dev.opsani.com/blake", token="sadasdsa")
servo.add_connector(vegeta)

@app.command()
def schema():
    """
    Display the JSON Schema 
    """
    # print(Servo.schema_json(indent=2))
    print(Config.schema_json(indent=2))

@app.command()
def validate(file: typer.FileText = typer.Argument(...)):
    """
    Validate given file against the JSON Schema
    """
    # TODO: read the raw config file
    config = yaml.load(file, Loader=yaml.FullLoader)
    connector_config = config['vegeta']
    try:
        config = Config.parse_obj(connector_config)
        typer.echo("√ Valid connector configuration")
    except ValidationError as e:
        typer.echo("X Invalid connector configuration")
        print(e)
    # typer.echo("Validate" + str(connector_config))
    pyaml.p({ 'vegeta': connector_config})

# TODO: Needs to take a list of connectors
# default to using all of them
@app.command()
def generate():
    """
    Generate a new config file
    """
    pass

# TODO: Does this need to be shared?
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

if __name__ == "__main__":
    app()