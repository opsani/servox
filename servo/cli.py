import typer
import inspect
import sys
from servo.connector import Connector, Servo, Optimizer, ServoSettings, VegetaSettings
from devtools import debug
import json
import yaml
import pydantic
from pydantic import ValidationError
from pydantic.schema import schema
from pydantic.json import pydantic_encoder
from servo.connector import ServoSettings, VegetaSettings
from typing import get_type_hints
from pathlib import Path
from pygments import highlight
from pygments.lexers import YamlLexer, JsonLexer
from pygments.formatters import TerminalFormatter
from tabulate import tabulate

# SERVO_OPTIMIZER (--optimizer -o no default)
# SERVO_TOKEN (--token -t no default)
# SERVO_TOKEN_FILE (--token-file -T ./servo.token)
# SERVO_CONFIG_FILE (--config-file -c ./servo.yaml)

# Use callback to define top-level options
# TODO: Need a way to intelligently opt in or out of this. Maybe a new decorator
def root_callback(optimizer: str = typer.Option(None, help="Opsani optimizer (format is example.com/app)"), 
             token: str = typer.Option(None, help="Opsani API access token"), 
             base_url: str = typer.Option("https://api.opsani.com/", help="Base URL for connecting to Opsani API")):
    pass

app = typer.Typer(name="servox", add_completion=True, callback=root_callback)

# TODO: check if there is a servo.yaml (Need to support --config/-c at some point)
# TODO: Load from env or arguments
settings: ServoSettings = None
optimizer = Optimizer('dev.opsani.com/fake-app-name', '0000000000000000000000000000000000000000000000000000000')
config_file = Path.cwd() / 'servo.yaml'
if config_file.exists():
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    config['optimizer'] = optimizer.dict()
    settings = ServoSettings.parse_obj(config)
else:
    settings = ServoSettings(optimizer=optimizer)

# TODO: What is the behavior here outside of a project?
servo = Servo(settings)
for cls in Connector.all():
    if cls != Servo:
        # NOTE: Read the type hint to find our settings class
        hints = get_type_hints(cls)
        settings_cls = hints['settings']
        settings = settings_cls.construct()
        connector = cls(settings)
        cli = connector.cli()
        if cli is not None:
            app.add_typer(cli)

# NOTE: Two modes of operation: In an assembly and out

@app.command()
def new() -> None:
    """Creates a new servo assembly at [PATH]"""
    # TODO: Specify a list of connectors (or default to all)
    # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
    # TODO: Options for Docker Compose and Kubernetes?
    pass

@app.command()
def run() -> None:
    """Run the servo"""
    pass

@app.command()
def console() -> None:
    """Open an interactive console"""
    # TODO: Load up the environment and trigger IPython
    pass

@app.command()
def info() -> None:
    '''Display information about the assembly'''
    # TODO: add verbose output to show more
    headers = ["NAME", "VERSION", "DESCRIPTION"]
    table = [
        [servo.name, servo.version, servo.description]
    ]

    for connector in servo.available_connectors():
        table.append([connector.name, connector.version, connector.description])
    
    typer.echo(tabulate(table, headers, tablefmt="plain"))
    # typer.echo((
    #     f"{servo.name} v{servo.version} ({servo.maturity})\n"
    #     f"{servo.description}\n"
    #     f"{servo.homepage}\n"
    #     f"Licensed under the terms of {servo.license}\n"
    # ))

@app.command()
def settings() -> None:
    '''Display the fully resolved settings'''
    settings = servo.settings.dict(exclude={'optimizer'}, exclude_unset=True)
    settings_yaml = yaml.dump(settings, indent=4, sort_keys=True)
    typer.echo(highlight(settings_yaml, YamlLexer(), TerminalFormatter()))

@app.command()
def check() -> None:
    '''Check the health of the assembly'''
    # TODO: Requires a config file
    # TODO: Run checks for all active connectors
    pass

@app.command()
def version() -> None:
    '''Display version and exit'''
    # TODO: Refactor this to share the implementation in Vegeta
    pass

@app.command()
def schema() -> None:
    '''Display configuration schema'''
    # TODO: Read config file, find all loaded connectors, bundle into a schema...
    ServoModel = pydantic.create_model(
        'ServoModel',
        servo=(ServoSettings, ...),
        vegeta=(VegetaSettings, ...)
    )
    typer.echo(highlight(ServoModel.schema_json(indent=2), JsonLexer(), TerminalFormatter()))

    # TODO: top level only includes definitions and no keys (put under an option)
    # from pydantic.schema import schema as pydantic_schema
    # top_level_schema = pydantic_schema([ServoSettings, VegetaSettings], title='Servo Schema')
    # typer.echo(json.dumps(top_level_schema, indent=2, default=pydantic_encoder))

@app.command(name='validate')
def validate(file: typer.FileText = typer.Argument('servo.yaml')) -> None:
    """Validate servo configuration file"""
    ServoModel = pydantic.create_model(
        'ServoModel',
        servo=(ServoSettings, ...),
        vegeta=(VegetaSettings, ...)
    )
    try:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config_descriptor = ServoModel.parse_obj(config)
        typer.echo("√ Valid servo configuration")
    except ValidationError as e:
        typer.echo("X Invalid servo configuration")
        typer.echo(e, err=True)

@app.command(name='generate')
def generate() -> None:
    """Generate servo configuration"""
    # TODO: Dump the Servo settings, then all connectors by id

    pass

### Begin connector subcommands
connector_app = typer.Typer(name='connector', help="Manage connectors")
app.add_typer(connector_app)

@connector_app.command(name='list')
def connectors_list() -> None:
    '''List connectors in the assembly'''
    pass

@connector_app.command(name='add')
def connectors_add() -> None:
    '''Add a connector to the assembly'''
    pass

@connector_app.command(name='remove')
def connectors_remove() -> None:
    '''Remove a connector from the assembly'''
    pass

### Begin developer subcommands
# NOTE: registered as top level commands for convenience in dev

@app.command(name='test')
def developer_test() -> None:
    '''Run automated tests'''
    pass

@app.command(name='lint')
def developer_lint() -> None:
    '''Emit opinionated linter warnings and suggestions'''
    pass

@app.command(name='format')
def developer_format() -> None:
    '''Apply automatic formatting to the codebase'''
    pass

# Run the Typer CLI
def main():
    app()
