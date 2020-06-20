import typer
import inspect
import sys
from servo.connector import Connector, Servo, Optimizer, ServoSettings, VegetaSettings
from devtools import debug
import json
import pydantic
from pydantic.schema import schema
from pydantic.json import pydantic_encoder
from servo.connector import ServoSettings, VegetaSettings
from typing import get_type_hints

# OPSANI_OPTIMIZER (--optimizer -o no default)
# OPSANI_TOKEN (--token -t no default)
# OPSANI_TOKEN_FILE (--token-file -T ./servo.token)
# OPSANI_CONFIG_FILE (--config-file -c ./servo.yaml)

# Use callback to define top-level options
# TODO: Need a way to intelligently opt in or out of this. Maybe a new decorator
def root_callback(optimizer: str = typer.Option(None, help="Opsani optimizer (format is example.com/app)"), 
             token: str = typer.Option(None, help="Opsani API access token"), 
             base_url: str = typer.Option("https://api.opsani.com/", help="Base URL for connecting to Opsani API")):
    pass

app = typer.Typer(name="servox", add_completion=True, callback=root_callback)

# TODO: Move this into Servo settings
# TODO: Load from env or arguments
optimizer = Optimizer('dev.opsani.com/fake-app-name', '0000000000000000000000000000000000000000000000000000000')
servo = Servo(optimizer)
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

# NOTE: Two modes of operation. In an assembly and out
# TODO: To load the subclasses we need config and know which ones are active...

@app.command()
def new() -> None:
    """Creates a new servo assembly at [PATH]"""
    pass

@app.command()
def run() -> None:
    """Run the servo"""
    pass

@app.command()
def console() -> None:
    """Open an interactive console"""
    pass

@app.command()
def info() -> None:
    '''Display information about the assembly'''
    pass

@app.command()
def settings() -> None:
    '''Display the fully resolved settings'''
    pass

@app.command()
def check() -> None:
    '''Check the health of the assembly'''
    pass

@app.command()
def version() -> None:
    '''Display version and exit'''
    pass

### Begin config subcommands
# TODO: It may make more sense to have these top-level
config_app = typer.Typer(name='config', help="Manage configuration")
app.add_typer(config_app)

@app.command()
def config() -> None:
    """Display servo configuration"""
    pass

@config_app.command(name='schema')
def config_schema() -> None:
    '''Display configuration schema'''
    # TODO: Read config file, find all loaded connectors, bundle into a schema...
    # ServoModel = pydantic.create_model(
    #     'ServoModel',
    #     servo=(ServoSettings, ...),
    #     vegeta=(VegetaSettings, ...)
    # )
    # typer.echo(ServoModel.schema_json(indent=2))
    top_level_schema = schema([ServoSettings, VegetaSettings], title='Servo Schema')
    print(json.dumps(top_level_schema, indent=2, default=pydantic_encoder))

@config_app.command(name='validate')
def config_validate() -> None:
    """Validate servo configuration file"""
    pass

@config_app.command(name='generate')
def config_generate() -> None:
    """Generate servo configuration"""
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
