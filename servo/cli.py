import typer
import inspect
import sys
from servo.connector import Connector, Servo, Optimizer, ServoSettings, VegetaSettings
from devtools import debug
import json
import yaml
import pydantic
import subprocess
import shlex
from pydantic import ValidationError, Extra
from pydantic.schema import schema as pydantic_schema
from pydantic.json import pydantic_encoder
from servo.connector import ServoSettings, VegetaSettings
from typing import get_type_hints, Union, List
from pathlib import Path
from pygments import highlight
from pygments.lexers import YamlLexer, JsonLexer
from pygments.formatters import TerminalFormatter
from tabulate import tabulate

# SERVO_OPTIMIZER (--optimizer -o no default)
# SERVO_TOKEN (--token -t no default)
# SERVO_TOKEN_FILE (--token-file -T ./servo.token)
# SERVO_CONFIG_FILE (--config-file -c ./servo.yaml)

servo: Servo = None

# Use callback to define top-level options
def root_callback(optimizer: str = typer.Option(None, help="Opsani optimizer (format is example.com/app)"), 
             token: str = typer.Option(None, help="Opsani API access token"), 
             base_url: str = typer.Option("https://api.opsani.com/", help="Base URL for connecting to Opsani API")):
    global servo

    # TODO: check if there is a servo.yaml (Need to support --config/-c at some point)
    # TODO: Load from env or arguments
    settings: ServoSettings = None
    optimizer = Optimizer('dev.opsani.com/fake-app-name', '0000000000000000000000000000000000000000000000000000000')
    config_file = Path.cwd() / 'servo.yaml'

    if config_file.exists():
        args = {}
        for c in Connector.all():
            if c is not Servo:
                args[c.default_id()] = (c.settings_class(), ...)
        ServoModel = pydantic.create_model(
            'Servo',
            __base__=ServoSettings,
            optimizer=optimizer,
            extra=Extra.forbid,
            **args,
        )
        try:
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            settings = ServoModel.parse_obj(config)
        except ValidationError as error:
            typer.echo(error, err=True)
            sys.exit(2)
    else:
        # If we do not have a project, build a minimal configuration
        settings = ServoSettings(optimizer=optimizer)

    # Connect the CLIs for all connectors
    # TODO: This should respect the connectors list when there is a config file present
    servo = Servo(settings)
    for cls in servo.available_connectors():
        settings = cls.settings_class().construct()
        connector = cls(settings)
        cli = connector.cli()
        if cli is not None:
            app.add_typer(cli)

app = typer.Typer(name="servox", add_completion=True, callback=root_callback)

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
def info(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Display verbose info")
) -> None:
    '''Display information about the assembly'''    
    headers = ["NAME", "VERSION", "DESCRIPTION"]
    row = [servo.name, servo.version, servo.description]    
    if verbose:
        headers += ["HOMEPAGE", "MATURITY", "LICENSE"]
        row += [servo.homepage, servo.maturity, servo.license]
    table = [row]
    for connector in servo.available_connectors():
        row = [connector.name, connector.version, connector.description]
        if verbose:
            row += [connector.homepage, connector.maturity, connector.license]
        table.append(row)
    
    typer.echo(tabulate(table, headers, tablefmt="plain"))

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
    typer.echo(f'{servo.name} v{servo.version}')
    pass

@app.command()
def schema(
    all: bool = typer.Option(False, "--all", "-a", help="Include models from all available connectors"),
    top_level: bool = typer.Option(False, "--top-level", help="Emit a top-level schema (only connector models)")
) -> None:
    '''Display configuration schema'''
    connectors = servo.available_connectors() if all else servo.active_connectors()
    if top_level:
        settings_classes = list(map(lambda c: c.settings_class(), connectors))
        top_level_schema = pydantic_schema(settings_classes, title='Servo Schema')
        typer.echo(highlight(json.dumps(top_level_schema, indent=2, default=pydantic_encoder), JsonLexer(), TerminalFormatter()))    
    else:
        args = {}
        for c in connectors:
            # FIXME: this should be id, not default_id but we need instances
            args[c.default_id()] = (c.settings_class(), ...)
        ServoModel = pydantic.create_model(
            'Servo',
            **args
        )
        typer.echo(highlight(ServoModel.schema_json(indent=2), JsonLexer(), TerminalFormatter()))    

@app.command(name='validate')
def validate(
    file: typer.FileText = typer.Argument('servo.yaml'),
    all: bool = typer.Option(False, "--all", "-a", help="Include models from all available connectors"),
) -> None:
    """Validate servo configuration file"""
    connectors = servo.available_connectors() if all else servo.active_connectors()
    args = {}
    for c in connectors:
        # FIXME: this should be id, not default_id but we need instances
        args[c.default_id()] = (c.settings_class(), ...)
    ServoModel = pydantic.create_model(
        'Servo',
        **args
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
    # TODO: Add force and output path options
    schema = servo.settings.dict(by_alias=True, exclude={'optimizer'})
    connectors = servo.available_connectors()
    for connector in connectors:
        # NOTE: We generate with a potentially incomplete settings instance
        # if there is required configuration without reasonable defaults. This
        # should be fine because the errors at load time will be clear and we cna
        # embed examples into the schema or put in sentinel values.
        settings = cls.settings_class().construct()
        connector = cls(settings)        
        schema[connector.default_id()] = connector.settings.dict(by_alias=True)
    
    # NOTE: We have to serialize through JSON first
    schema_obj = json.loads(json.dumps(schema))
    output_path = Path.cwd() / 'servo.yaml'
    output_path.write_text(yaml.dump(schema_obj))
    typer.echo("Generated servo.yaml")

### Begin developer subcommands
# NOTE: registered as top level commands for convenience in dev

@app.command(name='test')
def developer_test() -> None:
    '''Run automated tests'''
    __run('pytest --cov=servo --cov=tests --cov-report=term-missing --cov-config=setup.cfg tests')

@app.command(name='lint')
def developer_lint() -> None:
    '''Emit opinionated linter warnings and suggestions'''
    cmds = [
        'flake8 app --exclude=db',
        'mypy app',
        'black --check servo --diff',
        'isort --recursive --check-only servo',
    ]
    for cmd in cmds:        
        __run(cmd)

@app.command(name='format')
def developer_format() -> None:
    '''Apply automatic formatting to the codebase'''
    cmds = [
        'isort --recursive  --force-single-line-imports servo tests',
        'autoflake --recursive --remove-all-unused-imports --remove-unused-variables --in-place servo tests',
        'black app tests',
        'isort --recursive app tests'
    ]
    for cmd in cmds:        
        __run(cmd, check=True)

def __run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)

# Run the Typer CLI
def main():
    app()
