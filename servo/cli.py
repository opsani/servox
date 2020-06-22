import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Type, Union, Optional
from enum import Enum

import pydantic
import typer
import yaml
from devtools import pformat
from pydantic import Extra, ValidationError
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import JsonLexer, YamlLexer, PythonLexer
from tabulate import tabulate

from servo.connector import Connector, Optimizer, Servo, ServoSettings, ConnectorLoader

# Add the devtools debug() function to the CLI if its available
try:
    import builtins
    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug

# SERVO_OPTIMIZER (--optimizer -o no default)
# SERVO_TOKEN (--token -t no default)
# SERVO_TOKEN_FILE (--token-file -T ./servo.token)
# SERVO_CONFIG_FILE (--config-file -c ./servo.yaml)

servo: Servo = None
ServoModel: Type = None

# Use callback to define top-level options
# TODO: Make these args required
def root_callback(
    optimizer: str = typer.Option(
        None, help="Opsani optimizer (format is example.com/app)"
    ),
    token: str = typer.Option(None, help="Opsani API access token"),
    base_url: str = typer.Option(
        "https://api.opsani.com/", help="Base URL for connecting to Opsani API"
    ),
):
    global servo, ServoModel

    # TODO: check if there is a servo.yaml (Need to support --config/-c at some point)
    # TODO: Load from env or arguments
    settings: ServoSettings = None
    optimizer = Optimizer(
        "dev.opsani.com/fake-app-name",
        "0000000000000000000000000000000000000000000000000000000",
    )
    config_file = Path.cwd() / "servo.yaml"

    # Build our dynamic model
    # TODO: This logic moves to servo class
    # TODO: requirement of fields (...) should depend on how connectors are configured
    # TODO: when autoloaded, not required. When explicitly configured, is required.
    # TODO: Generation, info, etc and other commands need to be able to run with invalid config
    args = {}
    for c in Connector.all():
        if c is not Servo:
            args[c.default_id()] = (c.settings_class(), ...)

    ServoModel = pydantic.create_model(
        "Servo",
        __base__=ServoSettings,
        optimizer=(Optimizer, ...),        
        **args,
    )

    # Load a file if we have one
    if config_file.exists():
        try:
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            config['optimizer'] = optimizer.dict()
            settings = ServoModel.parse_obj(config)
        except ValidationError as error:
            typer.echo(error, err=True)
            sys.exit(2)
    else:
        # If we do not have a project, build a minimal configuration
        args = {}
        for c in Connector.all():
            if c is not Servo:
                args[c.default_id()] = c.settings_class().construct()
        settings = ServoModel(optimizer=optimizer, **args)

    # Connect the CLIs for all connectors
    # TODO: This should respect the connectors list when there is a config file present
    servo = Servo(settings)
    # for cls in servo.all_connectors():
    #     settings = cls.settings_class().construct()
    #     connector = cls(settings)
    #     cli = connector.cli()
    #     if cli is not None:
    #         app.add_typer(cli)


app = typer.Typer(name="servox", add_completion=True, callback=root_callback)

# Load all the connector plugins
loader = ConnectorLoader()
for connector in loader.load():
    debug(str(connector))
    settings = connector.settings_class().construct()
    connector = connector(settings)
    cli = connector.cli()
    if cli is not None:
        app.add_typer(cli)

@app.command()
def new() -> None:
    """Creates a new servo assembly at [PATH]"""
    # TODO: Specify a list of connectors (or default to all)
    # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
    # TODO: Options for Docker Compose and Kubernetes?


@app.command()
def run() -> None:
    """Run the servo"""


@app.command()
def console() -> None:
    """Open an interactive console"""
    # TODO: Load up the environment and trigger IPython


@app.command()
def info(
    all: bool = typer.Option(
        False, "--all", "-a", help="Include models from all available connectors"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Display verbose info"
    )
) -> None:
    """Display information about the assembly"""
    connectors = servo.all_connectors() if all else servo.active_connectors()
    headers = ["NAME", "VERSION", "DESCRIPTION"]
    row = [servo.name, servo.version, servo.description]
    if verbose:
        headers += ["HOMEPAGE", "MATURITY", "LICENSE"]
        row += [servo.homepage, servo.maturity, servo.license]
    table = [row]
    for connector in connectors:
        row = [connector.name, connector.version, connector.description]
        if verbose:
            row += [connector.homepage, connector.maturity, connector.license]
        table.append(row)

    typer.echo(tabulate(table, headers, tablefmt="plain"))

# Common output formats
YAML_FORMAT = 'yaml'
JSON_FORMAT = 'json'
DICT_FORMAT = 'dict'
HTML_FORMAT = 'html'
TEXT_FORMAT = 'text'
MARKDOWN_FORMAT = 'markdown'

class AbstractOutputFormat(str, Enum):
    '''Defines common behaviors for command specific output format enumerations'''

    def lexer(self) -> Optional['pygments.Lexer']:
        if self.value == YAML_FORMAT:
            return YamlLexer()
        elif self.value == JSON_FORMAT:
            return JsonLexer()
        elif self.value == DICT_FORMAT:
            return PythonLexer()
        elif self.value == SettingsOutputFormat.text:
            return None
        else:
            raise RuntimeError("no lexer configured for output format {self.value}")

class SettingsOutputFormat(AbstractOutputFormat):
    yaml = YAML_FORMAT
    json = JSON_FORMAT
    dict = DICT_FORMAT
    text = TEXT_FORMAT

@app.command()
def settings(
    format: SettingsOutputFormat = typer.Option(
        SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
    ),
    output: typer.FileTextWrite = typer.Option(
        None, "--output", "-o", help="Output settings to [FILE]"
    )
) -> None:
    """Display the fully resolved settings"""
    settings = servo.settings.dict(exclude={"optimizer", "extra"}, exclude_unset=True)
    settings_json = json.dumps(settings, indent=2)
    settings_dict = json.loads(settings_json)
    settings_dict_str = pformat(settings_dict)
    settings_yaml = yaml.dump(settings_dict, indent=4, sort_keys=True)

    if format == SettingsOutputFormat.text:
        pass
    else:
        lexer = format.lexer()
        if format == SettingsOutputFormat.yaml:
            data = settings_yaml
        elif format == SettingsOutputFormat.json:
            data = settings_json
        elif format == SettingsOutputFormat.dict:
            data = settings_dict_str
        else:
            raise RuntimeError("no handler configured for output format {format}")
        
        if output:
            output.write(data)
        else:
            typer.echo(
                highlight(
                    data, 
                    lexer, 
                    TerminalFormatter()
                )
            )

@app.command()
def check() -> None:
    """Check the health of the assembly"""
    # TODO: Requires a config file
    # TODO: Run checks for all active connectors


@app.command()
def version() -> None:
    """Display version and exit"""
    typer.echo(f"{servo.name} v{servo.version}")
    raise typer.Exit(0)

class SchemaOutputFormat(AbstractOutputFormat):    
    json = JSON_FORMAT
    text = TEXT_FORMAT    
    dict = DICT_FORMAT
    html = HTML_FORMAT

@app.command()
def schema(
    all: bool = typer.Option(
        False, "--all", "-a", help="Include models from all available connectors"
    ),
    top_level: bool = typer.Option(
        False, "--top-level", help="Emit a top-level schema (only connector models)"
    ),
    format: SchemaOutputFormat = typer.Option(
        SchemaOutputFormat.json, "--format", "-f", help="Select output format"
    ),
    output: typer.FileTextWrite = typer.Option(
        None, "--output", "-o", help="Output schema to [FILE]"
    )
) -> None:
    """Display configuration schema"""
    if format == SchemaOutputFormat.text or format == SchemaOutputFormat.html:
        typer.echo("error: not yet implemented", err=True)
        raise typer.Exit(1)    
    
    if top_level:
        if format == SchemaOutputFormat.json:
            output_data = servo.top_level_schema_json(all=all)
        
        elif format == SchemaOutputFormat.dict:            
            output_data = pformat(servo.top_level_schema(all=all))

    else:
        if format == SettingsOutputFormat.json:
            output_data = ServoModel.schema_json(indent=2)
        elif format == SettingsOutputFormat.dict:
            output_data = pformat(ServoModel.schema())
        else:
            raise RuntimeError("no handler configured for output format {format}")
    
    assert output_data is not None, "output_data not assigned"

    if output:
        output.write(output_data)
    else:
        typer.echo(
            highlight(
                output_data, 
                format.lexer(), 
                TerminalFormatter()
            )
        )

@app.command(name="validate")
def validate(
    file: typer.FileText = typer.Argument("servo.yaml"),
    all: bool = typer.Option(
        False, "--all", "-a", help="Include models from all available connectors"
    ),
) -> None:
    """Validate servo configuration file"""
    try:
        config = yaml.load(file, Loader=yaml.FullLoader)
        ServoModel.parse_obj(config)
        typer.echo("√ Valid servo configuration")
    except ValidationError as e:
        typer.echo("X Invalid servo configuration")
        typer.echo(e, err=True)


@app.command(name="generate")
def generate() -> None:
    """Generate servo configuration"""
    # TODO: Add force and output path options
    schema = servo.settings.dict(by_alias=True, exclude={"optimizer"})

    # NOTE: We generate with a potentially incomplete settings instance
    # if there is required configuration without reasonable defaults. This
    # should be fine because the errors at load time will be clear and we can
    # embed examples into the schema or put in sentinel values.
    # NOTE: We have to serialize through JSON first
    schema_obj = json.loads(json.dumps(schema))
    output_path = Path.cwd() / "servo.yaml"
    output_path.write_text(yaml.dump(schema_obj))
    typer.echo("Generated servo.yaml")


### Begin developer subcommands
# NOTE: registered as top level commands for convenience in dev


@app.command(name="test")
def developer_test() -> None:
    """Run automated tests"""
    __run(
        "pytest --cov=servo --cov=tests --cov-report=term-missing --cov-config=setup.cfg tests"
    )


@app.command(name="lint")
def developer_lint() -> None:
    """Emit opinionated linter warnings and suggestions"""
    cmds = [
        "flake8 servo",
        "mypy servo",
        "black --check servo --diff",
        "isort --recursive --check-only servo",
    ]
    for cmd in cmds:
        __run(cmd)


@app.command(name="format")
def developer_format() -> None:
    """Apply automatic formatting to the codebase"""
    cmds = [
        "isort --recursive  --force-single-line-imports servo tests",
        "autoflake --recursive --remove-all-unused-imports --remove-unused-variables --in-place servo tests",
        "black servo tests",
        "isort --recursive servo tests",
    ]
    for cmd in cmds:
        __run(cmd)


def __run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)


# Run the Typer CLI
def main():
    app()
