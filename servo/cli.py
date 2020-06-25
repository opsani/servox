import json
import os
import shlex
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Type, Union
import typer
import yaml
from devtools import pformat
from pydantic import ValidationError
from pydantic.json import pydantic_encoder
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import JsonLexer, PythonLexer, YamlLexer
from tabulate import tabulate

from servo.connector import ConnectorLoader, Optimizer
from servo.servo import BaseServoSettings, Servo, ServoAssembly

# Add the devtools debug() function to the CLI if its available
try:
    import builtins
    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug

# Application state available to all commands
# These objects are constructed in `root_callback`
assembly: ServoAssembly
ServoSettings: Type[BaseServoSettings]
servo: Servo
connectors_to_update = []

# Build the Typer CLI
cli = typer.Typer(name="servox", add_completion=True, no_args_is_help=True)

@cli.callback()
def root_callback(
    optimizer: str = typer.Option(
        None,
        envvar="OPSANI_OPTIMIZER",
        show_envvar=True,
        metavar="OPTIMIZER",
        help="Opsani optimizer to connect to (format is example.com/app)",        
    ),
    token: str = typer.Option(
        None,
        envvar="OPSANI_TOKEN",
        show_envvar=True,
        metavar="TOKEN",
        help="Opsani API access token",
    ),
    token_file: Path = typer.Option(
        None,
        envvar="OPSANI_TOKEN_FILE",
        show_envvar=True,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        help="File to load the access token from",
    ),
    base_url: str = typer.Option(
        "https://api.opsani.com/",
        "--base-url",
        envvar="OPSANI_BASE_URL",
        show_envvar=True,
        show_default=True,
        metavar="URL",
        help="Base URL for connecting to Opsani API",
    ),
    config_file: Path = typer.Option(
        "servo.yaml",
        "--config-file",
        "-c",
        envvar="OPSANI_CONFIG_FILE",
        show_envvar=True,
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        help="Servo configuration file",
    ),
):
    if optimizer is None:
        raise typer.BadParameter("An optimizer must be specified")
    
    # Resolve token
    if token is None and token_file is None:
        raise typer.BadParameter(
            "API token must be provided via --token, --token-file, or ENV['OPSANI_TOKEN']"
        )

    if token is not None and token_file is not None:
        raise typer.BadParameter("--token and --token-file cannot both be given")

    if token_file is not None and token_file.exists():
        token = token_file.read_text()
    
    # TODO: this can be pushed to the optimizer model
    if len(token) == 0 or token.isspace():
        raise typer.BadParameter("token cannot be blank")

    optimizer = Optimizer(optimizer, token=token, base_url=base_url)

    # Assemble the Servo
    global assembly, servo, ServoSettings # TODO: This should probably return the instance instead of the model
    try:
        assembly, servo, ServoSettings = ServoAssembly.assemble(
            config_file=config_file, optimizer=optimizer
        )
    except ValidationError as error:
        typer.echo(error, err=True)
        raise typer.Exit(2) from error
    
    # FIXME: Update the settings of our pre-registered connectors
    for connector in connectors_to_update:
        settings = getattr(servo.settings, connector.config_path)
        connector.settings = settings

@cli.command()
def new() -> None:
    """Creates a new servo assembly at [PATH]"""
    # TODO: Specify a list of connectors (or default to all)
    # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
    # TODO: Options for Docker Compose and Kubernetes?


@cli.command()
def run() -> None:
    """Run the servo"""
    servo.run()

@cli.command()
def console() -> None:
    """Open an interactive console"""
    # TODO: Load up the environment and trigger IPython


@cli.command()
def info(
    all: bool = typer.Option(
        False, "--all", "-a", help="Include models from all available connectors"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Display verbose info"),
) -> None:
    """Display information about the assembly"""
    connectors = assembly.all_connectors() if all else servo.connectors
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
YAML_FORMAT = "yaml"
JSON_FORMAT = "json"
DICT_FORMAT = "dict"
HTML_FORMAT = "html"
TEXT_FORMAT = "text"
MARKDOWN_FORMAT = "markdown"


class AbstractOutputFormat(str, Enum):
    """Defines common behaviors for command specific output format enumerations"""

    def lexer(self) -> Optional["pygments.Lexer"]:
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


@cli.command()
def settings(
    format: SettingsOutputFormat = typer.Option(
        SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
    ),
    output: typer.FileTextWrite = typer.Option(
        None, "--output", "-o", help="Output settings to [FILE]"
    ),
) -> None:
    """Display the fully resolved settings"""
    settings = servo.settings.dict(exclude={"optimizer"}, exclude_unset=True)
    settings_json = json.dumps(settings, indent=2, default=pydantic_encoder)
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
            typer.echo(highlight(data, lexer, TerminalFormatter()))


@cli.command()
def check() -> None:
    """Check the health of the assembly"""
    # TODO: Requires a config file
    # TODO: Run checks for all active connectors (or pick them)


@cli.command()
def version() -> None:
    """Display version and exit"""
    typer.echo(f"{servo.name} v{servo.version}")
    raise typer.Exit(0)


class SchemaOutputFormat(AbstractOutputFormat):
    json = JSON_FORMAT
    text = TEXT_FORMAT
    dict = DICT_FORMAT
    html = HTML_FORMAT


@cli.command()
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
    ),
) -> None:
    """Display configuration schema"""
    if format == SchemaOutputFormat.text or format == SchemaOutputFormat.html:
        typer.echo("error: not yet implemented", err=True)
        raise typer.Exit(1)

    if top_level:
        if format == SchemaOutputFormat.json:
            output_data = assembly.top_level_schema_json(all=all)

        elif format == SchemaOutputFormat.dict:
            output_data = pformat(assembly.top_level_schema(all=all))

    else:
        if format == SettingsOutputFormat.json:
            output_data = ServoSettings.schema_json(indent=2)
        elif format == SettingsOutputFormat.dict:
            output_data = pformat(ServoSettings.schema())
        else:
            raise RuntimeError("no handler configured for output format {format}")

    assert output_data is not None, "output_data not assigned"

    if output:
        output.write(output_data)
    else:
        typer.echo(highlight(output_data, format.lexer(), TerminalFormatter()))


@cli.command(name="validate")
def validate(
    file: typer.FileText = typer.Argument("servo.yaml"),
    all: bool = typer.Option(
        False, "--all", "-a", help="Include models from all available connectors"
    ),
) -> None:
    """Validate servo configuration file"""
    try:
        assembly.parse_file(file)
        typer.echo("√ Valid servo configuration")
    except ValidationError as e:
        typer.echo("X Invalid servo configuration")
        typer.echo(e, err=True)


@cli.command(name="generate")
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


@cli.command(name="test")
def developer_test() -> None:
    """Run automated tests"""
    __run(
        "pytest --cov=servo --cov=tests --cov-report=term-missing --cov-config=setup.cfg tests"
    )


@cli.command(name="lint")
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


@cli.command(name="format")
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
