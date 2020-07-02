import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Union, Optional

import typer
import yaml
from devtools import pformat
from pydantic import ValidationError
from pydantic.json import pydantic_encoder
from pygments import highlight
from pygments.formatters import TerminalFormatter
from tabulate import tabulate

from servo.connector import Connector, ConnectorSettings, Optimizer
from servo.servo import Servo, ServoAssembly, Events
from servo.servo_runner import ServoRunner
from servo.types import *

# Add the devtools debug() function to the CLI if its available
try:
    import builtins
    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug


# Represents an option to include specific CLI commands
# A value of `None` (the typical default) enables auto-detection logic
CommandOption = Optional[bool]


# NOTE: There is a life-cycle dependency issue where we want to have commands
# available only for active connectors but this requires the config file to be
# parsed and the commands to be registered. The connectors have to be updated
# after the servo is assembled to carry the right state. See entry_points.py
# and the callback on ServoCLI for details.
connectors_to_update = []


class SharedCommandsMixin:
    servo: Servo
    settings: ConnectorSettings
    connectors: List[Connector]
    hide_servo_options: bool = True

    def add_shared_commands(self,
        version: CommandOption = True,
        schema: CommandOption = None,
        settings: CommandOption = None,
        generate: CommandOption = None,
        validate: CommandOption = None,
        events: CommandOption = None,
        describe: CommandOption = None,
        check: CommandOption = None,        
        measure: CommandOption = None,
        adjust: CommandOption = None,
        promote: CommandOption = None,
        **kwargs,
    ):
        class SettingsOutputFormat(AbstractOutputFormat):
            yaml = YAML_FORMAT
            json = JSON_FORMAT
            dict = DICT_FORMAT
            text = TEXT_FORMAT

        @self.command()
        def settings(
            format: SettingsOutputFormat = typer.Option(
                SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
            ),
            output: typer.FileTextWrite = typer.Option(
                None, "--output", "-o", help="Output settings to [FILE]"
            ),
        ) -> None:
            """Display the fully resolved settings"""
            settings = self.settings.dict(exclude_unset=True)
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
                    raise RuntimeError(
                        "no handler configured for output format {format}"
                    )

                if output:
                    output.write(data)
                else:
                    typer.echo(highlight(data, lexer, TerminalFormatter()))

        @self.command()
        def check() -> None:
            """Check the health of the assembly"""
            # TODO: Requires a config file
            # TODO: Run checks for all active connectors (or pick them)
            results: List[EventResult] = self.servo.dispatch_event(
                Events.CHECK, include=self.connectors
            )
            headers = ["CONNECTOR", "CHECK", "STATUS", "COMMENT"]
            table = []
            for result in results:
                check: CheckResult = result.value
                status = "√ PASSED" if check.success else "X FAILED"
                row = [result.connector.name, check.name, status, check.comment]
                table.append(row)
            
            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @self.command()
        def describe() -> None:
            """
            Describe metrics and settings
            """
            results: List[EventResult] = self.servo.dispatch_event(
                Events.DESCRIBE, include=self.connectors
            )
            headers = ["CONNECTOR", "COMPONENTS", "METRICS"]
            table = []
            for result in results:
                description: Description = result.value
                components_column = []
                for component in description.components:                     
                    for setting in component.settings:
                        components_column.append(f"{component.name}.{setting.name}={setting.value}")

                metrics_column = []
                for metric in description.metrics:
                    metrics_column.append(f"{metric.name} ({metric.unit})")

                row = [result.connector.name, "\n".join(components_column), "\n".join(metrics_column)]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        class SchemaOutputFormat(AbstractOutputFormat):
            json = JSON_FORMAT
            text = TEXT_FORMAT
            dict = DICT_FORMAT
            html = HTML_FORMAT

        @self.command()
        def schema(
            all: bool = typer.Option(
                False,
                "--all",
                "-a",
                help="Include models from all available connectors",
                hidden=self.hide_servo_options,
            ),
            top_level: bool = typer.Option(
                False,
                "--top-level",
                help="Emit a top-level schema (only connector models)",
                hidden=self.hide_servo_options,
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
                    output_data = self.assembly.top_level_schema_json(all=all)

                elif format == SchemaOutputFormat.dict:
                    output_data = pformat(self.assembly.top_level_schema(all=all))

            else:
                
                settings_class = self.settings.__class__
                if format == SchemaOutputFormat.json:
                    output_data = settings_class.schema_json(indent=2)
                elif format == SchemaOutputFormat.dict:
                    output_data = pformat(settings_class.schema())
                else:
                    raise RuntimeError(
                        "no handler configured for output format {format}"
                    )

            assert output_data is not None, "output_data not assigned"

            if output:
                output.write(output_data)
            else:
                typer.echo(highlight(output_data, format.lexer(), TerminalFormatter()))

        @self.command(name="validate")
        def validate(
            file: Path = typer.Argument(
                "servo.yaml",
                exists=True,
                file_okay=True,
                dir_okay=False,
                writable=False,
                readable=True,
            ),
            all: bool = typer.Option(
                False,
                "--all",
                "-a",
                help="Include models from all available connectors",
                hidden=self.hide_servo_options,
            ),
        ) -> None:
            """Validate servo configuration file"""
            try:
                self.connector.settings_model().parse_file(file)
                typer.echo(f"√ Valid {self.connector.name} configuration in {file}")
            except (ValidationError, yaml.scanner.ScannerError) as e:
                typer.echo(f"X Invalid {self.connector.name} configuration in {file}")
                typer.echo(e, err=True)
                raise typer.Exit(1)

        @self.command()
        def events():
            """
            Display registered events
            """
            # TODO: Format this output
            for connector in self.connectors:
                debug(connector.name, connector.__events__)

        class VersionOutputFormat(AbstractOutputFormat):
            text = TEXT_FORMAT
            json = JSON_FORMAT

        @self.command()
        def version(
            short: bool = typer.Option(
                False,
                "--short",
                "-s",
                help="Display short version details",
                hidden=self.hide_servo_options,
            ),
            format: VersionOutputFormat = typer.Option(
                VersionOutputFormat.text, "--format", "-f", help="Select output format"
            ),
        ):
            """
            Display version
            """
            if short:
                if format == VersionOutputFormat.text:
                    typer.echo(f"{self.connector.name} v{self.connector.version}")
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": self.connector.name,
                        "version": str(self.connector.version),
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")
            else:
                if format == VersionOutputFormat.text:
                    typer.echo(
                        (
                            f"{self.connector.name} v{self.connector.version} ({self.connector.maturity})\n"
                            f"{self.connector.description}\n"
                            f"{self.connector.homepage}\n"
                            f"Licensed under the terms of {self.connector.license}"
                        )
                    )
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": self.connector.name,
                        "version": str(self.connector.version),
                        "maturity": str(self.connector.maturity),
                        "description": self.connector.description,
                        "homepage": self.connector.homepage,
                        "license": str(self.connector.license),
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")

            raise typer.Exit(0)


class ConnectorCLI(typer.Typer, SharedCommandsMixin):
    """
    ConnectorCLI is a subclass of typer.Typer that provides a CLI interface for
    connectors within the Servo assembly. 

    Actions common to all connectors are implemented directly on the class.
    Connectors can define their own actions within the Connector subclass.
    """

    connector: Connector

    @property
    def connectors(self) -> List["Connector"]:
        return [self.connector]

    def __init__(self, 
        connector: Connector,
        name: Optional[str] = None,
        help: Optional[str] = None,
        completion: CommandOption = False,
        **kwargs,
    ):
        self.connector = connector
        name = name if name is not None else connector.command_name
        help = help if help is not None else connector.description
        completion = completion if completion else False
        super().__init__(name=name, help=help, add_completion=completion, **kwargs)
        self.add_shared_commands()
        self.add_commands()

    ##
    # Convenience accessors

    @property
    def settings(self) -> ConnectorSettings:
        return self.connector.settings

    @property
    def optimizer(self) -> Optimizer:
        return self.connector.optimizer

    # Register connector specific commands
    def add_commands(self):
        @self.command(name="generate")
        def generate(
            defaults: bool = typer.Option(
                False,
                "--defaults",
                "-d",
                help="Include default values in the generated output",
            )
        ) -> None:
            """Generate a configuration file"""
            # TODO: Add force, output path, and format options
            exclude_unset = not defaults
            settings = self.connector.settings_model().generate()
            schema = json.loads(
                json.dumps(
                    {
                        self.connector.config_key_path: settings.dict(
                            by_alias=True, exclude_unset=exclude_unset
                        )
                    }
                )
            )
            output_path = Path.cwd() / f"{self.connector.command_name}.yaml"
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated {self.connector.command_name}.yaml")


class ServoCLI(typer.Typer, SharedCommandsMixin):
    assembly: ServoAssembly

    @property
    def optimizer(self) -> Optimizer:
        return self.servo.optimizer

    @property
    def connector(self) -> Servo:
        return self.servo

    @property
    def connectors(self) -> List[Connector]:
        return self.servo.connectors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hide_servo_options = False
        self.add_shared_commands(**kwargs)
        self.add_commands()

    def add_commands(self):
        @self.command()
        def console() -> None:
            """Open an interactive console"""
            # TODO: Load up the environment and trigger IPython
            typer.echo("Not yet implemented.", err=True)
            raise typer.Exit(2)

        @self.command()
        def new() -> None:
            """Creates a new servo assembly at [PATH]"""
            # TODO: Specify a list of connectors (or default to all)
            # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
            # TODO: Options for Docker Compose and Kubernetes?
            typer.echo("Not yet implemented.", err=True)
            raise typer.Exit(2)

        @self.command()
        def run(
            interactive: bool = typer.Option(
                False,
                "--interactive",
                "-i",
                help="Include models from all available connectors",
            )
        ) -> None:
            """Run the servo"""
            ServoRunner(self.servo, interactive=interactive).run()

        @self.command()
        def connectors(
            all: bool = typer.Option(
                False,
                "--all",
                "-a",
                help="Include models from all available connectors",
            ),
            verbose: bool = typer.Option(
                False, "--verbose", "-v", help="Display verbose info"
            ),
        ) -> None:
            """Display information about the assembly"""
            connectors = (
                self.assembly.all_connectors() if all else self.servo.connectors
            )
            headers = ["NAME", "VERSION", "DESCRIPTION"]
            row = [self.servo.name, self.servo.version, self.servo.description]
            if verbose:
                headers += ["HOMEPAGE", "MATURITY", "LICENSE"]
                row += [self.servo.homepage, self.servo.maturity, self.servo.license]
            table = [row]
            for connector in connectors:
                row = [connector.name, connector.version, connector.description]
                if verbose:
                    row += [connector.homepage, connector.maturity, connector.license]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @self.command(name="generate")
        def generate(
            defaults: bool = typer.Option(
                False,
                "--defaults",
                "-d",
                help="Include default values in the generated output",
            )
        ) -> None:
            """Generate a configuration file"""
            # TODO: Add force, output path, and format options

            exclude_unset = not defaults
            args = {}
            for c in self.assembly.all_connectors():
                args[c.__key_path__] = c.settings_model().generate()
            settings = self.assembly.settings_model(**args)

            # NOTE: We have to serialize through JSON first (not all fields serialize directly)
            schema = json.loads(
                json.dumps(settings.dict(by_alias=True, exclude_unset=exclude_unset))
            )
            output_path = Path.cwd() / f"{self.connector.command_name}.yaml"
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated {self.connector.command_name}.yaml")

        @self.callback()
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
                envvar="SERVO_CONFIG_FILE",
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
                raise typer.BadParameter(
                    "--token and --token-file cannot both be given"
                )

            if token_file is not None and token_file.exists():
                token = token_file.read_text()

            if len(token) == 0 or token.isspace():
                raise typer.BadParameter("token cannot be blank")

            optimizer = Optimizer(optimizer, token=token, base_url=base_url)

            # Assemble the Servo
            try:
                assembly, servo, ServoSettings = ServoAssembly.assemble(
                    config_file=config_file, optimizer=optimizer
                )
            except ValidationError as error:
                typer.echo(error, err=True)
                raise typer.Exit(2) from error

            # Hydrate our state
            self.assembly = assembly
            self.servo = servo
            self.settings = servo.settings

            # FIXME: Update the settings of our pre-registered connectors
            for connector, connector_cli in connectors_to_update:
                settings = getattr(servo.settings, connector.config_key_path)
                connector.settings = settings
                connector_cli.servo = servo


# Build the Typer CLI
cli = ServoCLI(name="servox", add_completion=True, no_args_is_help=True)

### Begin developer subcommands
# NOTE: registered as top level commands for convenience in dev

dev_typer = typer.Typer(name="dev", help="Developer utilities")


@dev_typer.command(name="test")
def developer_test() -> None:
    """Run automated tests"""
    __run(
        "pytest --cov=servo --cov=tests --cov-report=term-missing --cov-config=setup.cfg tests"
    )


@dev_typer.command(name="lint")
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


@dev_typer.command(name="format")
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


cli.add_typer(dev_typer)


def __run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)
