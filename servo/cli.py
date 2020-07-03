import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union, Set, Type, Callable

import typer
import yaml
from devtools import pformat
from pygments import highlight
from pygments.formatters import TerminalFormatter
from tabulate import tabulate

from pydantic import ValidationError
from pydantic.json import pydantic_encoder
from servo.connector import Connector, ConnectorSettings, Optimizer
from servo.servo import Events, Servo, ServoAssembly
from servo.servo_runner import ServoRunner
from servo.types import *

import click
from typer.models import CommandFunctionType, Default

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

# TODO: test passing callback as argument to command, via initializer for root callbacks
# TODO: Tests to write: check the classes we get (OrderedGroup, Command, Context)
# TODO: Test passing of correct context
# TODO: Test the ordering of commands on the root typer

# TODO: Print out better args when hit with debug()
class Context(typer.Context):
    """
    Context models state required by different CLI invocations.

    Hydration of the state if handled by callbacks on the `CLI` class.
    """

    # Basic configuration
    config_file: Optional[Path] = None
    optimizer: Optional[Optimizer] = None

    # Assembled servo
    assembly: Optional[ServoAssembly] = None
    servo: Optional[Servo] = None

    # Active connector
    connector: Optional[Connector] = None

    # TODO: Logger?

    def __init__(
        self,
        command: 'Command', # NOTE: NEEDS TO BE OUR CLASS
        *args,
        config_file: Optional[Path] = None,
        optimizer: Optional[Optimizer] = None,
        assembly: Optional[ServoAssembly] = None,
        servo: Optional[Servo] = None,
        connector: Optional[Connector] = None,
        **kwargs
    ):
        debug(command)
        self.config_file = config_file
        self.optimizer = optimizer
        self.assembly = assembly
        self.servo = servo
        self.connector = connector
        return super().__init__(command, *args, **kwargs)

class ContextMixin:
    # NOTE: Override the Click `make_context` base method to inject our class
    def make_context(self, info_name, args, parent=None, **extra):
        if parent and not issubclass(parent.__class__, Context):
            raise ValueError(f"Encountered an unexpected parent subclass type '{parent.__class__}' while attempting to create a context")

        for key, value in self.context_settings.items():
            if key not in extra:
                extra[key] = value

        debug("### Parent is context: ", parent, "self=", self)
        # if parent.__class__ == click.core.Context:
        #     raise ValueError("asdsad")
        if isinstance(parent, Context):            
            attributes = {"config_file", "optimizer", "assembly", "servo", "connector"}
            for attribute in attributes:
                if attribute not in extra:
                    extra[attribute] = getattr(parent, attribute)

        ctx = Context(self, info_name=info_name, parent=parent, **extra)
        with ctx.scope(cleanup=False):
            debug(ctx, parent, args, f"{self .__class__.__bases__}")
            self.parse_args(ctx, args)
        print("@@@@@ MY CONTEXT IS ", ctx, extra)
        return ctx

class Command(click.Command, ContextMixin):
    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)
    
    def main(self, *args, **kwargs):
        print("Alive")
        return super().main(*args, **kwargs)

class Group(click.Group, ContextMixin):
    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)

    # # NOTE: Rely on ordering of modern Python dicts
    # def list_commands(self, ctx):
    #     return self.commands

class OrderedGroup(Group, ContextMixin):
    # def make_context(self, info_name, args, parent=None, **extra):
    #     return ContextMixin.make_context(self, info_name, args, parent, **extra)

    # NOTE: Rely on ordering of modern Python dicts
    def list_commands(self, ctx):
        return self.commands

class CLI(typer.Typer):
    # CLI registry
    __clis__: Set["CLI"] = set()
        
    @classmethod
    def register(
        cls,
        context_selector: Optional[Union[Type[Optimizer], Type[Servo], Type[Connector]]] = None,
        *args, 
        **kwargs
    ):
        cli = cls(context_selector, *args, **kwargs)
        cls.__clis__.add(cli)

        @cli.callback()
        def connector_callback(context: Context):
            # TODO: Needs to handle other patterns
            for connector in context.servo.connectors:
                if isinstance(connector, context_selector):
                    context.connector = connector


        return cli

    def __init__(
        self, 
        context_selector: Optional[Union[Type[Optimizer], Type[Servo], Type[Connector]]] = None,
        *args,
        name: Optional[str] = None,
        help: Optional[str] = None,
        command_type: Optional[Type[click.Command]] = None, 
        callback: Optional[Callable] = None,
        **kwargs):
        callback = self.root_callback
        if context_selector is not None:            
            if issubclass(context_selector, Connector):
                if name is None:
                    name = context_selector.command_name() # TODO: This can just be turned into a class var property
                    # raise ValueError(f"sdasdsa: {context_selector}")
                if help is None:
                    help = context_selector.description
                if callback is None:
                    callback = self.root_callback
        
        # NOTE: Set default command class to get custom context
        if command_type is None:
            command_type = Command
        if callback is None:
            # TODO: Infer this from the `context_type`
            callback = self.root_callback
        debug("^^^ Initializing", command_type, callback)
        super().__init__(*args, name=name, help=help, cls=command_type, callback=callback, **kwargs) 
        
    def command(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        **kwargs,
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        # NOTE: Set default command class to get custom context
        if cls is None:
            cls = Command
        return super().command(*args, cls=cls, **kwargs)
    
    def callback(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        **kwargs,
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        # NOTE: Override the default to inject our Command class
        if cls is None:
            cls = Command
        print("INSIDE CALLBACK!!")
        return super().callback(*args, cls=cls, **kwargs)
    
    # TODO: Maybe name add_subcli?
    def add_cli(
        self,
        cli: "CLI",
        *args,
        cls: Optional[Type[click.Command]] = None,
        **kwargs,
    ) -> None:
        print("ADDING CLI!!")
        if not isinstance(cli, CLI):
            raise ValueError(f"Cannot add cli of type '{cli.__class__}: not a servo.cli.CLI")
        if cls is None:
            cls = Group
        return self.add_typer(cli, *args, cls=cls, **kwargs)
    
    # TODO: servo_callback, optimizer_callback, connector_callback, config_callback
    # TODO: Alias these options for reuse cli.OptimizerOption, cli.TokenOption, cli.ConfigFileOption
    @staticmethod
    def root_callback(
        ctx: Context,
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
        debug("!!! Context is", ctx)
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

        # Populate the context for use by other commands other 
        ctx.assembly = assembly
        ctx.servo = servo
        #ctx.settings = servo.settings

        debug(ctx, ctx.servo)

        # FIXME: Update the settings of our pre-registered connectors
        # for connector, connector_cli in connectors_to_update:
        #     settings = getattr(servo.settings, connector.config_key_path)
        #     connector.settings = settings
        #     connector_cli.servo = servo
        
        # def hello():
        #     print("FSDADA")
            
        # typer_click_object = typer.main.get_command(self)
        # debug(typer_click_object)
        # typer_click_object.add_command(hello, "hello")

class ServoCLI(CLI):
    """
    Provides the top-level commandline interface for interacting with the servo.
    """

    def __init__(
        self, 
        *args,
        name: Optional[str] = "servo", 
        command_type: Optional[Type[click.Command]] = None,
        add_completion: bool = True, 
        no_args_is_help: bool = True,
        **kwargs
    ) -> None:        
        # NOTE: We pass OrderedGroup to suppress sorting of commands alphabetically
        if command_type is None:
            command_type = OrderedGroup
        super().__init__(
            *args,
            Servo,
            command_type=command_type, 
            name=name, 
            add_completion=add_completion, 
            no_args_is_help=no_args_is_help,
            **kwargs
        )
        self.add_commands()
    
    def add_commands(self) -> None:
        self.add_core_commands()
        self.add_config_commands()
        self.add_connector_commands()
        self.add_misc_commands()

        @self.command()
        def foo(context: Context) -> None:
            """Foo"""
            debug(context)
            debug(context.servo)
            debug(context.optimizer)
            typer.echo("Foo!")
            raise typer.Exit(0)

        # Add command groups in order
        # self.add_core_commands()
        # self.add_config_commands()
        # self.add_assembly_commands()
    
    def add_core_commands(self) -> None:
        pass
    
    def add_config_commands(self) -> None:
        class SettingsOutputFormat(AbstractOutputFormat):
            yaml = YAML_FORMAT
            json = JSON_FORMAT
            dict = DICT_FORMAT
            text = TEXT_FORMAT

        @self.command()
        def settings(
            context: Context,
            format: SettingsOutputFormat = typer.Option(
                SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
            ),
            output: typer.FileTextWrite = typer.Option(
                None, "--output", "-o", help="Output settings to [FILE]"
            ),
        ) -> None:
            """Display the fully resolved settings"""
            debug("\n\n\n!! Called with context", context, context.servo)
            settings = context.servo.settings.dict(exclude_unset=True)
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
    
    def add_connector_commands(self) -> None:
        pass
        print("\n\n\nWTF!!", self.__clis__)
        for cli in self.__clis__:
            print(cli)
            # debug(cli)
            self.add_cli(cli)
    
    def add_misc_commands(self) -> None:
        pass

class SharedCommandsMixin:
    servo: Servo
    settings: ConnectorSettings
    connectors: List[Connector]
    hide_servo_options: bool = True

    def command_enabled(self, command: str, enabled: CommandOption) -> bool:
        debug(command, enabled,)
        return True
        if enabled in (True, False,):
            return enabled
        
        if enabled is not None:
            raise TypeError(f"Unexpected CommandOption value of '{enabled}' encountered: must be True, False, or None.")
        
        # Apply auto-discvery heuristics to determine if we should enable the command
        core_commands = {"version",}
        config_commands = {"settings", "schema", "validate", "generate",}
        metrics_commands = {"describe", "measure", "check"}
        components_commands = {"describe", "adjust", "promote", "check"}
        event_driven_commands = metrics_commands.union(components_commands)

        # Always enabled unless you opt out
        if command in core_commands:
            return True

        # If you have any configuration you'll need the basics
        if command in config_commands and self.connector.settings.__class__ != ConnectorSettings:
            return True
        
        # Use the event registry to determine intent
        if command in event_driven_commands:
            return self.connector.responds_to_event(command)

        return False

    def add_shared_commands(
        self,
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

        if self.command_enabled("settings", settings):
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

        if self.command_enabled("check", check):
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

        if self.command_enabled("describe", describe):
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
                            components_column.append(
                                f"{component.name}.{setting.name}={setting.value}"
                            )

                    metrics_column = []
                    for metric in description.metrics:
                        metrics_column.append(f"{metric.name} ({metric.unit})")

                    row = [
                        result.connector.name,
                        "\n".join(components_column),
                        "\n".join(metrics_column),
                    ]
                    table.append(row)

                typer.echo(tabulate(table, headers, tablefmt="plain"))

        class SchemaOutputFormat(AbstractOutputFormat):
            json = JSON_FORMAT
            text = TEXT_FORMAT
            dict = DICT_FORMAT
            html = HTML_FORMAT

        if self.command_enabled("schema", schema):
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

        if self.command_enabled("validate", validate):
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

        if self.command_enabled("events", events):
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

        if self.command_enabled("version", version):
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

# TODO: Connector gets an optional class method called cli_type
# TODO: when its loading you use event declarations to build the CLI arguments

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

    def __init__(
        self,
        connector: Connector,
        name: Optional[str] = None,
        help: Optional[str] = None,
        completion: CommandOption = False,
        **kwargs,
    ):
        self.connector = connector
        name = name if name is not None else connector.command_name()
        help = help if help is not None else connector.description
        completion = completion if completion else False
        super().__init__(name=name, help=help, add_completion=completion)
        # self.add_shared_commands(**kwargs)
        # self.add_commands(**kwargs)

    ##
    # Convenience accessors

    @property
    def settings(self) -> ConnectorSettings:
        return self.connector.settings

    @property
    def optimizer(self) -> Optimizer:
        return self.connector.optimizer

    # Register connector specific commands
    def add_commands(
        self,
        *,
        generate: CommandOption = False,
    ):
        if self.command_enabled("generate", generate):
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


class _ServoCLI(typer.Typer, SharedCommandsMixin):
    assembly: ServoAssembly
    servo: Servo

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
        # self.add_shared_commands(**kwargs)
        # stuff_callback()
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

            
            # self.add_shared_commands()
def new_servo_cli() -> ServoCLI:
    cli = ServoCLI()


    ### Begin developer subcommands
    # NOTE: registered as top level commands for convenience in dev

    dev_typer = typer.Typer(name="dev", help="Developer utilities")

    # @cli.callback(invoke_without_command=True)
    def testing(ctx: typer.Context):
        print("!!!! In here!")
        debug(ctx)
        for command_info in dev_typer.registered_commands:
            debug(command_info.name)
            if command_info.name == 'test':                
                print("Hiding!")
                dev_typer.hidden = True
            
            typer_click_object = typer.main.get_command(cli)
            debug(typer_click_object)
            def hello():
                print("FSDADA")
            typer_click_object.add_command(hello, "hello")
    
    @dev_typer.command(name="wtf")
    def thing():
        print('fffff')

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

    return cli


def __run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)
