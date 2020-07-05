import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union, Set, Type, Callable
from enum import Enum

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
from typer.models import CommandFunctionType, Default, DefaultPlaceholder

# Add the devtools debug() function to the CLI if its available
try:
    import builtins
    from devtools import debug
except ImportError:
    pass
else:
    builtins.debug = debug


class Section(str, Enum):
    ASSEMBLY = "Assembly Commands"
    OPS = "Operational Commands"
    CONFIG = "Configuration Commands"    
    CONNECTORS = "Connector Commands"
    COMMANDS = "Commands"
    OTHER = "Other Commands"    

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

    # NOTE: Section defaults generally only apply to Groups (see notes below)
    section: Section = Section.COMMANDS

    @classmethod
    def attributes(cls) -> Set[str]:
        """Returns the names of the attributes to be hydrated by ContextMixin"""
        return {"config_file", "optimizer", "assembly", "servo", "connector", "section"}

    def __init__(
        self,
        command: 'Command',
        *args,
        config_file: Optional[Path] = None,
        optimizer: Optional[Optimizer] = None,
        assembly: Optional[ServoAssembly] = None,
        servo: Optional[Servo] = None,
        connector: Optional[Connector] = None,
        section: Section = Section.COMMANDS,
        **kwargs
    ):
        self.config_file = config_file
        self.optimizer = optimizer
        self.assembly = assembly
        self.servo = servo
        self.connector = connector
        self.section = section
        return super().__init__(command, *args, **kwargs)

class ContextMixin:
    # NOTE: Override the Click `make_context` base method to inject our class
    def make_context(self, info_name, args, parent=None, **extra):
        if parent and not issubclass(parent.__class__, Context):
            raise ValueError(f"Encountered an unexpected parent subclass type '{parent.__class__}' while attempting to create a context")

        for key, value in self.context_settings.items():
            if key not in extra:
                extra[key] = value

        if isinstance(parent, Context):
            for attribute in Context.attributes():
                if attribute not in extra:
                    extra[attribute] = getattr(parent, attribute)

        ctx = Context(self, info_name=info_name, parent=parent, **extra)
        with ctx.scope(cleanup=False):
            self.parse_args(ctx, args)
        return ctx

class Command(click.Command, ContextMixin):
    @property
    def section(self) -> Optional[Section]:
        # NOTE: The `callback` property is the decorated function. See `command()` on CLI
        return getattr(self.callback, 'section', None)
        
    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)

class Group(click.Group, ContextMixin):
    @property
    def section(self) -> Optional[Section]:
        # NOTE: For Groups, Typer doesn't give us a great way to pass the state (can't decorate callback fn)
        # so instead we hang it on the context and rely on the command() to override it
        if self.context_settings:
            return self.context_settings.get('section', None)
        else:
            return None

    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)

    def format_commands(self, ctx, formatter):
        """
        Formats all commands into sections
        """

        sections_of_commands: Dict[Section, List[Tuple[str, Command]]] = {}
        for section in Section:
            sections_of_commands[section] = []

        for command_name in self.list_commands(ctx):
            command = self.get_command(ctx, command_name)
            if command.hidden:
                continue
            
            # Determine the command section
            # NOTE: We may have non-CLI instances so we guard attribute access
            section = getattr(command, 'section', Section.COMMANDS)

            commands = sections_of_commands.get(section, [])
            commands.append((command_name, command, ))
            sections_of_commands[section] = commands
    
        for section, commands in sections_of_commands.items():
            if len(commands) == 0:
                continue

            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

            rows = []
            for name, command in commands:
                help = command.get_short_help_str(limit)
                rows.append((name, help))
            
            with formatter.section(section):
                formatter.write_dl(rows)

class OrderedGroup(Group):
    # NOTE: Rely on ordering of modern Python dicts
    def list_commands(self, ctx):
        return self.commands

class CLI(typer.Typer):
    # CLI registry
    __clis__: Set["CLI"] = set()

    section: Section = Section.COMMANDS
    
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
        callback: Optional[Callable] = Default(None),
        section: Section = Section.COMMANDS,
        **kwargs):
        if context_selector is not None:
            if issubclass(context_selector, Connector):
                if name is None:
                    name = _command_name_from_config_key_path(context_selector.__key_path__)
                if help is None:
                    help = context_selector.description
                if isinstance(callback, DefaultPlaceholder):
                    callback = self.root_callback
        
        # NOTE: Set default command class to get custom context
        if command_type is None:
            command_type = Group
        if isinstance(callback, DefaultPlaceholder):
            callback = self.root_callback
        self.section = section
        super().__init__(*args, name=name, help=help, cls=command_type, callback=callback, **kwargs) 

    def command(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        section: Section = None,
        **kwargs,
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        # NOTE: Set default command class to get custom context & section support
        if cls is None:
            cls = Command
        
        # NOTE: This is a little fancy. We are decorating the function with the
        # section metadata and then returning the Typer decorator implementation
        parent_decorator = super().command(*args, cls=cls, **kwargs)
        def decorator(f: CommandFunctionType) -> CommandFunctionType:
            f.section = section if section else self.section
            return parent_decorator(f)
        
        return decorator
    
    def callback(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        **kwargs,
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        # NOTE: Override the default to inject our Command class
        if cls is None:
            cls = Group
        return super().callback(*args, cls=cls, **kwargs)
    
    def add_cli(
        self,
        cli: "CLI",
        *args,
        cls: Optional[Type[click.Command]] = None,
        section: Optional[Section] = None,
        context_settings: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ) -> None:
        if not isinstance(cli, CLI):
            raise ValueError(f"Cannot add cli of type '{cli.__class__}: not a servo.cli.CLI")
        if cls is None:
            cls = Group
        if context_settings is None:
            context_settings = {}
        section = section if section else cli.section
        # NOTE: Hang section state on the context for `Group` to pick up later
        context_settings['section'] = section
        return self.add_typer(cli, *args, cls=cls, context_settings=context_settings, **kwargs)
    
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

        # Populate the context for use by other commands 
        ctx.assembly = assembly
        ctx.servo = servo

class ServoCLI(CLI):
    """
    Provides the top-level commandline interface for interacting with the servo.
    """

    def __init__(
        self, 
        *args,
        name: Optional[str] = None,
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
    
    def _not_yet_implemented(self):
        typer.echo("error: not yet implemented", err=True)
        raise typer.Exit(2)
    
    def add_commands(self) -> None:
        self.add_core_commands()
        self.add_config_commands()
        self.add_assembly_commands()
        self.add_connector_commands()
        self.add_misc_commands()
    
    def add_assembly_commands(self) -> None:
        #     # TODO: Specify a list of connectors (or default to all)
        #     # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
        #     # TODO: Options for Docker Compose and Kubernetes?        
        @self.command(section=Section.ASSEMBLY)
        def new() -> None:
            """
            Create a new servo assembly
            """
            _not_yet_implemented()
        
        @self.command(section=Section.ASSEMBLY)
        def info() -> None:
            """
            Display info about the assembly
            """
            # TODO: Events, components, metrics
            # TODO: This needs to be a subgroup?
            _not_yet_implemented()
        
        # TODO: Where does this live?
        # TODO: Components, Metrics, Events
        # @self.command()
        # def events():
        #     """
        #     Display registered events
        #     """
        #     # TODO: Format this output
        #     for connector in self.connectors:
        #         debug(connector.name, connector.__events__)

        @self.command(section=Section.ASSEMBLY)
        def connectors(
            context: Context,
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
            """Manage connectors"""
            connectors = (
                context.assembly.all_connectors() if all else context.servo.connectors
            )
            headers = ["NAME", "VERSION", "DESCRIPTION"]
            row = [context.servo.name, context.servo.version, context.servo.description]
            if verbose:
                headers += ["HOMEPAGE", "MATURITY", "LICENSE"]
                row += [context.servo.homepage, context.servo.maturity, context.servo.license]
            table = [row]
            for connector in connectors:
                row = [connector.name, connector.version, connector.description]
                if verbose:
                    row += [connector.homepage, connector.maturity, connector.license]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))
        
        @self.command(section=Section.ASSEMBLY)
        def image() -> None:
            """
            Manage assembly container images
            """
            _not_yet_implemented()

    def add_core_commands(self, section=Section.OPS) -> None:        
        @self.command(section=section)
        def run(
            context: Context,
            interactive: bool = typer.Option(
                False,
                "--interactive",
                "-i",
                help="Include models from all available connectors",
            )
        ) -> None:
            """
            Run the servo
            """
            ServoRunner(context.servo, interactive=interactive).run()
        
        @self.command(section=section)
        def check(context: Context) -> None:
            """
            Check that the servo is ready to run
            """
            # TODO: Requires a config file
            # TODO: Run checks for all active connectors (or pick them)
            results: List[EventResult] = context.servo.dispatch_event(
                Events.CHECK, include=context.servo.connectors
            )
            headers = ["CONNECTOR", "CHECK", "STATUS", "COMMENT"]
            table = []
            for result in results:
                check: CheckResult = result.value
                status = "√ PASSED" if check.success else "X FAILED"
                row = [result.connector.name, check.name, status, check.comment]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        # TODO: Select one or more connectors, resource types
        # TODO: servo describe metrics, metrics/total_
        # TODO: Maybe it's just a describe -l
        @self.command(section=section)
        def describe(context: Context) -> None:
            """
            Display information about servo resources
            """
            results: List[EventResult] = context.servo.dispatch_event(
                Events.DESCRIBE, include=context.servo.connectors
            )
            # TODO: Include events, allow specifying in a list
            # TODO: Add --components --metrics OR 
            # TODO: Format output variously
            # TODO: This needs to actually run a describe op
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
        
        @self.command(section=section)
        def baseline() -> None:
            """
            Adjust settings to baseline configuration
            """
            _not_yet_implemented()

        @self.command(section=section)
        def measure() -> None:
            """
            Capture measurements for one or more metrics
            """
            _not_yet_implemented()
        
        @self.command(section=section)
        def adjust() -> None:
            """
            Adjust settings for one or more components
            """
            _not_yet_implemented()

        @self.command(section=section)
        def promote() -> None:
            """
            Promote optimized settings to the cluster
            """
            _not_yet_implemented()
    
    def add_config_commands(self, section=Section.CONFIG) -> None:
        class SettingsOutputFormat(AbstractOutputFormat):
            yaml = YAML_FORMAT
            json = JSON_FORMAT
            dict = DICT_FORMAT
            text = TEXT_FORMAT

        @self.command(section=section)
        def settings(
            context: Context,
            format: SettingsOutputFormat = typer.Option(
                SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
            ),
            output: typer.FileTextWrite = typer.Option(
                None, "--output", "-o", help="Output settings to [FILE]"
            ),
        ) -> None:
            """
            Display configured settings
            """
            settings = context.servo.settings.dict(exclude_unset=True)
            settings_json = json.dumps(settings, indent=2, default=pydantic_encoder)
            settings_dict = json.loads(settings_json)
            settings_dict_str = pformat(settings_dict)
            settings_yaml = yaml.dump(settings_dict, indent=2, sort_keys=True)

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
        
        class SchemaOutputFormat(AbstractOutputFormat):
            json = JSON_FORMAT
            text = TEXT_FORMAT
            dict = DICT_FORMAT
            html = HTML_FORMAT

        # TODO: Needs to support connector names, keys
        @self.command(section=section)
        def schema(
            context: Context,
            all: bool = typer.Option(
                False,
                "--all",
                "-a",
                help="Include models from all available connectors",
            ),
            top_level: bool = typer.Option(
                False,
                "--top-level",
                help="Emit a top-level schema (only connector models)",
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
                    output_data = context.assembly.top_level_schema_json(all=all)

                elif format == SchemaOutputFormat.dict:
                    output_data = pformat(context.assembly.top_level_schema(all=all))

            else:

                settings_class = context.servo.settings.__class__
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
        
        # TODO: Support connector selection
        @self.command(section=section)
        def validate(
            context: Context,
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
            ),
        ) -> None:
            """Validate a configuration file"""
            try:
                context.servo.settings_model().parse_file(file)
                typer.echo(f"√ Valid {context.servo.name} configuration in {file}")
            except (ValidationError, yaml.scanner.ScannerError) as e:
                typer.echo(f"X Invalid {context.servo.name} configuration in {file}")
                typer.echo(e, err=True)
                raise typer.Exit(1)
        
        # TODO: There is a duplicate command to untangle!
        # TODO: This should work with an incomplete config
        # TODO: Needs to be able to work with set of connector targets
        @self.command()
        def generate(
            context: Context,
            defaults: bool = typer.Option(
                False,
                "--defaults",
                "-d",
                help="Include default values in the generated output",
            )
        ) -> None:
            """Generate a configuration file"""
            # TODO: Add force, output path, and format options
            # TODO: When dumping specific connectors, need to use the config key path

            exclude_unset = not defaults
            settings = context.assembly.settings_model.generate()

            # NOTE: We have to serialize through JSON first (not all fields serialize directly to YAML)
            schema = json.loads(
                json.dumps(settings.dict(by_alias=True, exclude_unset=exclude_unset))
            )
            output_path = Path.cwd() / f"servo.yaml"
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated servo.yaml")
    
    def add_connector_commands(self) -> None:
        for cli in self.__clis__:
            self.add_cli(cli, section=Section.CONNECTORS)
    
    def add_misc_commands(self, section=Section.OTHER) -> None:
        class VersionOutputFormat(AbstractOutputFormat):
            text = TEXT_FORMAT
            json = JSON_FORMAT

        @self.command(section=section)
        def version(
            context: Context,
            short: bool = typer.Option(
                False,
                "--short",
                "-s",
                help="Display short version details",
            ),
            format: VersionOutputFormat = typer.Option(
                VersionOutputFormat.text, "--format", "-f", help="Select output format"
            ),
        ):
            """
            Display version
            """
            # TODO: Update to work with specific connectors
            if short:
                if format == VersionOutputFormat.text:
                    typer.echo(f"{context.servo.name} v{context.servo.version}")
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": context.servo.name,
                        "version": str(context.servo.version),
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")
            else:
                if format == VersionOutputFormat.text:
                    typer.echo(
                        (
                            f"{context.servo.name} v{context.servo.version} ({context.servo.maturity})\n"
                            f"{context.servo.description}\n"
                            f"{context.servo.homepage}\n"
                            f"Licensed under the terms of {context.servo.license}"
                        )
                    )
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": context.servo.name,
                        "version": str(context.servo.version),
                        "maturity": str(context.servo.maturity),
                        "description": context.servo.description,
                        "homepage": context.servo.homepage,
                        "license": str(context.servo.license),
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")

            raise typer.Exit(0)

def new_servo_cli() -> ServoCLI:
    cli = ServoCLI()

    ### Begin developer subcommands
    # NOTE: registered as top level commands for convenience in dev

    dev_typer = CLI(name="dev", help="Developer utilities")

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


    cli.add_cli(dev_typer, section=Section.OTHER)

    return cli


def __run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)

def _command_name_from_config_key_path(key_path: str) -> str:
    # foo.bar.this_key => this-key
    return key_path.split(".", 1)[-1].replace("_", "-").lower()
