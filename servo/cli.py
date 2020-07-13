import json
import shlex
import subprocess
import sys
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set, Type, Union

import click
import typer
import yaml
from bullet import Check, colors
from devtools import pformat
from loguru import logger
from pydantic import ValidationError
from pydantic.json import pydantic_encoder
from pygments import highlight
from pygments.formatters import TerminalFormatter
from tabulate import tabulate
from typer.models import CommandFunctionType, Default, DefaultPlaceholder

from servo.connector import Connector, Optimizer
from servo.events import EventHandler, Preposition
from servo.servo import (
    Events,
    Servo,
    ServoAssembly,
    _connector_class_from_string,
    _create_config_model,
    _create_config_model_from_routes,
    _default_routes,
)
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


class Section(str, Enum):
    ASSEMBLY = "Assembly Commands"
    OPS = "Operational Commands"
    CONFIG = "Configuration Commands"
    CONNECTORS = "Connector Commands"
    COMMANDS = "Commands"
    OTHER = "Other Commands"


class Context(typer.Context):
    """
    Context models state required by different CLI invocations.

    Hydration of the state if handled by callbacks on the `CLI` class.
    """

    # Basic configuration
    config_file: Optional[Path] = None
    optimizer: Optional[Optimizer] = None

    token: Optional[str] = None
    token_file: Optional[Path] = None
    base_url: Optional[str] = None

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
        return {
            "config_file",
            "optimizer",
            "assembly",
            "servo",
            "connector",
            "section",
            "token",
            "token_file",
            "base_url",
        }

    def __init__(
        self,
        command: "Command",
        *args,
        config_file: Optional[Path] = None,
        optimizer: Optional[Optimizer] = None,
        assembly: Optional[ServoAssembly] = None,
        servo: Optional[Servo] = None,
        connector: Optional[Connector] = None,
        section: Section = Section.COMMANDS,
        token: Optional[str] = None,
        token_file: Optional[Path] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.config_file = config_file
        self.optimizer = optimizer
        self.assembly = assembly
        self.servo = servo
        self.connector = connector
        self.section = section
        self.token = token
        self.token_file = token_file
        self.base_url = base_url
        return super().__init__(command, *args, **kwargs)


class ContextMixin:
    # NOTE: Override the Click `make_context` base method to inject our class
    def make_context(self, info_name, args, parent=None, **extra):
        if parent and not issubclass(parent.__class__, Context):
            raise ValueError(
                f"Encountered an unexpected parent subclass type '{parent.__class__}' while attempting to create a context"
            )

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
        return getattr(self.callback, "section", None)

    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)


class Group(click.Group, ContextMixin):
    @property
    def section(self) -> Optional[Section]:
        # NOTE: For Groups, Typer doesn't give us a great way to pass the state (can't decorate callback fn)
        # so instead we hang it on the context and rely on the command() to override it
        if self.context_settings:
            return self.context_settings.get("section", None)
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
            section = getattr(command, "section", Section.COMMANDS)

            commands = sections_of_commands.get(section, [])
            commands.append((command_name, command,))
            sections_of_commands[section] = commands

        for section, commands in sections_of_commands.items():
            if len(commands) == 0:
                continue

            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

            # Sort the connector and other commands as ordering isn't explicit
            if section in (Section.CONNECTORS, Section.OTHER,):
                commands = sorted(commands)

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
    section: Section = Section.COMMANDS

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        help: Optional[str] = None,
        command_type: Optional[Type[click.Command]] = None,
        callback: Optional[Callable] = Default(None),
        section: Section = Section.COMMANDS,
        **kwargs,
    ):

        # NOTE: Set default command class to get custom context
        if command_type is None:
            command_type = Group
        if isinstance(callback, DefaultPlaceholder):
            callback = self.root_callback
        self.section = section
        super().__init__(
            *args, name=name, help=help, cls=command_type, callback=callback, **kwargs
        )

    @property
    def logger(self) -> Logger:
        return logger

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
        self, *args, cls: Optional[Type[click.Command]] = None, **kwargs,
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
            raise ValueError(
                f"Cannot add cli of type '{cli.__class__}: not a servo.cli.CLI"
            )
        if cls is None:
            cls = Group
        if context_settings is None:
            context_settings = {}
        section = section if section else cli.section
        # NOTE: Hang section state on the context for `Group` to pick up later
        context_settings["section"] = section
        return self.add_typer(
            cli, *args, cls=cls, context_settings=context_settings, **kwargs
        )

    # TODO: servo_callback, optimizer_callback, connector_callback, config_callback
    # TODO: probably put these on a Callbacks class or something
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
        ctx.config_file = config_file
        ctx.optimizer = optimizer
        ctx.token = token
        ctx.token_file = token_file
        ctx.base_url = base_url

        # TODO: This should be pluggable
        if ctx.invoked_subcommand not in {"init", "schema", "generate", "validate"}:
            CLI.assemble_from_context(ctx)

    @staticmethod
    def assemble_from_context(ctx: Context):
        if ctx.optimizer is None:
            raise typer.BadParameter("An optimizer must be specified")

        # Resolve token
        if ctx.token is None and ctx.token_file is None:
            raise typer.BadParameter(
                "API token must be provided via --token (ENV['OPSANI_TOKEN']) or --token-file (ENV['OPSANI_TOKEN_FILE'])"
            )

        if ctx.token is not None and ctx.token_file is not None:
            raise typer.BadParameter("--token and --token-file cannot both be given")

        if ctx.token_file is not None and ctx.token_file.exists():
            ctx.token = ctx.token_file.read_text()

        if len(ctx.token) == 0 or ctx.token.isspace():
            raise typer.BadParameter("token cannot be blank")

        optimizer = Optimizer(ctx.optimizer, token=ctx.token, base_url=ctx.base_url)

        # Assemble the Servo
        try:
            assembly, servo, ServoSettings = ServoAssembly.assemble(
                config_file=ctx.config_file, optimizer=optimizer
            )
        except ValidationError as error:
            typer.echo(error, err=True)
            raise typer.Exit(2) from error

        # Populate the context for use by other commands
        ctx.assembly = assembly
        ctx.servo = servo

    @staticmethod
    def connectors_instance_callback(
        context: typer.Context, value: Optional[Union[str, List[str]]]
    ) -> Optional[Union[Connector, List[Connector]]]:
        """
        Transforms a one or more connector key-paths into Connector instances
        """
        if value:
            if isinstance(value, str):
                # Lookup by key
                for connector in context.servo.connectors:
                    if connector.config_key_path == value:
                        return connector
                raise typer.BadParameter(f"no connector found for key '{value}'")
            else:
                connectors: List[Connector] = []
                for key in value:
                    size = len(connectors)
                    for connector in context.servo.connectors:
                        if connector.config_key_path == key:
                            connectors.append(connector)
                            break
                    if len(connectors) == size:
                        raise typer.BadParameter(f"no connector found for key '{key}'")
                return connectors
        else:
            return None

    @staticmethod
    def connectors_type_callback(
        context: typer.Context, value: Optional[Union[str, List[str]]]
    ) -> Optional[Union[Type[Connector], List[Type[Connector]]]]:
        """
        Transforms a one or more connector key-paths into Connector types
        """
        if value:
            if isinstance(value, str):
                if connector := _connector_class_from_string(value):
                    return connector
                else:
                    raise typer.BadParameter(
                        f"no Connector type found for key '{value}'"
                    )
            else:
                connectors: List[Connector] = []
                for key in value:
                    if connector := _connector_class_from_string(key):
                        connectors.append(connector)
                    else:
                        raise typer.BadParameter(
                            f"no Connector type found for key '{key}'"
                        )
                return connectors
        else:
            return None

    @staticmethod
    def connector_routes_callback(
        context: typer.Context, value: Optional[List[str]]
    ) -> Optional[Dict[str, Type[Connector]]]:
        """
        Transforms a one or more connector key-paths into Connectors
        """
        if not value:
            return None

        routes: Dict[str, Type[Connector]] = {}
        for key in value:
            if ":" in key:
                # We have an alias descriptor
                key_path, identifier = key.split(":", 2)
            else:
                # Vanilla key-path or class name
                key_path = None
                identifier = key

            if connector_class := _connector_class_from_string(identifier):
                if key_path is None:
                    key_path = connector_class.__key_path__
                routes[key_path] = connector_class
            else:
                raise typer.BadParameter(f"no connector found for key '{identifier}'")

        return routes


class ConnectorCLI(CLI):
    connector_type: Type[Connector]

    # CLI registry
    __clis__: Set["CLI"] = set()

    def __init__(
        self,
        connector_type: Type[Connector],
        *args,
        name: Optional[str] = None,
        help: Optional[str] = None,
        command_type: Optional[Type[click.Command]] = None,
        callback: Optional[Callable] = Default(None),
        section: Section = Section.COMMANDS,
        **kwargs,
    ):
        # Register for automated inclusion in the ServoCLI
        ConnectorCLI.__clis__.add(self)

        # TODO: This will not find the right connector in aliased configurations
        # TODO: Use the subcommand name to find our instance
        def connector_callback(context: Context):
            for connector in context.servo.connectors:
                if isinstance(connector, connector_type):
                    context.connector = connector

        if name is None:
            name = _command_name_from_config_key_path(connector_type.__key_path__)
        if help is None:
            help = connector_type.description
        if isinstance(callback, DefaultPlaceholder):
            callback = connector_callback

        super().__init__(
            *args,
            name=name,
            help=help,
            command_type=command_type,
            callback=callback,
            section=section,
            **kwargs,
        )


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
        **kwargs,
    ) -> None:
        # NOTE: We pass OrderedGroup to suppress sorting of commands alphabetically
        if command_type is None:
            command_type = OrderedGroup
        super().__init__(
            *args,
            command_type=command_type,
            name=name,
            add_completion=add_completion,
            no_args_is_help=no_args_is_help,
            **kwargs,
        )
        self.add_commands()

    def _not_yet_implemented(self):
        typer.echo("error: not yet implemented", err=True)
        raise typer.Exit(2)

    def add_commands(self) -> None:
        self.add_ops_commands()
        self.add_config_commands()
        self.add_assembly_commands()
        self.add_connector_commands()
        self.add_other_commands()

    def add_assembly_commands(self) -> None:
        # TODO: Generate pyproject.toml, Dockerfile, README.md, LICENSE, and boilerplate
        # TODO: Options for Docker Compose and Kubernetes?

        @self.command(section=Section.ASSEMBLY)
        def init(context: Context) -> None:
            """
            Initialize a servo assembly
            """
            dotenv_file = Path(".env")
            write_dotenv = True
            if dotenv_file.exists():
                write_dotenv = typer.confirm(
                    f"File '{dotenv_file}' already exists. Overwrite it?"
                )

            if write_dotenv:
                optimizer = typer.prompt(
                    "Opsani optimizer? (format: dev.opsani.con/app-name)",
                    default=context.optimizer,
                )
                token = typer.prompt("API token?", default=context.token)
                dotenv_file.write_text(
                    f"OPSANI_OPTIMIZER={optimizer}\nOPSANI_TOKEN={token}\n"
                )
                typer.echo(".env file initialized")

            customize = typer.confirm(
                f"Generating servo.yaml. Do you want to select the connectors?"
            )
            if customize:
                check = Check(
                    "Which connectors do you want to activate? ",
                    choices=list(map(lambda c: c.name, ServoAssembly.all_connectors())),
                    check=" √",
                    margin=2,
                    check_color=colors.bright(colors.foreground["green"]),
                    check_on_switch=colors.bright(colors.foreground["green"]),
                    background_color=colors.background["black"],
                    background_on_switch=colors.background["white"],
                    word_color=colors.foreground["white"],
                    word_on_switch=colors.foreground["black"],
                )

                result = check.launch()
                connectors = list(
                    filter(
                        None,
                        map(
                            lambda c: c.__key_path__ if c.name in result else None,
                            ServoAssembly.all_connectors(),
                        ),
                    )
                )
            else:
                connectors = None

            typer_click_object = typer.main.get_group(self)
            context.invoke(
                typer_click_object.commands["generate"], connectors=connectors
            )

        @self.command(section=Section.ASSEMBLY, hidden=True)
        def new() -> None:
            # TODO: --dotenv --compose
            """
            Create a new servo assembly at [PATH]
            """
            _not_yet_implemented()

        show_cli = CLI(name="show", help="Display one or more resources")

        @show_cli.command()
        def components(context: Context) -> None:
            """
            Display adjustable components
            """
            results = context.servo.dispatch_event("components")
            headers = ["COMPONENT", "SETTINGS", "CONNECTOR"]
            table = []
            for result in results:
                result.value
                for component in result.value:
                    settings_list = sorted(
                        list(map(lambda s: s.__str__(), component.settings))
                    )
                    row = [
                        component.name,
                        "\n".join(settings_list),
                        result.connector.name,
                    ]
                    table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @show_cli.command()
        def events(
            context: Context,
            all: bool = typer.Option(
                None,
                "--all",
                "-a",
                help="Include models from all available connectors",
            ),
            by_connector: bool = typer.Option(
                None,
                "--by-connector",
                "-c",
                help="Display output by connector instead of event",
            ),
            before: bool = typer.Option(None, help="Display before event handlers",),
            on: bool = typer.Option(None, help="Display on event handlers",),
            after: bool = typer.Option(None, help="Display after event handlers",),
        ) -> None:
            """
            Display event handler info
            """
            event_handlers: List[EventHandler] = []
            connectors = (
                context.assembly.all_connectors() if all else context.servo.connectors
            )
            for connector in connectors:
                event_handlers.extend(connector.__event_handlers__)

            # If we have switched any on the preposition only include explicitly flagged
            preposition_switched = list(
                filter(lambda s: s is not None, (before, on, after))
            )
            if preposition_switched:
                if False in preposition_switched:
                    # Handle explicit exclusions
                    prepositions = [
                        Preposition.BEFORE,
                        Preposition.ON,
                        Preposition.AFTER,
                    ]
                    if before == False:
                        prepositions.remove(Preposition.BEFORE)
                    if on == False:
                        prepositions.remove(Preposition.ON)
                    if after == False:
                        prepositions.remove(Preposition.AFTER)
                else:
                    # Add explicit inclusions
                    prepositions = []
                    if before:
                        prepositions.append(Preposition.BEFORE)
                    if on:
                        prepositions.append(Preposition.ON)
                    if after:
                        prepositions.append(Preposition.AFTER)
            else:
                prepositions = [Preposition.BEFORE, Preposition.ON, Preposition.AFTER]

            sorted_event_names = sorted(
                list(set(map(lambda handler: handler.event.name, event_handlers)))
            )
            table = []

            if by_connector:
                headers = ["CONNECTOR", "EVENTS"]
                connector_types_by_name = dict(
                    map(
                        lambda handler: (handler.connector_type.__name__, connector,),
                        event_handlers,
                    )
                )
                sorted_connector_names = sorted(connector_types_by_name.keys())
                for connector_name in sorted_connector_names:
                    connector_types_by_name[connector_name]
                    event_labels = []
                    for event_name in sorted_event_names:
                        for preposition in prepositions:
                            handlers = list(
                                filter(
                                    lambda h: h.event.name == event_name
                                    and h.preposition == preposition
                                    and h.connector_type.__name__ == connector_name,
                                    event_handlers,
                                )
                            )
                            if handlers:
                                if preposition != Preposition.ON:
                                    event_labels.append(f"{preposition} {event_name}")
                                else:
                                    event_labels.append(event_name)

                    row = [connector_name, "\n".join(event_labels)]
                    table.append(row)
            else:
                headers = ["EVENT", "CONNECTORS"]
                for event_name in sorted_event_names:
                    for preposition in prepositions:
                        handlers = list(
                            filter(
                                lambda h: h.event.name == event_name
                                and h.preposition == preposition,
                                event_handlers,
                            )
                        )
                        if handlers:
                            sorted_connector_names = sorted(
                                list(
                                    set(
                                        map(
                                            lambda handler: handler.connector_type.__name__,
                                            handlers,
                                        )
                                    )
                                )
                            )
                            if preposition != Preposition.ON:
                                label = f"{preposition} {event_name}"
                            else:
                                label = event_name
                            row = [label, "\n".join(sorted(sorted_connector_names))]
                            table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @show_cli.command()
        def metrics(context: Context) -> None:
            """
            Display measurable metrics
            """
            metrics_to_connectors: Dict[str, tuple(str, Set[str])] = {}
            results = context.servo.dispatch_event("metrics")
            for result in results:
                for metric in result.value:
                    units_and_connectors = metrics_to_connectors.get(
                        metric.name, [metric.unit, set()]
                    )
                    units_and_connectors[1].add(result.connector.__class__.__name__)
                    metrics_to_connectors[metric.name] = units_and_connectors

            headers = ["METRIC", "UNIT", "CONNECTORS"]
            table = []
            for metric in sorted(metrics_to_connectors.keys()):
                units_and_connectors = metrics_to_connectors[metric]
                unit = units_and_connectors[0]
                unit_str = f"{unit.name} ({unit.value})"
                row = [metric, unit_str, "\n".join(sorted(units_and_connectors[1]))]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        self.add_cli(show_cli, section=Section.ASSEMBLY)

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
            if verbose:
                headers += ["HOMEPAGE", "MATURITY", "LICENSE"]
            table = []
            for connector in connectors:
                row = [connector.name, connector.version, connector.description]
                if verbose:
                    row += [connector.homepage, connector.maturity, connector.license]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

    def add_ops_commands(self, section=Section.OPS) -> None:
        @self.command(section=section)
        def run(
            context: Context,
            interactive: bool = typer.Option(
                False,
                "--interactive",
                "-i",
                help="Run in interactive mode (examine and confirm commands)",
            ),
        ) -> None:
            """
            Run the servo
            """
            ServoRunner(context.servo, interactive=interactive).run()

        def validate_connectors_respond_to_event(
            connectors: Iterable[Connector], event: str
        ) -> None:
            for connector in connectors:
                if not connector.responds_to_event(event):
                    raise typer.BadParameter(
                        f"connectors of type '{connector.__class__.__name__}' do not support checks (at key '{connector.config_key_path}')"
                    )

        @self.command(section=section)
        def check(
            context: Context,
            connectors: Optional[List[str]] = typer.Argument(
                None,
                help="The connectors to check",
                callback=self.connectors_instance_callback,
            ),
        ) -> None:
            """
            Check that the servo is ready to run
            """
            # TODO: Requires a config file

            # Validate that explicit args support check events
            if connectors:
                validate_connectors_respond_to_event(connectors, Events.CHECK)
            else:
                connectors = context.servo.connectors

            results: List[EventResult] = context.servo.dispatch_event(
                Events.CHECK, include=connectors
            )
            headers = ["CONNECTOR", "CHECK", "STATUS", "COMMENT"]
            table = []
            for result in results:
                check: CheckResult = result.value
                status = "√ PASSED" if check.success else "X FAILED"
                row = [result.connector.name, check.name, status, check.comment]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @self.command(section=section)
        def describe(
            context: Context,
            connectors: Optional[List[str]] = typer.Argument(
                None,
                help="The connectors to describe",
                callback=self.connectors_instance_callback,
            ),
        ) -> None:
            """
            Display current state of servo resources
            """

            # Validate that explicit args support describe events
            if connectors:
                validate_connectors_respond_to_event(connectors, Events.DESCRIBE)
            else:
                connectors = context.servo.connectors

            results: List[EventResult] = context.servo.dispatch_event(
                Events.DESCRIBE, include=connectors
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

        def metrics_callback(
            context: typer.Context, value: Optional[List[str]]
        ) -> Optional[List[Metric]]:
            if not value:
                return value

            all_metrics_by_name: Dict[str, Metric] = {}
            results = context.servo.dispatch_event("metrics")
            for result in results:
                for metric in result.value:
                    all_metrics_by_name[metric.name] = metric

            metrics: List[Metric] = []
            for metric_name in value:
                if metric := all_metrics_by_name.get(metric_name, None):
                    metrics.append(metric)
                else:
                    raise typer.BadParameter(f"no metric found named '{metric_name}'")

            return metrics

        @self.command(section=section, hidden=True)
        def baseline() -> None:
            """
            Adjust settings to baseline configuration
            """
            _not_yet_implemented()

        @self.command(section=section)
        def measure(
            context: Context,
            metrics: Optional[List[str]] = typer.Argument(
                None, help="The metrics to measure", callback=metrics_callback
            ),
        ) -> None:
            """
            Capture measurements for one or more metrics
            """
            # TODO: Limit the dispatch to the connectors that support the target metrics
            aggregate_measurement = Measurement.construct()
            results: List[EventResult] = context.servo.dispatch_event(
                Events.MEASURE, metrics=metrics, control=Control()
            )
            for result in results:
                measurement = result.value
                aggregate_measurement.readings.extend(measurement.readings)
                aggregate_measurement.annotations.update(measurement.annotations)

            metric_names = list(map(lambda m: m.name, metrics)) if metrics else None
            headers = ["METRIC", "UNIT", "READINGS"]
            table = []
            for reading in aggregate_measurement.readings:
                if metric_names is None or reading.metric.name in metric_names:
                    values = list(map(lambda r: f"{r[1]} @ {r[0]}", reading.values))
                    row = [
                        reading.metric.name,
                        reading.metric.unit,
                        "\n".join(values),
                    ]
                    table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @self.command(section=section)
        def adjust(
            context: Context,
            settings: Optional[List[str]] = typer.Argument(
                None,
                help="The settings to adjust (format is [COMPONENT].[SETTING=[VALUE])",
            ),
        ) -> None:
            """
            Adjust settings for one or more components
            """
            components: List[Component] = []
            for descriptor in settings:
                component_name, setting_descriptor = descriptor.split(".", 1)
                setting_name, value = setting_descriptor.split("=", 1)
                # TODO: This setting object is incomplete annd needs to be modeled
                setting = Setting.construct(name=setting_name, value=float(value))
                component = Component(name=component_name, settings=[setting])
                components.append(component)

            # TODO: Should be modeled directly as an adjustment instead of jamming into Description
            description = Description(components=components)
            results: List[EventResult] = context.servo.dispatch_event(
                Events.ADJUST, description.opsani_dict()
            )
            for result in results:
                adjustment = result.value
                status = adjustment.get("status", "undefined")

                if status == "ok":
                    self.logger.info(f"{result.connector.name} - Adjustment completed")
                else:
                    raise ConnectorError(
                        'Adjustment driver failed with status "{}" and message:\n{}'.format(
                            status, str(adjustment.get("message", "undefined"))
                        ),
                        status=status,
                        reason=adjustment.get("reason", "undefined"),
                    )

        @self.command(section=section, hidden=True)
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
        def config(
            context: Context,
            format: SettingsOutputFormat = typer.Option(
                SettingsOutputFormat.yaml, "--format", "-f", help="Select output format"
            ),
            output: typer.FileTextWrite = typer.Option(
                None, "--output", "-o", help="Output configuration to [FILE]"
            ),
            keys: Optional[List[str]] = typer.Argument(
                None, help="Display settings for specific keys"
            ),
        ) -> None:
            """
            Display configured settings
            """
            include = set(keys) if keys else None
            settings = context.servo.configuration.dict(
                exclude_unset=True, include=include
            )
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
            connector: Optional[str] = typer.Argument(
                None,
                help="Display schema for a specific connector by key or class name",
                callback=self.connectors_type_callback,
            ),
        ) -> None:
            """Display configuration schema"""
            if format == SchemaOutputFormat.text or format == SchemaOutputFormat.html:
                typer.echo("error: not yet implemented", err=True)
                raise typer.Exit(1)

            if top_level:
                CLI.assemble_from_context(context)

                if format == SchemaOutputFormat.json:
                    output_data = context.assembly.top_level_schema_json(all=all)

                elif format == SchemaOutputFormat.dict:
                    output_data = pformat(context.assembly.top_level_schema(all=all))

            else:

                if connector:
                    if isinstance(connector, Connector):
                        settings_class = connector.settings.__class__
                    elif issubclass(connector, Connector):
                        settings_class = connector.config_model()
                    else:
                        raise typer.BadParameter(
                            f"unexpected connector type '{connector.__class__.__name__}'"
                        )
                else:
                    CLI.assemble_from_context(context)
                    settings_class = context.servo.configuration.__class__
                if format == SchemaOutputFormat.json:
                    output_data = settings_class.schema_json(indent=2)
                elif format == SchemaOutputFormat.dict:
                    output_data = pformat(settings_class.schema())
                else:
                    raise RuntimeError(
                        f"no handler configured for output format {format}"
                    )

            assert output_data is not None, "output_data not assigned"

            if output:
                output.write(output_data)
            else:
                typer.echo(highlight(output_data, format.lexer(), TerminalFormatter()))

        # TODO: Specify connectors with `alias:connector` syntax for dictionary
        @self.command(section=section)
        def validate(
            context: Context,
            connectors: Optional[List[str]] = typer.Argument(
                None,
                metavar="CONNECTORS",
                help="Connectors to validate configuration for. \nFormats: `connector`, `ConnectorClass`, `alias:connector`, `alias:ConnectorClass`",
            ),
            file: Path = typer.Option(
                "servo.yaml",
                "--file",
                "-f",
                exists=True,
                file_okay=True,
                dir_okay=False,
                writable=False,
                readable=True,
                help="Output file to validate",
            ),
            quiet: bool = typer.Option(
                False, "--quiet", "-q", help="Do not echo generated output to stdout",
            ),
        ) -> None:
            """Validate a configuration"""
            try:
                # NOTE: When connector descriptor is provided the validation is constrained
                routes = self.connector_routes_callback(
                    context=context, value=connectors
                )
                config_model, routes = _create_config_model(
                    config_file=file, routes=routes
                )
                config_model.parse_file(file)
            except (ValidationError, yaml.scanner.ScannerError, KeyError) as e:
                if not quiet:
                    typer.echo(f"X Invalid configuration in {file}", err=True)
                    typer.echo(e, err=True)
                raise typer.Exit(1)

            if not quiet:
                typer.echo(f"√ Valid configuration in {file}")

        @self.command(section=section)
        def generate(
            context: Context,
            connectors: Optional[List[str]] = typer.Argument(
                None,
                metavar="CONNECTORS",
                help="Connectors to generate configuration for. \nFormats: `connector`, `ConnectorClass`, `alias:connector`, `alias:ConnectorClass`",
            ),
            file: Path = typer.Option(
                "servo.yaml",
                "--file",
                "-f",
                exists=False,
                file_okay=True,
                dir_okay=False,
                writable=True,
                readable=True,
                help="Output file to write",
            ),
            defaults: bool = typer.Option(
                False,
                "--defaults",
                "-d",
                help="Include default values in the generated output",
            ),
            standalone: bool = typer.Option(
                False,
                "--standalone",
                "-s",
                help="Exclude connectors descriptor in generated output",
            ),
            quiet: bool = typer.Option(
                False, "--quiet", "-q", help="Do not echo generated output to stdout",
            ),
            force: bool = typer.Option(
                False, "--force", help="Overwrite output file without prompting",
            ),
        ) -> None:
            """Generate a configuration"""
            exclude_unset = not defaults
            exclude = {"connectors"} if standalone else {}

            routes = (
                self.connector_routes_callback(context=context, value=connectors)
                if connectors
                else _default_routes()
            )

            # Build a settings model from our routes
            config_model = _create_config_model_from_routes(routes)
            settings = config_model.generate()

            if connectors and len(connectors):
                # Check is we have any aliases and assign dictionary
                connectors_dict: Dict[str, str] = {}
                aliased = False
                for identifier in connectors:
                    if ":" in identifier:
                        alias, id = identifier.split(":", 1)
                        connectors_dict[alias] = id
                        aliased = True
                    else:
                        connectors_dict[identifier] = identifier

                if aliased:
                    settings.connectors = connectors_dict
                else:
                    # If there are no aliases just assign input values
                    settings.connectors = connectors

            # NOTE: We have to serialize through JSON first (not all fields serialize directly to YAML)
            schema = json.loads(
                json.dumps(
                    settings.dict(
                        by_alias=True, exclude_unset=exclude_unset, exclude=exclude
                    )
                )
            )
            if file.exists() and force == False:
                delete = typer.confirm(f"File '{file}' already exists. Overwrite it?")
                if not delete:
                    raise typer.Abort()
            config = yaml.dump(schema)
            file.write_text(config)
            if not quiet:
                typer.echo(highlight(config, YamlLexer(), TerminalFormatter()))
                typer.echo(f"Generated {file}")

    def add_connector_commands(self) -> None:
        for cli in ConnectorCLI.__clis__:
            self.add_cli(cli, section=Section.CONNECTORS)

    def add_other_commands(self, section=Section.OTHER) -> None:
        # TODO: This should auto-detect if we are in a dev copy
        dev_cli = CLI(name="dev", help="Developer utilities", callback=None)

        @dev_cli.command()
        def test() -> None:
            """Run automated tests"""
            _run(
                "pytest --cov=servo --cov=tests --cov-report=term-missing --cov-config=setup.cfg tests"
            )

        @dev_cli.command()
        def lint() -> None:
            """Emit opinionated linter warnings and suggestions"""
            cmds = [
                "flake8 servo",
                "mypy servo",
                "black --check servo --diff",
                "isort --recursive --check-only servo",
            ]
            for cmd in cmds:
                _run(cmd)

        @dev_cli.command()
        def format() -> None:
            """Apply automatic formatting to the codebase"""
            cmds = [
                "isort --recursive  --force-single-line-imports servo tests",
                "autoflake --recursive --remove-all-unused-imports --remove-unused-variables --in-place servo tests",
                "black servo tests",
                "isort --recursive servo tests",
            ]
            for cmd in cmds:
                _run(cmd)

        self.add_cli(dev_cli, section=Section.OTHER)

        class VersionOutputFormat(AbstractOutputFormat):
            text = TEXT_FORMAT
            json = JSON_FORMAT

        @self.command(section=section)
        def version(
            context: Context,
            short: bool = typer.Option(
                False, "--short", "-s", help="Display short version details",
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


def _run(args: Union[str, List[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)


def _command_name_from_config_key_path(key_path: str) -> str:
    # foo.bar.this_key => this-key
    return key_path.split(".", 1)[-1].replace("_", "-").lower()
