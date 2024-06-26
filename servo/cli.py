# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import datetime
import enum
import functools
import json
import os
import pathlib
import shlex
import subprocess
import sys
import textwrap
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
)

import bullet
import click
import devtools
import kubernetes_asyncio
import kubernetes_asyncio.config
import loguru
import pydantic
import pygments
import pygments.formatters
import typer
import typer.core
import yaml

# Expose helpers
from tabulate import tabulate
from timeago import format as timeago

import servo
import servo.runner
import servo.utilities.yaml
import servo.utilities.strings

from servo.connectors.kubernetes_helpers import DeploymentHelper


class Section(enum.StrEnum):
    assembly = "Assembly Commands"
    ops = "Operational Commands"
    config = "Configuration Commands"
    connectors = "Connector Commands"
    commands = "Commands"
    other = "Other Commands"


class LogLevel(enum.StrEnum):
    trace = "TRACE"
    debug = "DEBUG"
    info = "INFO"
    success = "SUCCESS"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class ConfigOutputFormat(servo.AbstractOutputFormat):
    yaml = servo.YAML_FORMAT
    json = servo.JSON_FORMAT
    dict = servo.DICT_FORMAT
    text = servo.TEXT_FORMAT
    configmap = servo.CONFIGMAP_FORMAT


class SchemaOutputFormat(servo.AbstractOutputFormat):
    json = servo.JSON_FORMAT
    text = servo.TEXT_FORMAT
    dict = servo.DICT_FORMAT
    html = servo.HTML_FORMAT


class VersionOutputFormat(servo.AbstractOutputFormat):
    text = servo.TEXT_FORMAT
    json = servo.JSON_FORMAT


# FIXME: Eliminate the mixin and put our context object onto the Click.obj instance
class Context(typer.Context):
    """
    Context models state required by different CLI invocations.

    Hydration of the state if handled by callbacks on the `CLI` class.
    """

    # Basic configuration
    config_file: Optional[pathlib.Path] = None
    name: Optional[str] = None
    limit: Optional[int] = None

    # Assembled servo
    assembly: Optional[servo.Assembly] = None
    servo_: Optional[servo.Servo] = None

    # Active connector
    connector: Optional[servo.BaseConnector] = None

    # NOTE: Section defaults generally only apply to Groups (see notes below)
    section: Section = Section.commands

    @classmethod
    def attributes(cls) -> Set[str]:
        """Returns the names of the attributes to be hydrated by ContextMixin"""
        return {
            "config_file",
            "name",
            "assembly",
            "servo_",
            "connector",
            "section",
            "limit",
        }

    @property
    def servo(self) -> servo.Servo:
        return self.servo_

    def __init__(
        self,
        command: "Command",
        *args,
        config_file: Optional[pathlib.Path] = None,
        name: Optional[str] = None,
        assembly: Optional[servo.Assembly] = None,
        servo_: Optional[servo.Servo] = None,
        connector: Optional[servo.BaseConnector] = None,
        section: Section = Section.commands,
        limit: Optional[int] = None,
        **kwargs,
    ) -> None:  # noqa: D107
        self.config_file = config_file
        self.name = name
        self.assembly = assembly
        self.servo_ = servo_
        self.connector = connector
        self.section = section
        self.limit = limit
        super().__init__(command, *args, **kwargs)


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


class Command(typer.core.TyperCommand, ContextMixin):
    @property
    def section(self) -> Optional[Section]:
        # NOTE: The `callback` property is the decorated function. See `command()` on CLI
        return getattr(self.callback, "section", None)

    def make_context(self, info_name, args, parent=None, **extra):
        return ContextMixin.make_context(self, info_name, args, parent, **extra)


class Group(typer.core.TyperGroup, ContextMixin):
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

        sections_of_commands: dict[Section, list[Tuple[str, Command]]] = {}
        for section in Section:
            sections_of_commands[section] = []

        for command_name in self.list_commands(ctx):
            command = self.get_command(ctx, command_name)
            if command.hidden:
                continue

            # Determine the command section
            # NOTE: We may have non-CLI instances so we guard attribute access
            section = getattr(command, "section", Section.commands)

            commands = sections_of_commands.get(section, [])
            commands.append(
                (
                    command_name,
                    command,
                )
            )
            sections_of_commands[section] = commands

        for section, commands in sections_of_commands.items():
            if len(commands) == 0:
                continue

            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

            # Sort the connector and other commands as ordering isn't explicit
            if section in (
                Section.connectors,
                Section.other,
            ):
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


class CLI(typer.Typer, servo.logging.Mixin):
    section: Section = Section.commands

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        help: Optional[str] = None,
        command_type: Optional[Type[click.Command]] = None,
        callback: Optional[Callable] = typer.models.Default(None),
        section: Section = Section.commands,
        **kwargs,
    ) -> None:  # noqa: D107
        # NOTE: Set default command class to get custom context
        if command_type is None:
            command_type = Group
        if isinstance(callback, typer.models.DefaultPlaceholder):
            callback = self.root_callback
        self.section = section
        super().__init__(
            *args,
            name=name,
            help=help,
            cls=command_type,
            callback=callback,
            pretty_exceptions_enable=False,
            **kwargs,
        )

    def command(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        section: Section = None,
        **kwargs,
    ) -> Callable[[typer.models.CommandFunctionType], typer.models.CommandFunctionType]:
        # NOTE: Set default command class to get custom context & section support
        if cls is None:
            cls = Command

        # NOTE: This is a little fancy. We are decorating the function with the
        # section metadata and then returning the Typer decorator implementation
        parent_decorator = super().command(*args, cls=cls, **kwargs)

        def decorator(
            f: typer.models.CommandFunctionType,
        ) -> typer.models.CommandFunctionType:
            f.section = section if section else self.section
            return parent_decorator(f)

        return decorator

    def callback(
        self,
        *args,
        cls: Optional[Type[click.Command]] = None,
        **kwargs,
    ) -> Callable[[typer.models.CommandFunctionType], typer.models.CommandFunctionType]:
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
        context_settings: Optional[dict[Any, Any]] = None,
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

    @staticmethod
    def root_callback(
        ctx: Context,
        config_file: pathlib.Path = typer.Option(
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
        name: Optional[str] = typer.Option(
            None,
            "--name",
            "-n",
            envvar="SERVO_NAME",
            show_envvar=True,
            help="Name of the servo to use",
        ),
        limit: Optional[int] = typer.Option(
            None,
            "--limit",
            help="Limit multi-servo concurrency",
        ),
        log_level: LogLevel = typer.Option(
            LogLevel.info,
            "--log-level",
            "-l",
            envvar="SERVO_LOG_LEVEL",
            show_envvar=True,
            help="Set the log level",
        ),
        no_color: Optional[bool] = typer.Option(
            None,
            "--no-color",
            envvar=["SERVO_NO_COLOR", "NO_COLOR"],
            help="Disable colored output",
        ),
    ):
        ctx.config_file = config_file
        ctx.name = name
        ctx.limit = limit
        servo.logging.set_level(log_level)
        servo.logging.set_colors(not no_color)

        # TODO: This should be pluggable. Base it off of the section?
        if ctx.invoked_subcommand not in {
            "init",
            "connectors",
            "schema",
            "generate",
            "validate",
            "version",
        }:
            CLI.assemble_from_context(ctx)

    @staticmethod
    def assemble_from_context(ctx: Context) -> None:
        if ctx.config_file is None:
            raise typer.BadParameter("Config file must be specified")

        if not ctx.config_file.exists():
            raise typer.BadParameter(f"Config file '{ctx.config_file}' does not exist")

        # Conditionalize based on multi-servo options
        optimizer = None
        configs = list(yaml.full_load_all(open(ctx.config_file)))
        if not isinstance(configs, list):
            raise TypeError(
                f'error: config file "{ctx.config_file}" parsed to an unexpected value of type "{configs.__class__}"'
            )

        if len(configs) == 0:
            configs.append({})

        for config in configs:
            if not isinstance(config, dict):
                raise TypeError(
                    f'error: config file "{ctx.config_file}" parsed to an unexpected value of type "{config.__class__}"'
                )

        # Assemble the Servo
        assembly: servo.Assembly = run_async(
            servo.Assembly.assemble(config_file=ctx.config_file, configs=configs)
        )

        # Target a specific servo if possible
        ctx.assembly = assembly
        if assembly.servos:
            if len(assembly.servos) == 1:
                ctx.servo_ = assembly.servos[0]

                if ctx.name and ctx.servo_.name != ctx.name:
                    raise typer.BadParameter(f'No servo was found named "{ctx.name}"')

            elif ctx.name:
                for servo_ in assembly.servos:
                    if servo_.name == ctx.name:
                        ctx.servo_ = servo_
                        break

                if ctx.servo_ is None:
                    raise typer.BadParameter(f'No servo was found named "{ctx.name}"')

    @staticmethod
    def connectors_named(
        names: list[str], servo_: servo.Servo
    ) -> list[servo.BaseConnector]:
        connectors: list[servo.BaseConnector] = []
        for name in names:
            size = len(connectors)
            for connector in servo_.all_connectors:
                if connector.name == name:
                    connectors.append(connector)
                    break

            if len(connectors) == size:
                raise typer.BadParameter(f"no connector found named '{name}'")

        return connectors

    @staticmethod
    def connectors_type_callback(
        context: typer.Context, value: Optional[Union[str, list[str]]]
    ) -> Optional[Union[Type[servo.BaseConnector], list[Type[servo.BaseConnector]]]]:
        """
        Transforms a one or more connector key-paths into Connector types
        """
        if value:
            if isinstance(value, str):
                if connector := servo.connector._connector_class_from_string(value):
                    return connector
                else:
                    raise typer.BadParameter(
                        f"no Connector type found for key '{value}'"
                    )
            else:
                connectors: list[Type[servo.BaseConnector]] = []
                for key in value:
                    if connector := servo.connector._connector_class_from_string(key):
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
        context: typer.Context, value: Optional[list[str]]
    ) -> Optional[dict[str, Type[servo.BaseConnector]]]:
        """
        Transforms a one or more connector descriptors into a dict of names to Connectors
        """
        if not value:
            return None

        routes: dict[str, Type[servo.BaseConnector]] = {}
        for key in value:
            if ":" in key:
                # We have an alias descriptor
                name, identifier = key.split(":", 2)
            else:
                # Vanilla key-path or class name
                name = None
                identifier = key

            if connector_class := servo.connector._connector_class_from_string(
                identifier
            ):
                if name is None:
                    name = connector_class.__default_name__
                routes[name] = connector_class
            else:
                raise typer.BadParameter(
                    f"no connector found for identifier '{identifier}'"
                )

        return routes

    @staticmethod
    def duration_callback(
        context: typer.Context, value: Optional[str]
    ) -> Optional[servo.Duration]:
        """
        Transform a string into a Duration object.

        Parses duration strings.
        """
        if not value:
            return None

        try:
            return servo.Duration(value)
        except ValueError as e:
            raise typer.BadParameter(f"invalid duration parameter: {e}") from e


class ConnectorCLI(CLI):
    connector_type: Type[servo.BaseConnector]

    # CLI registry
    __clis__: Set["CLI"] = set()

    def __init__(
        self,
        connector_type: Type[servo.BaseConnector],
        *args,
        name: Optional[str] = None,
        help: Optional[str] = None,
        command_type: Optional[Type[click.Command]] = None,
        callback: Optional[Callable] = typer.models.Default(None),
        section: Section = Section.commands,
        **kwargs,
    ) -> None:  # noqa: D107
        # Register for automated inclusion in the ServoCLI
        ConnectorCLI.__clis__.add(self)

        def connector_callback(
            context: Context,
            connector: Optional[str] = typer.Option(
                None,
                "--connector",
                "-c",
                metavar="CONNECTOR",
                help="Connector to activate",
            ),
        ) -> None:
            if context.servo is None:
                raise typer.BadParameter(f"A servo must be selected")

            instances = list(
                filter(
                    lambda c: isinstance(c, connector_type), context.servo.connectors
                )
            )
            instance_count = len(instances)
            if instance_count == 0:
                raise typer.BadParameter(
                    f'no instances of "{connector_type.__name__}" are active the in servo "{context.servo.name}"'
                )
            elif instance_count == 1:
                context.connector = instances[0]
            else:
                names = []
                for instance in instances:
                    if instance.name == connector:
                        context.connector = instance
                        break
                    names.append(instance.name)

                if context.connector is None:
                    if connector is None:
                        raise typer.BadParameter(
                            f'multiple instances of "{connector_type.__name__}" found in servo "{context.servo.name}": select one of {repr(names)}'
                        )
                    else:
                        raise typer.BadParameter(
                            f'no connector named "{connector}" of type "{connector_type.__name__}" found in servo "{context.servo.name}": select one of {repr(names)}'
                        )

        if name is None:
            name = servo.utilities.strings.commandify(connector_type.__default_name__)
        if help is None:
            help = connector_type.description
        if isinstance(callback, typer.models.DefaultPlaceholder):
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
    ) -> None:  # noqa: D107
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

    def add_commands(self) -> None:
        self.add_ops_commands()
        self.add_config_commands()
        self.add_assembly_commands()
        self.add_connector_commands()

    @property
    def logger(self) -> loguru.Logger:
        return servo.logger

    def add_assembly_commands(self) -> None:
        @self.command(section=Section.assembly)
        def init(
            context: Context,
            dotenv: bool = typer.Option(
                True,
                help="Generate .env file",
            ),
        ) -> None:
            """
            Initialize a servo assembly
            """
            if dotenv:
                dotenv_file = pathlib.Path(".env")
                write_dotenv = True
                if dotenv_file.exists():
                    write_dotenv = typer.confirm(
                        f"File '{dotenv_file}' already exists. Overwrite it?"
                    )

                if write_dotenv:
                    default_optimizer = os.environ.get("OPSANI_OPTIMIZER")
                    optimizer = typer.prompt(
                        "Opsani optimizer? (format: dev.opsani.com/app-name)",
                        default=default_optimizer,
                    )
                    optimizer != default_optimizer or typer.echo()
                    default_token = os.environ.get("OPSANI_TOKEN")
                    token = typer.prompt("API token?", default=default_token)
                    token != default_token or typer.echo()
                    dotenv_file.write_text(
                        f"OPSANI_OPTIMIZER={optimizer}\nOPSANI_TOKEN={token}\nSERVO_LOG_LEVEL=DEBUG\n"
                    )
                    typer.echo(".env file initialized")

            customize = typer.confirm(
                f"Generating servo.yaml. Do you want to select the connectors?"
            )
            if customize:
                types = servo.Assembly.all_connector_types()
                types.remove(servo.Servo)

                check = bullet.Check(
                    "\nWhich connectors do you want to activate? [space to (de)select]",
                    choices=list(map(lambda c: c.name, types)),
                    check=" √",
                    indent=0,
                    margin=4,
                    align=4,
                    pad_right=4,
                    check_color=bullet.colors.bright(bullet.colors.foreground["green"]),
                    check_on_switch=bullet.colors.bright(
                        bullet.colors.foreground["black"]
                    ),
                    background_color=bullet.colors.background["black"],
                    background_on_switch=bullet.colors.background["white"],
                    word_color=bullet.colors.foreground["white"],
                    word_on_switch=bullet.colors.foreground["black"],
                )

                result = check.launch()
                connectors = list(
                    filter(
                        None,
                        map(
                            lambda c: c.__default_name__ if c.name in result else None,
                            servo.Assembly.all_connector_types(),
                        ),
                    )
                )
            else:
                connectors = []

            typer_click_object = typer.main.get_group(self)
            context.invoke(
                typer_click_object.commands["generate"], connectors=connectors
            )

        show_cli = CLI(name="show", help="Display one or more resources")

        @show_cli.command()
        def connectors(
            context: Context,
            verbose: bool = typer.Option(
                False, "--verbose", "-v", help="Display verbose info"
            ),
        ) -> None:
            """Manage connectors"""

            headers = ["NAME", "TYPE", "VERSION", "DESCRIPTION"]
            if verbose:
                headers += ["HOMEPAGE", "MATURITY", "LICENSE"]

            for servo_ in context.assembly.servos:
                if context.servo_ and context.servo_ != servo_:
                    continue

                connectors = servo_.all_connectors
                table = []
                connectors_by_type = {}
                for c in connectors:
                    c_type = c.__class__ if isinstance(c, servo.BaseConnector) else c
                    c_list = connectors_by_type.get(c_type, [])
                    c_list.append(c)
                    connectors_by_type[c_type] = c_list

                for connector_type in connectors_by_type.keys():
                    names = list(
                        map(lambda c: c.name, connectors_by_type[connector_type])
                    )
                    row = [
                        "\n".join(names),
                        connector_type.name,
                        connector_type.version,
                        connector_type.description,
                    ]
                    if verbose:
                        row += [
                            connector_type.homepage,
                            connector_type.maturity,
                            connector_type.license,
                        ]
                    table.append(row)

                if len(context.assembly.servos) > 1:
                    typer.echo(f"{servo_.name}")
                typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

        @show_cli.command()
        def components(context: Context) -> None:
            """Display adjustable components"""
            for servo_ in context.assembly.servos:
                if context.servo_ and context.servo_ != servo_:
                    continue

                results = run_async(servo_.dispatch_event(servo.Events.components))
                headers = ["COMPONENT", "SETTINGS", "CONNECTOR"]
                table = []
                for result in results:
                    for component in result.value:
                        settings_list = sorted(
                            list(
                                map(
                                    lambda s: f"{s.name}={s.human_readable_value} {s.summary()}",
                                    component.settings,
                                )
                            )
                        )
                        row = [
                            component.name,
                            "\n".join(settings_list),
                            result.connector.name,
                        ]
                        table.append(row)

                    if len(context.assembly.servos) > 1:
                        typer.echo(f"{servo_.name}")
                    typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

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
            before: bool = typer.Option(
                None,
                help="Display before event handlers",
            ),
            on: bool = typer.Option(
                None,
                help="Display on event handlers",
            ),
            after: bool = typer.Option(
                None,
                help="Display after event handlers",
            ),
        ) -> None:
            """
            Display event handler info
            """
            for servo_ in context.assembly.servos:
                if context.servo and context.servo != servo_:
                    continue

                event_handlers: list[servo.EventHandler] = []
                connectors = (
                    context.assembly.all_connector_types()
                    if all
                    else servo_.all_connectors
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
                            servo.Preposition.before,
                            servo.Preposition.on,
                            servo.Preposition.after,
                        ]
                        if before == False:
                            prepositions.remove(servo.Preposition.before)
                        if on == False:
                            prepositions.remove(servo.Preposition.on)
                        if after == False:
                            prepositions.remove(servo.Preposition.after)
                    else:
                        # Add explicit inclusions
                        prepositions = []
                        if before:
                            prepositions.append(servo.Preposition.before)
                        if on:
                            prepositions.append(servo.Preposition.on)
                        if after:
                            prepositions.append(servo.Preposition.after)
                else:
                    prepositions = [
                        servo.Preposition.before,
                        servo.Preposition.on,
                        servo.Preposition.after,
                    ]

                sorted_event_names = sorted(
                    list(set(map(lambda handler: handler.event.name, event_handlers)))
                )
                table = []

                if by_connector:
                    headers = ["CONNECTOR", "EVENTS"]
                    connector_types_by_name = dict(
                        map(
                            lambda handler: (
                                handler.connector_type.name,
                                connector,
                            ),
                            event_handlers,
                        )
                    )
                    sorted_connector_names = sorted(connector_types_by_name.keys())
                    for connector_name in sorted_connector_names:
                        event_labels = []
                        for event_name in sorted_event_names:
                            for preposition in prepositions:
                                handlers = list(
                                    filter(
                                        lambda h: h.event.name == event_name
                                        and h.preposition == preposition
                                        and h.connector_type.name == connector_name,
                                        event_handlers,
                                    )
                                )
                                if handlers:
                                    if preposition != servo.Preposition.on:
                                        event_labels.append(
                                            f"{preposition} {event_name}"
                                        )
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
                                                lambda handler: handler.connector_type.name,
                                                handlers,
                                            )
                                        )
                                    )
                                )
                                if preposition != servo.Preposition.on:
                                    label = f"{preposition} {event_name}"
                                else:
                                    label = event_name
                                row = [label, "\n".join(sorted(sorted_connector_names))]
                                table.append(row)

                if len(context.assembly.servos) > 1:
                    typer.echo(f"{servo_.name}")
                typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

        @show_cli.command()
        def metrics(context: Context) -> None:
            """
            Display measurable metrics
            """
            for servo_ in context.assembly.servos:
                if context.servo and context.servo != servo_:
                    continue

                metrics_to_connectors: dict[str, Tuple[str, Set[str]]] = {}
                results = run_async(servo_.dispatch_event("metrics"))
                for result in results:
                    for metric in result.value:
                        units_and_connectors = metrics_to_connectors.get(
                            metric.name, [metric.unit, set()]
                        )
                        units_and_connectors[1].add(result.connector.__class__.name)
                        metrics_to_connectors[metric.name] = units_and_connectors

                headers = ["METRIC", "UNIT", "CONNECTORS"]
                table = []
                for metric in sorted(metrics_to_connectors.keys()):
                    units_and_connectors = metrics_to_connectors[metric]
                    unit = units_and_connectors[0]
                    unit_str = f"{unit.name} ({unit.value})"
                    row = [metric, unit_str, "\n".join(sorted(units_and_connectors[1]))]
                    table.append(row)

                if len(context.assembly.servos) > 1:
                    typer.echo(f"{servo_.name}")
                typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

        self.add_cli(show_cli, section=Section.assembly)

        @self.command("list", section=Section.assembly)
        def list_(
            context: Context,
        ) -> None:
            """List servos in the assembly"""
            headers = ["NAME", "OPTIMIZER", "DESCRIPTION"]
            table = []

            for servo_ in context.assembly.servos:
                row = [
                    servo_.name,
                    servo_.optimizer.id,
                    servo_.description or "-",
                ]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain"))

        @self.command(section=Section.assembly)
        def connectors(
            context: Context,
            verbose: bool = typer.Option(
                False, "--verbose", "-v", help="Display verbose info"
            ),
        ) -> None:
            """Display active connectors"""
            table = []
            headers = ["NAME", "VERSION", "DESCRIPTION"]
            if verbose:
                headers += ["HOMEPAGE", "MATURITY", "LICENSE"]

            for connector_type in servo.Assembly.all_connector_types():
                row = [
                    connector_type.__default_name__,
                    str(connector_type.version),
                    connector_type.description,
                ]
                if verbose:
                    row += [
                        connector_type.homepage,
                        connector_type.maturity,
                        connector_type.license,
                    ]
                table.append(row)

            typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

    def add_ops_commands(self, section=Section.ops) -> None:
        @self.command(section=section)
        def run(
            context: Context,
            check: bool = typer.Option(
                False,
                "--check",
                "-c",
                help="Verify all checks pass before running",
                envvar="SERVO_RUN_CHECK",
            ),
            no_poll: Optional[bool] = typer.Option(
                None,
                "--no-poll",
                help="Do not poll the Opsani API for commands",
            ),
            interactive: Optional[bool] = typer.Option(
                None,
                "--interactive",
                "-i",
                help="Ask for confirmation before executing operations",
            ),
            dry_run: Optional[bool] = typer.Option(
                False,
                "--dry-run",
                "-d",
                help="Execute checks without running servo",
            ),
            debug: bool = typer.Option(
                False,
                "--debug",
                "--DEBUG",
                help="Prevent exception handling on running event loop to allow error propogation",
                envvar="SERVO_RUN_DEBUG",
            ),
        ) -> None:
            """
            Run the servo
            """

            if not context.assembly:
                raise typer.Abort("failed to assemble servo")

            if check or dry_run:
                # Check all targeted servos
                def print_callback(input: str) -> None:
                    typer.echo(input)

                try:
                    if context.servo:
                        ready = run_async(context.servo.check_servo(print_callback))
                    else:
                        # gather() expects a loop to exist at invocation time which is not compatible with the run_async
                        #  execution model. wrap the gather in a standard async function to work around this
                        async def gather_checks():
                            async with asyncio.TaskGroup() as tg:
                                tasks = [
                                    tg.create_task(s.check_servo(print_callback))
                                    for s in context.assembly.servos
                                ]
                            return (t.result() for t in tasks)

                        results = run_async(gather_checks())
                        ready = functools.reduce(lambda x, y: x and y, results)

                except* servo.ConnectorNotFoundError as e:
                    typer.echo(
                        "A connector named within the checks config was not found in the current Assembly"
                    )
                    raise typer.Exit(1) from e
                except* servo.EventHandlersNotFoundError as e:
                    typer.echo(
                        "At least one configured connector must respond to the Check event (Note the servo "
                        "responds to checks so this error should never raise unless something is well and truly wrong"
                    )
                    raise typer.Exit(1) from e

                if not ready:
                    raise typer.Exit(1)

            if not dry_run:
                poll = not no_poll
                servo.runner.AssemblyRunner(
                    context.assembly, poll=poll, interactive=bool(interactive)
                ).run(
                    debug=debug,
                )

        @self.command(section=section)
        def describe(
            context: Context,
            connectors: Optional[list[str]] = typer.Argument(
                None, help="The connectors to describe"
            ),
        ) -> None:
            """
            Display current state of servo resources
            """

            for servo_ in context.assembly.servos:
                if context.servo_ and context.servo_ != servo_:
                    continue

                # Validate that explicit args support describe events
                connectors_ = (
                    self.connectors_named(connectors, servo_=servo_)
                    if connectors
                    else servo_.all_connectors
                )

                results: list[servo.EventResult] = run_async(
                    servo_.dispatch_event(servo.Events.describe, include=connectors_)
                )
                headers = ["CONNECTOR", "COMPONENTS", "METRICS"]
                table = []
                for result in results:
                    description: servo.Description = result.value
                    components_column = []
                    for component in description.components:
                        for setting in component.settings:
                            components_column.append(
                                f"{component.name}.{setting.name}={setting.human_readable_value}"
                            )

                    metrics_column = []
                    for metric in description.metrics:
                        metrics_column.append(f"{metric.name} ({metric.unit})")

                    result.connector.name
                    row = [
                        result.connector.name,
                        "\n".join(components_column),
                        "\n".join(metrics_column),
                    ]
                    table.append(row)

                if len(context.assembly.servos) > 1:
                    typer.echo(f"{servo_.name}")
                typer.echo(tabulate(table, headers, tablefmt="plain"))

        def metrics_callback(
            context: typer.Context, value: Optional[list[str]]
        ) -> Optional[list[servo.Metric]]:
            if not value:
                return value

            all_metrics_by_name: dict[str, servo.Metric] = {}
            results = run_async(context.servo.dispatch_event("metrics"))
            for result in results:
                for metric in result.value:
                    all_metrics_by_name[metric.name] = metric

            metrics: list[servo.Metric] = []
            for metric_name in value:
                if metric := all_metrics_by_name.get(metric_name, None):
                    metrics.append(metric)
                else:
                    raise typer.BadParameter(f"no metric found named '{metric_name}'")

            return metrics

        @self.command(section=section)
        def measure(
            context: Context,
            metrics: Optional[list[str]] = typer.Argument(
                None, help="Metrics to measure", callback=metrics_callback
            ),
            connectors: Optional[list[str]] = typer.Option(
                None,
                "--connectors",
                "-c",
                help="Connectors to measure from",
                metavar="[CONNECTORS]...",
            ),
            duration: Optional[str] = typer.Option(
                "0",
                "--duration",
                "-d",
                help="Duration of the measurement",
                metavar="DURATION",
                callback=self.duration_callback,
            ),
            verbose: bool = typer.Option(
                False,
                "--verbose",
                "-v",
                help="Display verbose output",
            ),
            humanize: bool = typer.Option(
                True,
                help="Display human readable output for units",
            ),
        ) -> None:
            """
            Capture measurements for one or more metrics
            """
            for servo_ in context.assembly.servos:
                if context.servo_ and servo_ != context.servo_:
                    continue

                connectors_ = (
                    self.connectors_named(connectors, servo_)
                    if connectors
                    else list(
                        filter(
                            lambda c: c.responds_to_event(servo.Events.measure),
                            servo_.all_connectors,
                        )
                    )
                )

                if metrics:
                    # Filter target connectors by metrics
                    results: list[servo.EventResult] = run_async(
                        servo_.dispatch_event(servo.Events.metrics, include=connectors_)
                    )
                    for result in results:
                        result_metrics: list[servo.Metric] = result.value
                        metric_names: Set[str] = set(
                            map(lambda m: m.name, result_metrics)
                        )
                        if not metric_names | set(metrics):
                            connectors.remove(result.connector)

                # Capture the measurements
                results: list[servo.EventResult] = run_async(
                    servo_.dispatch_event(
                        servo.Events.measure,
                        metrics=metrics,
                        control=servo.Control(duration=duration),
                        include=connectors_,
                    )
                )

                # FIXME: The data that is crossing connector boundaries needs to be validated
                aggregated_by_metric: dict[
                    servo.Metric,
                    dict[
                        str,
                        dict[
                            servo.BaseConnector,
                            list[Tuple[servo.Numeric, servo.Reading]],
                        ],
                    ],
                ] = {}
                metric_names = list(map(lambda m: m.name, metrics)) if metrics else None
                headers = ["METRIC", "UNIT", "READINGS"]
                table = []
                for result in results:
                    measurement = result.value
                    if not measurement:
                        continue

                    for reading in measurement.readings:
                        metric = reading.metric

                        if isinstance(reading, servo.TimeSeries):
                            metric_to_timestamp = aggregated_by_metric.get(metric, {})
                            for data_point in reading.data_points:
                                time_key = f"{data_point[0]:%Y-%m-%d %H:%M:%S}"
                                timestamp_to_connector = metric_to_timestamp.get(
                                    time_key, {}
                                )
                                values = timestamp_to_connector.get(
                                    result.connector, []
                                )
                                values.append((data_point[1], reading))
                                timestamp_to_connector[result.connector] = values
                                metric_to_timestamp[time_key] = timestamp_to_connector

                            aggregated_by_metric[metric] = metric_to_timestamp

                        elif isinstance(reading, servo.DataPoint):
                            metric_to_timestamp = aggregated_by_metric.get(metric, {})
                            time_key = f"{reading.time:%Y-%m-%d %H:%M:%S}"
                            timestamp_to_connector = metric_to_timestamp.get(
                                time_key, {}
                            )
                            values = timestamp_to_connector.get(result.connector, [])
                            values.append((reading.value, reading))
                            timestamp_to_connector[result.connector] = values
                            metric_to_timestamp[time_key] = timestamp_to_connector
                            aggregated_by_metric[metric] = metric_to_timestamp

                        else:
                            raise TypeError(
                                f"unknown reading type: {reading.__class__.__name__}"
                            )

                # Print the table
                def attribute_connector(connector, reading) -> str:
                    return (
                        f"[{connector.name}{reading.id or ''}]"
                        if len(connectors) > 1
                        else ""
                    )

                headers = ["METRIC", "UNIT", "READINGS"]
                table = []
                for metric in sorted(aggregated_by_metric.keys(), key=lambda m: m.name):
                    readings_column = []
                    timestamp_to_connectors = aggregated_by_metric[metric]
                    for timestamp in sorted(timestamp_to_connectors.keys()):
                        for connector, values in timestamp_to_connectors[
                            timestamp
                        ].items():  # dict[BaseConnector, Tuple[Numeric, Reading]]
                            readings_column.extend(
                                list(
                                    map(
                                        lambda r: f"{r[0]:.2f} ({timeago(timestamp) if humanize else timestamp}) {attribute_connector(connector, r[1])}",
                                        values,
                                    )
                                )
                            )

                    row = [
                        metric.name,
                        metric.unit,
                        "\n".join(readings_column),
                    ]
                    table.append(row)

                if len(context.assembly.servos) > 1:
                    typer.echo(f"{servo_.name}")
                typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

        @self.command(section=section)
        def inject_sidecar(
            context: Context,
            target: str = typer.Argument(
                ...,
                help="Deployment or Pod to inject the sidecar on (deployment/NAME or pod/NAME)",
            ),
            image: str = typer.Option(
                ...,
                "--image",
                "-i",
                help="Image of the sidecar to be injected in form [REPO]/[IMAGE]:[TAG]",
            ),
            namespace: str = typer.Option(
                "default", "--namespace", "-n", help="Namespace of the target"
            ),
            service: Optional[str] = typer.Option(
                None, "--service", "-s", help="Service to target"
            ),
            port: Optional[str] = typer.Option(
                None, "--port", "-p", help="Port to target (NAME or NUMBER)"
            ),
        ) -> None:
            """
            Inject an Envoy sidecar to capture metrics
            """
            if not target.startswith(("deploy/", "deployment/", "pod/", "rollout/")):
                raise typer.BadParameter(
                    'target must prefixed with Kubernetes object kind of "deployment", "rollout" or "pod"'
                )

            if not (service or port):
                raise typer.BadParameter("service or port must be given")

            # TODO: Dry this up...
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                kubernetes_asyncio.config.load_incluster_config()
            else:
                kubeconfig = (
                    os.getenv("KUBECONFIG")
                    or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION
                )
                kubeconfig_path = pathlib.Path(os.path.expanduser(kubeconfig))
                if kubeconfig_path.exists():
                    run_async(
                        kubernetes_asyncio.config.load_kube_config(
                            config_file=os.path.expandvars(kubeconfig_path),
                        )
                    )

            if target.startswith("deploy"):
                deployment = run_async(
                    DeploymentHelper.read(target.split("/", 1)[1], namespace)
                )
                run_async(
                    DeploymentHelper.inject_sidecar(
                        deployment, "opsani-envoy", image, service=service, port=port
                    )
                )
                typer.echo(
                    f"Envoy sidecar injected to Deployment {deployment.metadata.name} in {namespace}"
                )

            elif target.startswith("rollout"):
                raise typer.BadParameter(
                    "Rollout sidecar injection is not yet implemented"
                )
            elif target.startswith("pod"):
                raise typer.BadParameter("Pod sidecar injection is not yet implemented")
            else:
                raise typer.BadParameter(f"unexpected sidecar target: {target}")

        @self.command(section=section)
        def eject_sidecar(
            context: Context,
            target: str = typer.Argument(
                ...,
                help="Deployment or Pod to eject the sidecar from (deployment/NAME or pod/NAME)",
            ),
            namespace: str = typer.Option(
                "default", "--namespace", "-n", help="Namespace of the target"
            ),
        ) -> None:
            """
            Eject an Envoy sidecar
            """
            if not target.startswith(("deploy/", "deployment/", "pod/")):
                raise typer.BadParameter(
                    'target must prefixed with Kubernetes object kind of "deployment" or "pod"'
                )

            # TODO: Dry this up...
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                kubernetes_asyncio.config.load_incluster_config()
            else:
                kubeconfig = (
                    os.getenv("KUBECONFIG")
                    or kubernetes_asyncio.config.kube_config.KUBE_CONFIG_DEFAULT_LOCATION
                )
                kubeconfig_path = pathlib.Path(os.path.expanduser(kubeconfig))
                if kubeconfig_path.exists():
                    run_async(
                        kubernetes_asyncio.config.load_kube_config(
                            config_file=os.path.expandvars(kubeconfig_path),
                        )
                    )

            if target.startswith("deploy"):
                deployment = run_async(
                    servo.connectors.kubernetes.Deployment.read(
                        target.split("/", 1)[1], namespace
                    )
                )
                ejected = run_async(deployment.eject_sidecar("opsani-envoy"))
                if ejected:
                    typer.echo(
                        f"Envoy sidecar ejected from Deployment {deployment.name} in {namespace}"
                    )
                else:
                    typer.echo(
                        f"No Envoy sidecar found in Deployment {deployment.name} in {namespace}",
                        err=True,
                    )
                    raise typer.Exit(code=1)

            elif target.startswith("pod"):
                raise typer.BadParameter("Pod sidecar ejection is not yet implemented")
            else:
                raise typer.BadParameter(f"unexpected sidecar target: {target}")

        @self.command(section=section)
        def adjust(
            context: Context,
            settings: Optional[list[str]] = typer.Argument(
                None,
                help="The settings to adjust (format is [COMPONENT].[SETTING=[VALUE])",
            ),
        ) -> None:
            """
            Adjust settings for one or more components
            """

            for servo_ in context.assembly.servos:
                if context.servo_ and context.servo_ != servo_:
                    continue

                adjustments: list[servo.Adjustment] = []
                for descriptor in settings:
                    try:
                        component_name, setting_descriptor = descriptor.split(".", 1)
                        setting_name, value = setting_descriptor.split("=", 1)
                    except ValueError:
                        raise typer.BadParameter(
                            f"unable to parse setting descriptor '{descriptor}': expected format is `component.setting=value`"
                        )

                    adjustment = servo.Adjustment(
                        component_name=component_name,
                        setting_name=setting_name,
                        value=value,
                    )
                    adjustments.append(adjustment)

                results: list[servo.EventResult] = run_async(
                    servo_.dispatch_event(servo.Events.adjust, adjustments)
                )
                if not results:
                    typer.echo(
                        "adjustment failed: no connector handled the request", err=True
                    )
                    raise typer.Exit(code=1)

                for result in results:
                    outcome = result.value

                    if isinstance(outcome, Exception):
                        message = str(outcome.get("message", "undefined"))
                        raise servo.ConnectorError(
                            f'Adjustment connector failed with error "{outcome}" and message:\n{message}'
                        )

                headers = ["CONNECTOR", "SETTINGS"]
                table = []
                for result in results:
                    description: servo.Description = result.value
                    settings_column = []
                    for component in description.components:
                        for setting in component.settings:
                            settings_column.append(
                                f"{component.name}.{setting.name}={setting.human_readable_value}"
                            )

                    result.connector.name
                    row = [result.connector.name, "\n".join(settings_column)]
                    table.append(row)

                    if len(context.assembly.servos) > 1:
                        typer.echo(f"{servo_.name}")
                    typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")

    def add_config_commands(self, section=Section.config) -> None:
        @self.command(section=section)
        def config(
            context: Context,
            format: ConfigOutputFormat = typer.Option(
                ConfigOutputFormat.yaml, "--format", "-f", help="Select output format"
            ),
            output: typer.FileTextWrite = typer.Option(
                None, "--output", "-o", help="Output configuration to [FILE]"
            ),
            keys: Optional[list[str]] = typer.Argument(
                None, help="Display settings for specific keys"
            ),
        ) -> None:
            """
            Display configured settings
            """
            include = set(keys) if keys else None
            export_options = dict(
                exclude_unset=True, exclude_defaults=True, include=include, indent=2
            )

            for servo_ in context.assembly.servos:
                if context.servo_ and context.servo_ != servo_:
                    continue

                if format == ConfigOutputFormat.text:
                    pass
                else:
                    lexer = format.lexer()
                    if format == ConfigOutputFormat.yaml:
                        data = servo_.config.yaml(sort_keys=True, **export_options)
                    elif format == ConfigOutputFormat.json:
                        data = servo_.config.json(**export_options)
                    elif format == ConfigOutputFormat.dict:
                        # NOTE: Round-trip through JSON to produce primitives
                        config_dict = servo_.config.json(**export_options)
                        data = devtools.pformat(json.loads(config_dict))
                    elif format == ConfigOutputFormat.configmap:
                        configured_at = datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        connectors = []
                        for connector in servo_.connectors:
                            connectors.append(
                                {
                                    "name": connector.name,
                                    "type": connector.full_name,
                                    "description": connector.description,
                                    "version": str(connector.version),
                                    "url": str(connector.homepage),
                                }
                            )
                        connectors_json_str = json.dumps(connectors, indent=None)

                        configmap = {
                            "apiVersion": "v1",
                            "kind": "ConfigMap",
                            "metadata": {
                                "name": "opsani-servo-config",
                                "labels": {
                                    "app.kubernetes.io/name": "servo",
                                    "app.kubernetes.io/version": str(servo_.version),
                                },
                                "annotations": {
                                    "servo.opsani.com/configured_at": configured_at,
                                    "servo.opsani.com/connectors": connectors_json_str,
                                },
                            },
                            "data": {
                                "servo.yaml": servo.utilities.yaml.PreservedScalarString(
                                    servo_.config.yaml(
                                        sort_keys=True,
                                        exclude={"optimizer"},
                                        **export_options,
                                    )
                                )
                            },
                        }
                        data = yaml.dump(
                            configmap, indent=2, sort_keys=False, explicit_start=True
                        )
                    else:
                        raise RuntimeError(
                            "no handler configured for output format {format}"
                        )

                    if output:
                        output.write(data)
                    else:
                        typer.echo(
                            pygments.highlight(
                                data, lexer, pygments.formatters.TerminalFormatter()
                            )
                        )

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
            output_data = ""

            if format == SchemaOutputFormat.text or format == SchemaOutputFormat.html:
                typer.echo("error: not yet implemented", err=True)
                raise typer.Exit(1)

            if top_level:
                CLI.assemble_from_context(context)

                if all is False and context.servo_ is None:
                    typer.echo(
                        "error: schema can only be outputted for all connectors or a single servo",
                        err=True,
                    )
                    raise typer.Exit(1)

                if format == SchemaOutputFormat.json:
                    output_data += context.servo_.top_level_schema_json(all=all)

                elif format == SchemaOutputFormat.dict:
                    output_data += devtools.pformat(
                        context.servo_.top_level_schema(all=all)
                    )

            else:
                if connector:
                    if isinstance(connector, servo.BaseConnector):
                        config_model = connector.config.__class__
                    elif issubclass(connector, servo.BaseConnector):
                        config_model = connector.config_model()
                    else:
                        raise typer.BadParameter(
                            f"unexpected connector type '{connector.__class__.__name__}'"
                        )
                else:
                    CLI.assemble_from_context(context)

                    if context.servo_ is None:
                        typer.echo(
                            "error: schema can only be outputted for a single servo",
                            err=True,
                        )
                        raise typer.Exit(1)

                    config_model = context.servo_.config.__class__

                if format == SchemaOutputFormat.json:
                    output_data += config_model.schema_json(indent=2)
                elif format == SchemaOutputFormat.dict:
                    output_data += devtools.pformat(config_model.schema())
                else:
                    raise RuntimeError(
                        f"no handler configured for output format {format}"
                    )

            assert output_data, "output_data not assigned"

            if output:
                output.write(output_data)
            else:
                typer.echo(
                    pygments.highlight(
                        output_data,
                        format.lexer(),
                        pygments.formatters.TerminalFormatter(),
                    )
                )

        @self.command(section=section)
        def validate(
            context: Context,
            connectors: Optional[list[str]] = typer.Argument(
                None,
                help="Connectors to validate configuration for. \nFormats: `connector`, `ConnectorClass`, `alias:connector`, `alias:ConnectorClass`",
            ),
            file: pathlib.Path = typer.Option(
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
                False,
                "--quiet",
                "-q",
                help="Do not echo generated output to stdout",
            ),
        ) -> None:
            """Validate a configuration"""
            try:
                configs = list(yaml.load_all(open(file), Loader=yaml.FullLoader))
                if not isinstance(configs, list):
                    raise ValueError(
                        f'error: config file "{file}" parsed to an unexpected value of type "{configs.__class__}"'
                    )

                # If we parsed an empty file, add an empty dict to work with
                if not configs:
                    configs.append({})

                # NOTE: When connector descriptor is provided the validation is constrained
                routes = self.connector_routes_callback(
                    context=context, value=connectors
                )

                for config in configs:
                    config_model, routes = servo.assembly._create_config_model(
                        config=config, routes=routes
                    )
                    config_model.parse_file(file)
            except (pydantic.ValidationError, yaml.scanner.ScannerError, KeyError) as e:
                if not quiet:
                    typer.echo(f"X Invalid configuration in {file}", err=True)
                    typer.echo(e, err=True)
                raise typer.Exit(1)

            if not quiet:
                typer.echo(f"√ Valid configuration in {file}")

        @self.command(section=section)
        def generate(
            context: Context,
            connectors: Optional[list[str]] = typer.Argument(
                None,
                help="Connectors to generate configuration for. \nFormats: `connector`, `ConnectorClass`, `alias:connector`, `alias:ConnectorClass`",
            ),
            file: pathlib.Path = typer.Option(
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
            name: Optional[str] = typer.Option(
                None,
                "--name",
                "-n",
                help="Set the name of the generated configuration",
            ),
            standalone: bool = typer.Option(
                False,
                "--standalone",
                "-s",
                help="Exclude connectors descriptor in generated output",
            ),
            quiet: bool = typer.Option(
                False,
                "--quiet",
                "-q",
                help="Do not echo generated output to stdout",
            ),
            force: bool = typer.Option(
                False,
                "--force",
                help="Overwrite output file without prompting",
            ),
            append: bool = typer.Option(
                False,
                "--append",
                help="Append the generated output to an existing file",
            ),
        ) -> None:
            """Generate a configuration"""
            exclude_unset = not defaults
            exclude = {"connectors", "servo"} if standalone else {}

            routes = (
                self.connector_routes_callback(context=context, value=connectors)
                if connectors
                else servo.connector._default_routes()
            )

            # Build a settings model from our routes
            config_model = servo.assembly._create_config_model_from_routes(routes)
            config = config_model.generate(name=name)

            if connectors and len(connectors):
                # Check is we have any aliases and assign dictionary
                connectors_dict: dict[str, str] = {}
                aliased = False
                for identifier in connectors:
                    if ":" in identifier:
                        alias, id = identifier.split(":", 1)
                        connectors_dict[alias] = id
                        aliased = True
                    else:
                        connectors_dict[identifier] = identifier

                if aliased:
                    config.connectors = connectors_dict
                else:
                    # If there are no aliases just assign input values
                    config.connectors = connectors

            config_yaml = config.yaml(
                by_alias=True,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_unset,
                exclude=exclude,
                exclude_none=True,
            )
            if file.exists():
                if append:
                    config_docs = list(yaml.full_load_all(file.read_text()))
                    incoming_doc = yaml.full_load(config_yaml)
                    config_docs.append(incoming_doc)
                    config_yaml = yaml.dump_all(config_docs)

                elif force == False:
                    delete = typer.confirm(
                        f"File '{file}' already exists. Overwrite it?"
                    )
                    if not delete:
                        raise typer.Abort()

            file.write_text(config_yaml)
            if not quiet:
                typer.echo(
                    pygments.highlight(
                        config_yaml,
                        pygments.lexers.YamlLexer(),
                        pygments.formatters.TerminalFormatter(),
                    )
                )
                typer.echo(f"Generated {file}")

    def add_connector_commands(self) -> None:
        for cli in ConnectorCLI.__clis__:
            self.add_cli(cli, section=Section.connectors)

        @self.command(section=Section.other)
        def version(
            context: Context,
            connector: Optional[str] = typer.Argument(
                None,
                help="Display version for a connector",
            ),
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
            if connector:
                connector_class = servo.connector._connector_class_from_string(
                    connector
                )
                if not connector_class:
                    raise typer.BadParameter(
                        f"no connector found for key '{connector}'"
                    )
            else:
                connector_class = servo.Servo

            if short:
                if format == VersionOutputFormat.text:
                    typer.echo(connector_class.version_summary())
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": connector_class.full_name,
                        "version": str(connector_class.version),
                        "cryptonym": connector_class.cryptonym,
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")
            else:
                if format == VersionOutputFormat.text:
                    typer.echo(connector_class.summary())
                elif format == VersionOutputFormat.json:
                    version_info = {
                        "name": connector_class.full_name,
                        "version": str(connector_class.version),
                        "cryptonym": connector_class.cryptonym,
                        "maturity": str(connector_class.maturity),
                        "description": connector_class.description,
                        "homepage": connector_class.homepage,
                        "license": str(connector_class.license),
                    }
                    typer.echo(json.dumps(version_info, indent=2))
                else:
                    raise typer.BadParameter(f"Unknown format '{format}'")

            raise typer.Exit(0)


def _run(args: Union[str, list[str]], **kwargs) -> None:
    args = shlex.split(args) if isinstance(args, str) else args
    process = subprocess.run(args, **kwargs)
    if process.returncode != 0:
        sys.exit(process.returncode)


def run_async(future: Union[asyncio.Future, asyncio.Task, Awaitable]) -> Any:
    """Run the asyncio event loop until Future is done.

    This function is similar to asyncio.run but avoids the behavior of always creating a new event loop even if one
    already exists.

    Args:
        future: The future to run.

    Returns:
        Any: The Future's result.

    Raises:
        Exception: Any exception raised during execution of the future.
    """
    existing_loop = True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        if str(e) == "no running event loop":
            existing_loop = False
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise
    try:
        return loop.run_until_complete(future)
    finally:
        if not existing_loop:
            loop.close()
            asyncio.set_event_loop(None)


def print_table(table, headers) -> None:
    typer.echo(tabulate(table, headers, tablefmt="plain") + "\n")
