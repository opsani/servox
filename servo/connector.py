import abc
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, get_type_hints, Union, Set
import importlib

import httpx
import semver
import typer
import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    BaseSettings,
    Extra,
    Field,
    FilePath,
    HttpUrl,
    ValidationError,
    constr,
    root_validator,
    validator,
)
from pydantic.schema import schema as pydantic_schema
from pydantic.json import pydantic_encoder
from servo.config import Settings, Version, License, Maturity

#####

from pkg_resources import EntryPoint, iter_entry_points
from typing import Generator

ENTRY_POINT_GROUP = "servo.connectors"

class ConnectorLoader:
    '''Dynamics discovers and loads connectors via entry points'''

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self.group = group

    def iter_entry_points(self) -> Generator[EntryPoint, None, None]:
        yield from iter_entry_points(group=self.group, name=None)

    def load(self) -> Generator[Any, None, None]:
        for entry_point in self.iter_entry_points():
            yield entry_point.resolve()

#####

class Connector(BaseModel, abc.ABC):
    """
    Connectors expose functionality to Servo assemblies by connecting external services and resources.
    """

    # Global registry of all available connectors
    __subclasses: ClassVar[Set[Type["Connector"]]] = set()

    # Connector metadata
    name: ClassVar[str] = None
    version: ClassVar[Version] = None
    description: ClassVar[Optional[str]] = None
    homepage: ClassVar[Optional[HttpUrl]] = None
    license: ClassVar[Optional[License]] = None
    maturity: ClassVar[Optional[Maturity]] = None

    # Instance configuration
    path: str
    settings: Settings
    _logger: logger

    @classmethod
    def all(cls) -> Set[Type["Connector"]]:
        '''Return a set of all Connector subclasses'''
        return cls.__subclasses

    @root_validator(pre=True)
    @classmethod
    def validate_metadata(cls, v):
        assert cls.name is not None, "name must be provided"
        assert cls.version is not None, "version must be provided"
        if isinstance(cls.version, str):
            # Attempt to parse
            cls.version = Version.parse(cls.version)
        assert isinstance(
            cls.version, (Version, semver.VersionInfo)
        ), "version is not a semantic versioning descriptor"
        return v

    @validator("path")
    @classmethod
    def validate_path(cls, v):
        assert bool(
            re.match("^[0-9a-zA-Z-_/\.]{4,128}$", v)
        ), "paths may only contain alphanumeric characters, hyphens, slashes, periods, and underscores"
        return v

    @classmethod
    def settings_class(cls) -> Type['Settings']:
        hints = get_type_hints(cls)
        settings_cls = hints["settings"]
        return settings_cls

    @classmethod
    def default_path(cls) -> str:
        name = cls.__name__.replace("Connector", "")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__.replace("Connector", " Connector")
        cls.version = semver.VersionInfo.parse("0.0.0")
        cls.__subclasses.add(cls)

    def __init__(self, settings: Settings, *, path: Optional[str] = None, **kwargs):
        path = path if path is not None else self.default_path()
        super().__init__(path=path, settings=settings, **kwargs)

    ##
    # Subclass services

    async def api_client(self) -> httpx.AsyncClient:
        """Yields an httpx.AsyncClient instance configured to talk to Opsani API"""
        async with httpx.AsyncClient() as client:
            yield client

    def logger(self) -> logger:
        """Returns the logger"""
        return self._logger

    def cli(self) -> Optional[typer.Typer]:
        """Returns a Typer CLI for the connector"""
        return None


def metadata(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[semver.VersionInfo] = None,
    homepage: Optional[HttpUrl] = None,
    license: Optional[License] = None,
    maturity: Optional[Maturity] = None,
):
    '''Decorate a connector class with metadata'''
    def decorator(cls):
        if name:
            cls.name = name
        if description:
            cls.description = description
        if version:
            cls.version = (
                version
                if isinstance(version, semver.VersionInfo)
                else Version.parse(version)
            )
        if homepage:
            cls.homepage = homepage
        if license:
            cls.license = license
        if maturity:
            cls.maturity = maturity
        return cls

    return decorator


class ConnectorCLI(typer.Typer):
    connector: Connector

    def __init__(self, connector: Connector, **kwargs):
        self.connector = connector
        name = kwargs.pop("name", connector.path) # TODO: Add command_name returning last element of path OR just rename to name
        help = kwargs.pop("help", connector.description)
        add_completion = kwargs.pop("add_completion", False)
        super().__init__(name=name, help=help, add_completion=add_completion, **kwargs)
        self.add_commands()

    # TODO: Converge the commands
    def add_commands(self):
        @self.command()
        def schema():
            """
            Display the schema 
            """
            # TODO: Support output formats (dict, json, yaml)...
            typer.echo(self.connector.settings.schema_json(indent=2))

        @self.command()
        def generate():
            """Generate a configuration file"""
            # TODO: support output paths/formats
            # NOTE: We have to serialize through JSON first
            schema = json.loads(json.dumps(self.connector.settings.dict(by_alias=True)))
            output_path = Path.cwd() / f"{self.connector.path}.yaml"
            output_path.write_text(yaml.dump(schema))
            typer.echo(f"Generated {self.connector.path}.yaml")

        @self.command()
        def validate(file: typer.FileText = typer.Argument(...), key: str = ""):
            """
            Validate given file against the JSON Schema
            """
            try:
                config = yaml.load(file, Loader=yaml.FullLoader)
                connector_config = config[key] if key != "" else config
                cls = type(self.connector.settings)
                config = cls.parse_obj(connector_config)
                typer.echo("√ Valid connector configuration")
            except (ValidationError, yaml.scanner.ScannerError) as e:
                typer.echo("X Invalid connector configuration", err=True)
                typer.echo(e, err=True)
                raise typer.Exit(1)

        @self.command()
        def info():
            """
            Display connector info
            """
            typer.echo(
                (
                    f"{self.connector.name} v{self.connector.version} ({self.connector.maturity})\n"
                    f"{self.connector.description}\n"
                    f"{self.connector.homepage}\n"
                    f"Licensed under the terms of {self.connector.license}\n"
                )
            )

        @self.command()
        def version():
            """
            Display version
            """
            typer.echo(f"{self.connector.name} v{self.connector.version}")
