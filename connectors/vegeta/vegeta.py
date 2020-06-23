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
import durationpy
import servo
from servo.connector import Connector, ConnectorCLI, ConnectorSettings, License, Maturity

# TODO: This should really come down to `from servo import Connector, ConnectorSettings`

###
### Vegeta


class TargetFormat(str, Enum):
    http = "http"
    json = "json"

    def __str__(self):
        return self.value


class VegetaSettings(ConnectorSettings):
    """
    Configuration of the Vegeta connector
    """

    rate: str = Field(
        description="Specifies the request rate per time unit to issue against the targets. Given in the format of request/time unit.",
    )
    duration: str = Field(
        description="Specifies the amount of time to issue requests to the targets.",
    )
    format: TargetFormat = Field(
        "http",
        description="Specifies the format of the targets input. Valid values are http and json. Refer to the Vegeta docs for details.",
    )
    target: Optional[str] = Field(
        description="Specifies a single formatted Vegeta target to load. See the format option to learn about available target formats. This option is exclusive of the targets option and will provide a target to Vegeta via stdin."
    )
    targets: Optional[FilePath] = Field(
        description="Specifies the file from which to read targets. See the format option to learn about available target formats. This option is exclusive of the target option and will provide targets to via through a file on disk."
    )
    connections: int = Field(
        10000,
        description="Specifies the maximum number of idle open connections per target host.",
    )
    workers: int = Field(
        10,
        description="Specifies the initial number of workers used in the attack. The workers will automatically increase to achieve the target request rate, up to max-workers.",
    )
    max_workers: int = Field(
        18446744073709551615,
        alias="max-workers",
        description="The maximum number of workers used to sustain the attack. This can be used to control the concurrency of the attack to simulate a target number of clients.",
        env="",
    )
    max_body: int = Field(
        -1,
        alias="max-body",
        description="Specifies the maximum number of bytes to capture from the body of each response. Remaining unread bytes will be fully read but discarded.",
        env="",
    )
    http2: bool = Field(
        True,
        description="Specifies whether to enable HTTP/2 requests to servers which support it.",
    )
    keepalive: bool = Field(
        True,
        description="Specifies whether to reuse TCP connections between HTTP requests.",
    )
    insecure: bool = Field(
        False,
        description="Specifies whether to ignore invalid server TLS certificates.",
    )

    @root_validator()
    @classmethod
    def validate_target(cls, values):
        target, targets = values.get("target"), values.get("targets")
        if target is None and targets is None:
            raise ValueError("target or targets must be configured")

        if target is not None and targets is not None:
            raise ValueError("target and targets cannot both be configured")

        return values

    @root_validator()
    @classmethod
    def validate_target_format(cls, values):
        target, targets = values.get("target"), values.get("targets")

        # Validate JSON target formats
        if target is not None and values.get("format") == TargetFormat.json:
            try:
                json.loads(target)
            except Exception as e:
                raise ValueError("the target is not valid JSON") from e

        if targets is not None and values.get("format") == TargetFormat.json:
            try:
                json.load(open(targets))
            except Exception as e:
                raise ValueError("the targets file is not valid JSON") from e

        # TODO: Add validation of JSON with JSON Schema (https://github.com/tsenart/vegeta/blob/master/lib/target.schema.json)
        # and HTTP format
        return values

    @validator("rate")
    @classmethod
    def validate_rate(cls, v):
        assert isinstance(
            v, (int, str)
        ), "rate must be an integer or a rate descriptor string"

        # Integer rates
        if isinstance(v, int) or v.isnumeric():
            return str(v)

        # Check for hits/interval
        components = v.split("/")
        assert len(components) == 2, "rate strings are of the form hits/interval"

        hits = components[0]
        duration = components[1]
        assert hits.isnumeric(), "rate must have an integer hits component"

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(duration)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    @validator("duration")
    @classmethod
    def validate_duration(cls, v):
        assert isinstance(
            v, (int, str)
        ), "duration must be an integer or a duration descriptor string"

        if v == "0" or v == 0:
            return v

        # Try to parse it from Golang duration string
        try:
            durationpy.from_str(v)
        except Exception as e:
            raise ValueError(str(e)) from e

        return v

    class Config:
        json_encoders = {TargetFormat: lambda t: t.value()}

@servo.connector.metadata(
    description="Vegeta load testing connector",
    version="0.5.0",
    homepage="https://github.com/opsani/vegeta-connector",
    license=License.APACHE2,
    maturity=Maturity.STABLE,
)
class VegetaConnector(Connector):
    settings: VegetaSettings

    # TODO: Measure

    def cli(self) -> ConnectorCLI:
        """Returns a Typer CLI for interacting with this connector"""
        cli = ConnectorCLI(self, help="Load generation with Vegeta")

        @cli.command()
        def loadgen():
            """
            Run an adhoc load generation
            """

        return cli
    
    # TODO: Message handlers...
    # Model the metrics

    def measure(self):
        pass

    def describe(self):
        pass
