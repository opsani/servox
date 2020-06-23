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

# TODO: This can move back into the connectors file or into a module

class Optimizer(BaseModel):
    org_domain: constr(
        regex=r"(([\da-zA-Z])([_\w-]{,62})\.){,127}(([\da-zA-Z])[_\w-]{,61})?([\da-zA-Z]\.((xn\-\-[a-zA-Z\d]+)|([a-zA-Z\d]{2,})))"
    )
    app_name: constr(regex=r"^[a-z\-]{3,64}$")
    token: str
    base_url: HttpUrl = "https://api.opsani.com/"

    def __init__(self, id: str = None, token: str = None, **kwargs):
        org_domain = kwargs.pop("org_domain", None)
        app_name = kwargs.pop("app_name", None)
        if id:
            org_domain, app_name = id.split("/")
        super().__init__(
            org_domain=org_domain, app_name=app_name, token=token, **kwargs
        )

    def id(self) -> str:
        """Returns the optimizer identifier"""
        return f"{self.org_domain}/{self.app_name}"

# TODO: Rename to BaseConnectorSettings
class Settings(BaseSettings):
    description: Optional[str]

    # Optimizer we are communicating with
    _optimizer: Optimizer

    class Config:
        env_prefix = "SERVO_"
        extra = Extra.forbid


class License(Enum):
    """Defined licenses"""

    MIT = "MIT"
    APACHE2 = "Apache 2.0"
    PROPRIETARY = "Proprietary"

    @classmethod
    def from_str(cls, identifier: str) -> "License":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No license identified by "{identifier}".')

    def __str__(self):
        return self.value


class Maturity(Enum):
    """Connector maturity level"""

    EXPERIMENTAL = "Experimental"
    STABLE = "Stable"
    ROBUST = "Robust"

    @classmethod
    def from_str(cls, identifier: str) -> "Maturity":
        """
        Returns a `License` for the given string identifier (e.g. "MIT").
        """
        for _, env in cls.__members__.items():
            if env.value == identifier:
                return env
        raise NameError(f'No maturity level identified by "{identifier}".')

    def __str__(self):
        return self.value


class Version(semver.VersionInfo):
    pass
