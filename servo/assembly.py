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
import contextvars
import functools
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import pydantic
import pydantic.json
import pydantic_settings
import yaml

import servo.configuration
import servo.connector
import servo.pubsub
import servo.servo
import servo.telemetry

__all__ = ["Assembly", "current_assembly"]


_current_context_var = contextvars.ContextVar("servox.current_assembly", default=None)


def current_assembly() -> Optional["Assembly"]:
    """
    Return the active assembly for the current execution context.

    The value is managed by a contextvar and is concurrency safe.
    """
    return _current_context_var.get()


class Assembly(pydantic.BaseModel):
    """
    An Assembly models the environment and runtime configuration of a collection of Servos.

    Connectors are dynamically loaded via setuptools entry points
    (see https://packaging.python.org/specifications/entry-points/)
    and the settings class for a Servo instance must be created at
    runtime because the servo.yaml configuration file includes settings
    for an arbitrary number of connectors mounted onto arbitrary keys
    in the config file.

    The Assembly class is responsible for handling the connector
    loading and creating a concrete BaseServoConfiguration model that supports
    the connectors available and activated in the assembly. An assembly
    is the combination of configuration and associated code artifacts
    in an executable environment (e.g. a Docker image or a Python virtualenv
    running on your workstation.

    NOTE: The Assembly class overrides the Pydantic base class implementations
    of the schema family of methods. See the method docstrings for specific details.
    """

    config_file: Optional[pathlib.Path] = None
    servos: List[servo.servo.Servo]
    _context_token: Optional[contextvars.Token] = pydantic.PrivateAttr(None)

    @classmethod
    async def assemble(
        cls,
        *,
        config_file: Optional[pathlib.Path] = None,
        configs: Optional[List[Dict[str, Any]]] = None,
        env: Optional[Dict[str, str]] = os.environ,
    ) -> "Assembly":
        """Assemble a Servo by processing configuration and building a dynamic settings model"""

        if config_file is None and configs is None:
            raise ValueError(f"cannot assemble with a config file and config objects")

        _discover_connectors()

        if config_file and not configs:
            # Build our Servo configuration from the config file + environment
            if not config_file.exists():
                raise FileNotFoundError(f"config file '{config_file}' does not exist")

            configs = list(yaml.load_all(open(config_file), Loader=yaml.FullLoader))
            if not isinstance(configs, list):
                raise ValueError(
                    f'error: config file "{config_file}" parsed to an unexpected value of type "{configs.__class__}"'
                )

            # If we parsed an empty file, add an empty dict to work with
            if not configs:
                configs.append({})

        if len(configs) > 1 and any([c for c in configs if not c.get("optimizer")]):
            raise ValueError(
                "cannot configure a multi-servo assembly without an optimizer specified in each config"
            )

        # Set up the event bus and pub/sub exchange
        pubsub_exchange = servo.pubsub.Exchange()
        servos: List[servo.servo.Servo] = []
        for config in configs:
            # TODO: Needs to be public / have a better name
            # TODO: We need to index the env vars here for multi-servo
            servo_config_model, routes = _create_config_model(config=config, env=env)
            servo_config = servo_config_model.model_validate(config)

            telemetry = servo.telemetry.Telemetry()

            # Build the servo object
            servo_ = servo.servo.Servo(
                config=servo_config,
                telemetry=telemetry,
                pubsub_exchange=pubsub_exchange,
                routes=routes,
            )
            servo_.load_connectors()
            servos.append(servo_)

        assembly = cls(
            config_file=config_file,
            servos=servos,
        )

        # Attach all connectors to the servo
        async with asyncio.TaskGroup() as tg:
            for s in servos:
                _ = tg.create_task(s.dispatch_event(servo.servo.Events.attach, s))

        return assembly

    def __init__(self, *args, servos: List[servo.Servo], **kwargs) -> None:
        super().__init__(*args, servos=servos, **kwargs)

        # Ensure object is shared by identity
        self.servos = servos

    ##
    # Utility functions

    async def dispatch_event(
        self,
        event: Union[servo.events.Event, str],
        *args,
        first: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        prepositions: servo.events.Preposition = (
            servo.events.Preposition.before
            | servo.events.Preposition.on
            | servo.events.Preposition.after
        ),
        return_exceptions: bool = False,
        **kwargs,
    ) -> Union[Optional[servo.events.EventResult], List[servo.events.EventResult]]:
        """Dispatch an event to all servos active in the assembly."""

        group = asyncio.gather(
            *list(
                map(
                    lambda s: s.dispatch_event(
                        event,
                        *args,
                        first=first,
                        include=include,
                        exclude=exclude,
                        prepositions=prepositions,
                        **kwargs,
                    ),
                    self.servos,
                )
            ),
            return_exceptions=return_exceptions,
        )
        results = await group
        if results:
            results = functools.reduce(lambda x, y: x + y, results)

        # TODO: This needs to be tested in multi-servo
        if first:
            return results[0] if results else None

        return results

    @classmethod
    def all_connector_types(cls) -> Set[Type[servo.connector.BaseConnector]]:
        """Return a set of all connector types in the assembly excluding the Servo"""
        return servo.connector._connector_subclasses.copy()

    async def add_servo(self, servo_: servo.servo.Servo) -> None:
        """Add a servo to the assembly.

        Once added, the servo is sent the startup event to prepare for execution.

        Args:
            servo_: The servo to add to the assembly.
        """
        self.servos.append(servo_)

        await servo_.attach()

        if self.is_running:
            await servo_.startup()

    async def remove_servo(
        self, servo_: servo.servo.Servo, reason: str | None = None
    ) -> None:
        """Remove a servo from the assembly.

        Before removal, the servo is sent the detach event to prepare for
        eviction from the assembly.

        Args:
            servo_: The servo to remove from the assembly.
        """

        await servo_.detach()

        if self.is_running:
            await servo_.shutdown(reason)

        self.servos.remove(servo_)

    async def startup(self):
        """Notify all servos that the assembly is starting up."""
        await asyncio.gather(
            *list(
                map(
                    lambda s: s.startup(),
                    self.servos,
                )
            )
        )

    async def shutdown(self, reason: str | None = None):
        """Notify all servos that the assembly is shutting down."""
        await asyncio.gather(
            *list(
                map(
                    lambda s: s.shutdown(reason=reason),
                    filter(
                        lambda s: s.is_running,
                        self.servos,
                    ),
                )
            )
        )


def _discover_connectors() -> Set[Type[servo.connector.BaseConnector]]:
    """
    Discover available connectors that are registered via setuptools entry points.

    See ConnectorLoader for details.
    """
    connectors = set()
    loader = servo.connector.ConnectorLoader()
    for connector in loader.load():
        connectors.add(connector)
    return connectors


def _create_config_model_from_routes(
    routes=Dict[str, Type[servo.connector.BaseConnector]],
    *,
    require_fields: bool = True,
) -> Type[servo.configuration.BaseServoConfiguration]:
    # Create Pydantic fields for each active route
    connector_versions: Dict[Type[servo.connector.BaseConnector], str] = (
        {}
    )  # use dict for uniquing and ordering
    setting_fields: Dict[
        str, Tuple[Type[servo.configuration.BaseConfiguration], Any]
    ] = {}
    default_value = (
        ... if require_fields else None
    )  # Pydantic uses ... for flagging fields required

    for name, connector_class in routes.items():
        config_model = _derive_config_model_for_route(name, connector_class)
        config_model.model_config["title"] = (
            f"{connector_class.full_name} Settings (named {name})"
        )
        setting_fields[name] = (Optional[config_model], default_value)
        connector_versions[connector_class] = (
            f"{connector_class.full_name} v{connector_class.version}"
        )

    # Create our model
    servo_config_model: servo.configuration.BaseServoConfiguration = (
        pydantic.create_model(
            "ServoConfiguration",
            __base__=servo.configuration.BaseServoConfiguration,
            **setting_fields,
        )
    )

    connectors_series = servo.utilities.join_to_series(
        list(connector_versions.values())
    )
    servo_config_model.model_config["title"] = "Servo Configuration Schema"
    servo_config_model.model_config["json_schema_extra"] = {
        "description": f"Schema for configuration of Servo v{servo.Servo.version} with {connectors_series}"
    }
    servo_config_model.model_config["env_nested_delimiter"] = "_"
    servo_config_model.model_config["extra"] = "ignore"

    return servo_config_model


# TODO: This needs a public API and better name. Prolly moves to configuration module
def _create_config_model(
    *,
    config: Dict[str, Any],  # TODO: Could be optional?
    routes: Dict[str, Type[servo.connector.BaseConnector]] = None,
    env: Optional[Dict[str, str]] = os.environ,
) -> Tuple[
    Type[servo.configuration.BaseServoConfiguration],
    Dict[str, Type[servo.connector.BaseConnector]],
]:
    require_fields: bool = False
    # map of config key in YAML to settings class for target connector
    routes = servo.connector._default_routes() if routes is None else routes

    if isinstance(config, dict):
        # NOTE: If `connectors` key is present in the config, require the keys to be present
        connectors_value = config.get("connectors", None)
        if connectors_value:
            routes = servo.connector._routes_for_connectors_descriptor(connectors_value)
            require_fields = True

    print(f"require_fields {require_fields}")
    servo_config_model = _create_config_model_from_routes(
        routes, require_fields=require_fields
    )
    return servo_config_model, routes


# TODO: Move to the strings utility module
def _normalize_name(name: str) -> str:
    """
    Normalizes the given name.
    """
    return re.sub(r"[^a-zA-Z0-9.\-_]", "_", name)


class SettingModelCacheEntry:
    connector_type: Type[servo.connector.BaseConnector]
    connector_name: str
    config_model: Type[servo.configuration.BaseConfiguration]

    def __init__(
        self,
        connector_type: Type[servo.connector.BaseConnector],
        connector_name: str,
        config_model: Type[servo.configuration.BaseConfiguration],
    ) -> None:
        self.connector_type = connector_type
        self.connector_name = connector_name
        self.config_model = config_model

    def __str__(self):
        return f"{self.config_model} for {self.connector_type.__name__} named '{self.connector_name}'"


__config_models_cache__: List[SettingModelCacheEntry] = []


def _derive_config_model_for_route(
    name: str, model: Type[servo.connector.BaseConnector]
) -> Type[servo.configuration.BaseConfiguration]:
    """Inherits a new Pydantic model from the given settings and set up nested environment variables"""
    # NOTE: It is important to produce a new model name to disambiguate the models within Pydantic
    # because key-paths are guanranteed to be unique, we can utilize it as a cache key
    base_config_model = model.config_model()

    # See if we already have a matching model
    config_model: Optional[Type[servo.configuration.BaseConfiguration]] = None
    model_names = set()
    for cache_entry in __config_models_cache__:
        model_names.add(cache_entry.config_model.__name__)
        if cache_entry.connector_type is model and cache_entry.connector_name == name:
            config_model = cache_entry.config_model
            break

    if config_model is None:
        if base_config_model == servo.configuration.BaseConfiguration:
            # Connector hasn't defined a settings class or is reusing one, use the connector class name as base name
            # This is essential for preventing `KeyError` exceptions in Pydantic schema generation
            # due to naming conflicts.
            model_name = f"{model.__name__}Settings"
        elif (
            name == model.__default_name__
            and not f"{base_config_model.__qualname__}" in model_names
        ):
            # Connector is mounted at the default route, use default name
            model_name = f"{base_config_model.__qualname__}"
        else:
            model_name = _normalize_name(f"{base_config_model.__qualname__}__{name}")

        config_model = pydantic.create_model(model_name, __base__=base_config_model)

    # Cache it for reuse
    cache_entry = SettingModelCacheEntry(
        connector_type=model, connector_name=name, config_model=config_model
    )
    __config_models_cache__.append(cache_entry)

    return config_model
