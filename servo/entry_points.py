# Entry points for executing functionality defined in other modules
# This wrapping is necessary for keeping Docker builds fast because
# the entry points must be present on the filesystem during package
# installation in order to be registered on the path and fast moving
# sources will unnecessarily bust the Docker cache and trigger full
# reinstalls of all package dependencies.
# Do not implement meaningful functionality here. Instead import and
# dispatch the intent into focused modules to do the real work.
import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from servo.cli import cli, connectors_to_update
from servo.connector import ConnectorLoader
from servo.servo import _default_routes, _routes_for_connectors_descriptor


def run_cli():
    load_dotenv()

    for connector in ConnectorLoader().load():
        logger.info(f"Loaded {connector}")

    routes = _default_routes()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config-file")
    namespace, r = parser.parse_known_args()
    if namespace.config_file:
        config_file = namespace.config_file
    else:
        config_file = os.getenv("SERVO_CONFIG_FILE", "servo.yaml")
    if Path(config_file).exists():
        try:
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            if isinstance(config, dict):  # Config file could be blank or malformed
                connectors_value = config.get("connectors", None)
                if connectors_value:
                    routes = _routes_for_connectors_descriptor(connectors_value)
        except (ValueError, TypeError) as error:
            logger.warning(
                f'Warning: an unexpected error was encountered while processing config "{config_file}": ({error})',
                file=sys.stderr,
            )
            routes = {}

    for path, connector_class in routes.items():
        settings = connector_class.settings_model().construct()
        connector = connector_class(settings)
        connectors_to_update.append(connector)
        connector_cli = connector.cli()
        if connector_cli is not None:
            cli.add_typer(connector_cli)

    cli()
