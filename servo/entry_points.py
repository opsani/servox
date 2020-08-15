# Entry points for executing functionality defined in other modules
# This wrapping is necessary for keeping Docker builds fast because
# the entry points must be present on the filesystem during package
# installation in order to be registered on the path and fast moving
# sources will unnecessarily bust the Docker cache and trigger full
# reinstalls of all package dependencies.
# Do not implement meaningful functionality here. Instead import and
# dispatch the intent into focused modules to do the real work.
from dotenv import find_dotenv, load_dotenv

from servo.cli import ServoCLI
from servo.connector import ConnectorLoader
from servo.logging import logger

def run_cli():
    load_dotenv(find_dotenv(usecwd=True))

    # NOTE: We load connectors here because waiting until assembly
    # is too late for registering CLI commands
    try:
        for connector in ConnectorLoader().load():
            logger.debug(f"Loaded {connector.__qualname__}")
    except Exception:
        logger.exception("failed loading connectors via discovery", backtrace=True, diagnose=True)

    cli = ServoCLI()
    cli()
