# Entry points for executing functionality defined in other modules
# This wrapping is necessary for keeping Docker builds fast because
# the entry points must be present on the filesystem during package
# installation in order to be registered on the path and fast moving
# sources will unnecessarily bust the Docker cache and trigger full
# reinstalls of all package dependencies.
# Do not implement meaningful functionality here. Instead import and
# dispatch the intent into focused modules to do the real work.
# noqa

import dotenv
import uvloop

import servo
import servo.cli


def run_cli() -> None:
    """Run the Servo CLI."""
    uvloop.install()
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

    # NOTE: We load connectors here because waiting until assembly
    # is too late for registering CLI commands
    try:
        for connector in servo.connector.ConnectorLoader().load():
            servo.logger.debug(f"Loaded {connector.__qualname__}")
    except Exception:
        servo.logger.exception(
            "failed loading connectors via discovery", backtrace=True, diagnose=True
        )

    cli = servo.cli.ServoCLI()
    cli()
