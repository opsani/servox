# Entry points for executing functionality defined in other modules
# This wrapping is necessary for keeping Docker builds fast because
# the entry points must be present on the filesystem during package
# installation in order to be registered on the path and fast moving
# sources will unnecessarily bust the Docker cache and trigger full
# reinstalls of all package dependencies.
# Do not implement meaningful functionality here. Instead import and
# dispatch the intent into focused modules to do the real work.
from optparse import OptionParser
from dotenv import load_dotenv
from servo.connector import ConnectorLoader
from servo.cli import cli, connectors_to_update

def run_cli():
    # parser.add_option("-f", "--file", dest="file", metavar="FILE", default="servo.yaml")

    # (options, args) = parser.parse_args()

    # if path := Path(options.file).exists():
    #     pass
    #     # TODO: if connectors is empty, activate everything
    #     # If there are aliases, activate those


    # debug(options, args)
    # # return


    # FIXME: This should be handled after parsing the options but Click doesn't make it super easy
    # Only active connectors should be registered as commands (and aliases should be registered as well)
    loader = ConnectorLoader()
    for connector in loader.load():
        settings = connector.settings_model().construct()
        connector = connector(settings)
        connectors_to_update.append(connector)
        connector_cli = connector.cli()
        if connector_cli is not None:
            cli.add_typer(connector_cli)

    load_dotenv()
    cli()
