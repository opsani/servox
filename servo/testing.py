from servo.connector import Connector
from servo.cli import ConnectorCLI


class TestConnector(Connector):
    def cli(self) -> ConnectorCLI:
        cli = ConnectorCLI(self, name="testing", help="Testing connector")

        @cli.command()
        def testing():
            """
            Just a test
            """

        return cli
