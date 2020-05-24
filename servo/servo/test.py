import typer
import inspect
import sys
from typing import ClassVar

def command(fn):
    class Command:
        def __init__(self, fn):
            self.fn = fn

        def __set_name__(self, owner, name):
            owner._typer.command()(self.fn)
    return Command(fn)

class JSONSchemaMixin:
    pass

class Connector():
    name: ClassVar[str] = "foo" # TODO: Need to figure out the name factoring
    _typer: ClassVar[typer.Typer] = typer.Typer(name=name)
    
    @command
    def another():
        print("another")

class VegetaConnector(Connector):
    # name = "vegeta"
    # typer = typer.Typer(name=name)

    @command
    def wtf():
        print("wtf")

class PrometheusConnector(Connector):
    pass

class CLI:
    app = typer.Typer()

    def __init__(self) -> None:
        super().__init__()
        self.add_connector(VegetaConnector())
    
    def add_connector(self, connector):
        self.app.add_typer(connector._typer)
    
    @app.command()
    def test():
        print("test")

if __name__ == "__main__":
    cli = CLI()
    cli.app()