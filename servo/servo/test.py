import typer
import inspect
import sys
from typing import ClassVar

def command(*args, **kwargs):
    def wrapper(fn):
        class Command:
            def __init__(self, fn):
                self.fn = fn

            def __set_name__(self, owner, name):
                setattr(owner, name, self.fn)
                owner._typer.command(*args, **kwargs)(self.fn)
        return Command(fn)
    return wrapper

def callback(*args, **kwargs):
    def wrapper(fn):
        class Callback:
            def __init__(self, fn):
                self.fn = fn

            def __set_name__(self, owner, name):
                owner._typer.callback(*args, **kwargs)(self.fn)
        return Callback(fn)
    return wrapper

def jsonschema(cls):
    def schema(self):
        """
        Display the JSON Schema 
        """
        pass

    def validate(file: typer.FileText = typer.Argument(...)):
        """
        Validate given file against the JSON Schema
        """
        pass
    
    # Attach the methods and directly invoke Typer decorator
    setattr(cls, 'schema', schema)
    cls._typer.command()(schema)
    setattr(cls, 'validate', validate)
    cls._typer.command()(validate)
    return cls

@jsonschema
class Connector():
    name: ClassVar[str] = "foo" # TODO: Need to figure out the name factoring
    _typer: ClassVar[typer.Typer] = typer.Typer(name=name)
    
    @command()
    def another():
        print("another")

class VegetaConnector(Connector):
    name = "vegeta"
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