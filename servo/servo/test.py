import typer
import inspect
import sys
from typing import ClassVar

# _registry = {}

# def register_class(cls):
#     _registry[cls.__name__] = cls

# class MyMeta(type):
#     # def __new__(self,name,base,ns):
#     #     print("meta new ", self)
#     #     return type.__new__(self,name,base,ns)

#     def __init__(self,name,base,ns):
#         print("meta init ", self)
#         register_class(self)
#         self.command = typer.Typer
#         type.__init__(self,name,base,ns)

# def command():    
#     def decorator(f):
#         print(f)
#         print(f.__globals__)
#         print(f.__qualname__)
#         # c = f.__globals__[f.__qualname__.rsplit('.', 1)[0]]    
#         # c = getattr(inspect.getmodule(f), f.__qualname__.rsplit('.', 1)[0])
#         name = f.__qualname__.rsplit('.', 1)[0]
#         print("NAME: ", name)
#         # cls = eval(name)
#         # print(cls)
#         print('-------')
#         print(_registry)
#         cls = _registry[name]
#         print(cls)
#         # if cls.cli == None:
#         #     cls.cli = typer.Typer()
#         return cls.command.command(f)
#     return decorator

def command(fn):
    print(f"command invoked for {fn}")
    class Command:
        def __init__(self, fn):
            self.fn = fn

        def __set_name__(self, owner, name):
            print(f"decorating {self.fn} and name {name} using {owner}")
            self.fn.class_name = owner.__name__

            # then replace ourself with the original method
            # setattr(owner, name, self.fn)
            # print(f"Typer is {owner._typer}")
            owner._typer.command()(self.fn)
    return Command(fn)

class JSONSchemaMixin:
    pass

class Connector():
    name: ClassVar[str] = "foo" # TODO: Need to figure out the name factoring
    _typer: ClassVar[typer.Typer] = typer.Typer(name=name)
    
    # @classmethod
    # def command(cls, *args,**kwargs):
    #     return cls._typer.command(*args,**kwargs)

    # @_typer.command()
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