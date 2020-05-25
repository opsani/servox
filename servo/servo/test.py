import typer
import inspect
import sys
from typing import ClassVar, Any, Optional, ClassVar

def command(*args, **kwargs):
    def wrapper(fn):
        print(f"command called on {fn}")
        class Command:
            def __init__(self, fn):
                self.fn = fn

            def __set_name__(self, owner, name):
                print(f"self {self} setting {owner} to {name}")
                # TODO: At this point we are good to go and can hydrate anything that is nil
                # self.create_typer_if_necessary(owner)
                # owner.name = owner.__name__
                # owner.typer().command(*args, **kwargs)(self.fn)
                owner._cmd.command(*args, **kwargs)(self.fn)

        return Command(fn)
    return wrapper

class ConnectorMeta(type):
    def __init__(self,name,base,ns):
        type.__init__(self,name,base,ns)
        conn_name = name.replace('Connector', '').lower()
        # if self._cmd is None:
        if hasattr(self, "_cmd") == False:
            print(f"Setting new _cmd ref for connector {conn_name}")
            self._cmd = typer.Typer(name=conn_name)

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
    cls._cmd.command()(schema)
    setattr(cls, 'validate', validate)
    cls._cmd.command()(validate)
    return cls

@jsonschema
class Connector(metaclass=ConnectorMeta):
    def name(self) -> str:
        return self.__class__.__name__.replace('Connector', '').lower()
    
    def add_connector(self, conn: 'Connector') -> None:
        self._cmd.add_typer(conn._cmd, name=conn.name())

class VegetaConnector(Connector):
    @command()
    def loadgen():
        pass

class PrometheusConnector(Connector):
    @command()
    def measure():
        pass

# make a singleton?
def root_connector() -> Connector:
    class RootConnector(Connector):        
        def run(self) -> None:
            self._cmd()

        def __call__(self) -> Any:
            return self.run()

        @command()
        def listen():
            pass

    root_cmd = RootConnector()
    conn_name = root_cmd.__class__.__name__.replace('Connector', '').lower()
        # if self._cmd is None:
        #     print(f"Setting new _cmd ref for connector {conn_name}")
        #     self._cmd = typer.Typer(name=conn_name)
    root_cmd._cmd = typer.Typer(name=conn_name)
    return root_cmd

root = root_connector()
vegeta = VegetaConnector()
root.add_connector(vegeta)

if __name__ == "__main__":
    root()

# def command(*args, **kwargs):
#     def wrapper(fn):
#         print(f"command called on {fn}")
#         class Command:
#             def __init__(self, fn):
#                 self.fn = fn

#             def __set_name__(self, owner, name):
#                 print(f"self {self} setting {owner} to {name}")
#                 # TODO: At this point we are good to go and can hydrate anything that is nil
#                 # self.create_typer_if_necessary(owner)
#                 owner.name = owner.__name__
#                 # owner.typer().command(*args, **kwargs)(self.fn)
#                 owner.cli.command(*args, **kwargs)(self.fn)
            
#             # def create_typer_if_necessary(self, owner) -> None:
#             #     if owner.name is None:
#             #         owner.name = owner.__name__
#             #         owner._typer = typer.Typer(name=owner.name)

#         return Command(fn)
#     return wrapper

# def callback(*args, **kwargs):
#     def wrapper(fn):
#         class Callback:
#             def __init__(self, fn):
#                 self.fn = fn

#             def __set_name__(self, owner, name):
#                 owner.typer().callback(*args, **kwargs)(self.fn)
#         return Callback(fn)
#     return wrapper

# def jsonschema(cls):
#     def schema(self):
#         """
#         Display the JSON Schema 
#         """
#         pass

#     def validate(file: typer.FileText = typer.Argument(...)):
#         """
#         Validate given file against the JSON Schema
#         """
#         pass
    
#     # Attach the methods and directly invoke Typer decorator
#     setattr(cls, 'schema', schema)
#     cls.cli.command()(schema)
#     setattr(cls, 'validate', validate)
#     cls.cli.command()(validate)
#     return cls

# @jsonschema
# class Connector:
#     cli = typer.Typer()
#     # name: ClassVar[str] = None #"foo" # TODO: Need to figure out the name factoring
#     # _typer: ClassVar[typer.Typer] = None #typer.Typer(name=name)
#     # 

#     # @classmethod
#     # def typer(cls: 'Connector') -> typer.Typer:
#     #     if cls._typer is None:
#     #         cls._typer = typer.Typer(name=cls.name)
#     #     return cls._typer

#     # def create_typer_if_necessary(cls: 'Connector') -> None:
#     #     if cls.name is None:
#     #         cls.name = cls.__name__
#     #         cls._typer = typer.Typer(name=cls.name)

#     # @command()
#     # def another():
#     #     print("another")
    
#     # print("DADASDAS")
#     # @command()
#     # def testinggggg():
#     #     print("wtf")

# class VegetaConnector(Connector):
#     # _typer = typer.Typer(name="vegeta")
#     # cli.info.name = "sadasds"
#     # name = "vegeta"
#     # typer = typer.Typer(name=name)

#     @command()
#     def wtf():
#         print("wtf")

# class PrometheusConnector(Connector):
#     pass

# class CLI(Connector):
#     # Since we are the root command, explicitly set the base state
#     # __name: ClassVar[str] = None
#     # _typer: ClassVar[typer.Typer] = typer.Typer()
#     # cli = typer.Typer()

#     def __init__(self) -> None:
#         super().__init__()
#         #self.add_connector(VegetaConnector())
    
#     def add_connector(self, connector):
#         print(f"Adding connector {connector.typer()} to typer {self.typer()}")
#         # self.typer().add_typer(connector.typer())
#         # self.typer.add_typer(connector.typer())
    
#     @command()
#     def test():
#         print("test")
    
#     def run(self):
#         print(f"Executing typer instance {self.cli}")
#         self.cli()


# if __name__ == "__main__":
#     cli = CLI()
#     cli.run()