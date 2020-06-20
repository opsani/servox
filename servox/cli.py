import typer
import inspect
import sys

# OPSANI_OPTIMIZER (--optimizer -o no default)
# OPSANI_TOKEN (--token -t no default)
# OPSANI_TOKEN_FILE (--token-file -T ./servo.token)
# OPSANI_CONFIG_FILE (--config-file -c ./servo.yaml)
def root_callback() -> None:
    pass

def run() -> None:
    """Run the servo"""
    pass

def console() -> None:
    """Open an interactive console"""
    pass

def version() -> None:
    pass

## TODO: moves to config.py
def config() -> None:
    """Display servo configuration"""
    pass

def config_schema() -> None:
    pass

def config_validate() -> None:
    """Validate servo configuration file"""
    pass

def config_generate() -> None:
    """Generate servo configuration"""
    pass

def connectors() -> None:
    pass

def connectors_add() -> None:
    pass

def connectors_remove() -> None():
    pass

## TODO: Moves to developer.py
def developer_test() -> None:
    pass

def developer_lint() -> None:
    pass

def developer_format() -> None:
    pass

# TODO: Token probably needs to come from volume...
class Config:
    app: str
    token: str
    
    def __init__(self, app: str, token: str) -> None:
        self.app = app
        self.token = token

_registry = {}

def register_class(cls):
    _registry[cls.__name__] = cls

# TODO: Support abstract connector types like Adjust and Measure

# Wrapper for Typer.command
def command():    
    def decorator(f):
        print(f.__globals__)
        print(f.__qualname__)
        # c = f.__globals__[f.__qualname__.rsplit('.', 1)[0]]    
        # c = getattr(inspect.getmodule(f), f.__qualname__.rsplit('.', 1)[0])
        name = f.__qualname__.rsplit('.', 1)[0]
        cls = eval(name)
        print(cls)
        print('-------')
        print(_registry)
        cls = _registry[name]
        print(cls)
        # if cls.cli == None:
        #     cls.cli = typer.Typer()
        return cls.command.command(f)
    return decorator

# TODO: Do an abstract base class?
# TODO: Get all classes registered
class MyMeta(type):
    def __new__(self,name,base,ns):
        print("meta new ", self)
        return type.__new__(self,name,base,ns)

    def __init__(self,name,base,ns):
        print("meta init ", self)
        register_class(self)
        self.command = typer.Typer
        type.__init__(self,name,base,ns)

# Every command gets its own context
class Command(metaclass=MyMeta):
    command: typer.Typer = None

# Defines the basic commands for JSON schemas
class JSONSchemaCommandsMixin(Command):

    print("ABOUT TO COMMAND")
    # @command()
    def schema(self):
        """
        Display the JSON Schema 
        """
        pass

    # @command()
    def validate(self, file: typer.FileText = typer.Argument(...)):
        """
        Validate given file against the JSON Schema
        """
        pass

# Abstract CLI base class. Provides no commands
class CLI(Command):
    # Define the root command for our CLI
    # command = typer.Typer()

    # TODO: Maybe this becomes an @property
    @classmethod
    def root_command(cls):
        register_class(cls)
        cls.command = typer.Typer()

    # Access the root command
    def get_root_command(self) -> typer.Typer:
        return CLI.command
    
    # Add a new subcommand under the parent
    # @classmethod
    # def add_subcommand(cls, name) -> typer.Typer:
    #     register_class(cls)
    #     parent = cls.command
    #     command = typer.Typer()
    #     parent.add_typer(command, name=name)
    #     cls.command = command
    #     return command
    
    # @classmethod
    # def __init_subclass__(cls, **kwargs):
    #     _registry[cls.__name__] = cls
    #     print("Registered class {}", cls)
    #     super().__init_subclass__(**kwargs)
        # Add to the class registry
    #     print(f'In here!')
    #     cls.cli = typer.Typer()

# Sets the class as the root of a CLI hierarchy
# def root_command(cls):
#     print("\n\n\nIN ROOT COMMAND\n\n")
#     cls.command = typer.Typer()
#     return cls
    # class RootCommand(object):
    #     def __init__(self, *args):
    #         self.wrapped = cls(*args)

    #     def __getattr__(self, name):
    #         print('Getting the {} of {}'.format(name, self.wrapped))
    #         return getattr(self.wrapped, name)

    # return Wrapper

# def subcommand(name):
#     def decorator(cls):
#         cls.add_subcommand(name)
#         return cls
#     return decorator

# CLI
# Command
# Connector
# Vegeta
# Prometheus
# Kubernetes

class Connector(Command, JSONSchemaCommandsMixin):
    # Every connector has its own context

    # command = root_command()
    # root_command()
    
    name: str
    # app: typer.Typer = typer.Typer()
    # app = typer.Typer()
    
    # def __init__(self) -> None:
    #     self.app = typer.Typer()

    # cli: typer.Typer = typer.Typer()

    # @classmethod
    # def subcommand(cls, name):
    #     parent_command = cls.cli
    #     cls.cli = typer.Typer()

    # def typer(self) -> typer.Typer:
    #     if self._typer == None:
    #         self._typer = typer.Typer()
    #     return self._typer
            
        # if self._typer == None:
        #     self._typer = typer.Typer()
        # # return self._typer(fn)
        # def decorated(*args,**kwargs):
        #     return fn(*args,**kwargs)                         
        # return decorated 
    # @classmethod
    # def cli(cls):
    #     if cls._cli == None:
    #         cls._cli = typer.Typer()
    #     return cls._cli

    # When this is called:
    # Ensure that the class instance is populated
    # pass the call through with the args to the typer
    # def command(*args,**kwargs):
    # @staticmethod    
    # def command(self, f):
    #     print(f)
        # TODO: We have to capture all the args on the outside
        # def decorator(f):
        #     def wrapper(self, *args):
        #         # cls.cli().command(*args,**kwargs)
        #         print getattr(self, attribute)
        #         return f(self, *args)            
        # return decorator
        # self.cli().command()

    # @cli.command()
    # def info(self):
    #     """
    #     Display info about the servo
    #     """
    #     pass

    # # @cli.command()
    # def run(self):
    #     """
    #     Run the servo
    #     """
    #     pass

class VegetaConnector(Connector):
    # command = CLI.add_subcommand()
    # name = "vegeta"
    # app = typer.Typer()

    # @cli.command()
    # @command()
    print("aadsds")
    def loadgen(self):
        """
        Run the servo
        """
        pass

class Servo(Connector):
    vegeta: VegetaConnector

    def __init__(self) -> None:
        super().__init__()
        vegeta = VegetaConnector()
        # self.cli.add_typer(vegeta.cli, name=vegeta.name)
        self.vegeta = vegeta        

servo = Servo()

if __name__ == "__main__":
    servo.command()


# cli = CLI()
# if __name__ == "__main__":
#     cli.command()

# TODO: Need classes: Config, Connector, Servo
# Servo is the root class

# app = typer.Typer()

# # For sub-commands: https://typer.tiangolo.com/tutorial/subcommands/add-typer/
# # class Config:


# @app.command()
# def schema():
#     """
#     Display the JSON Schema 
#     """
#     pass


# @app.command()
# def validate(file: typer.FileText = typer.Argument(...)):
#     """
#     Validate given file against the JSON Schema
#     """
#     pass

# @app.command()
# def info():
#     """
#     Display info about the servo
#     """
#     pass

# @app.command()
# def run():
#     """
#     Run the servo
#     """
#     pass



# # if __name__ == "__main__":
# #     app()
