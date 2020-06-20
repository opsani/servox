import typer
import inspect
import sys

# OPSANI_OPTIMIZER (--optimizer -o no default)
# OPSANI_TOKEN (--token -t no default)
# OPSANI_TOKEN_FILE (--token-file -T ./servo.token)
# OPSANI_CONFIG_FILE (--config-file -c ./servo.yaml)

app = typer.Typer(name="servox", add_completion=True)

# TODO: Discover all the connector subclasses and let them register commands

def root_callback() -> None:
    pass

@app.command()
def new() -> None:
    """Creates a new servo assembly at [PATH]"""
    pass

@app.command()
def run() -> None:
    """Run the servo"""
    pass

@app.command()
def console() -> None:
    """Open an interactive console"""
    pass

@app.command()
def info() -> None:
    '''Display information about the assembly'''
    pass

@app.command()
def settings() -> None:
    '''Display the fully resolved settings'''
    pass

@app.command()
def check() -> None:
    '''Check the health of the assembly'''
    pass

@app.command()
def version() -> None:
    '''Display version and exit'''
    pass

### Begin config subcommands
# TODO: It may make more sense to have these top-level
config_app = typer.Typer(name='config', help="Manage configuration")
app.add_typer(config_app)

@app.command()
def config() -> None:
    """Display servo configuration"""
    pass

@config_app.command(name='schema')
def config_schema() -> None:
    '''Display configuration schema'''
    pass

@config_app.command(name='validate')
def config_validate() -> None:
    """Validate servo configuration file"""
    pass

@config_app.command(name='generate')
def config_generate() -> None:
    """Generate servo configuration"""
    pass

### Begin connector subcommands
connector_app = typer.Typer(name='connector', help="Manage connectors")
app.add_typer(connector_app)

@connector_app.command(name='list')
def connectors_list() -> None:
    '''List connectors in the assembly'''
    pass

@connector_app.command(name='add')
def connectors_add() -> None:
    '''Add a connector to the assembly'''
    pass

@connector_app.command(name='remove')
def connectors_remove() -> None:
    '''Remove a connector from the assembly'''
    pass

### Begin developer subcommands
# NOTE: registered as top level commands for convenience in dev

@app.command(name='test')
def developer_test() -> None:
    '''Run automated tests'''
    pass

@app.command(name='lint')
def developer_lint() -> None:
    '''Emit opinionated linter warnings and suggestions'''
    pass

@app.command(name='format')
def developer_format() -> None:
    '''Apply automatic formatting to the codebase'''
    pass

# Run the Typer CLI
def main():
    app()
