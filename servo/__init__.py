import importlib.metadata
import pathlib
from typing import Optional

import toml


def __get_version() -> Optional[str]:
    path = pathlib.Path(__file__).resolve().parents[1] / 'pyproject.toml'

    if path.exists():
        pyproject = toml.loads(open(str(path)).read())
        return pyproject['tool']['poetry']['version']
    else:
        try:
            return importlib.metadata.version("servox")
        except importlib.metadata.PackageNotFoundError:
            pass

    return None

__version__ = __get_version() or "0.0.0"
__cryptonym__ = "baseless allegation"

# Add the devtools debug() function to builtins if available
import builtins

import devtools

builtins.debug = devtools.debug

# Promote all symbols from submodules to the top-level package
from .assembly import *
from .checks import *
from .configuration import *
from .connector import *
from .errors import *
from .events import *
from .logging import *
from .pubsub import *
from .servo import *
from .types import *
from .utilities import *

# Resolve forward references
servo.events.EventResult.update_forward_refs()
servo.events.EventHandler.update_forward_refs()
