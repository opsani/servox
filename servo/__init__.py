import importlib.metadata

for pkg in {"servo", "servox"}:
    try:
        __version__ = importlib.metadata.version(pkg)
        break
    except importlib.metadata.PackageNotFoundError:
        pass

__codename__ = "pass the calimari"

import servo.assembly
import servo.connector
import servo.connectors
import servo.errors
import servo.events
import servo.types
import servo.cli
import servo.utilities

# Import the core classes
# These are what most developers will need
from .events import (
    Event,
    EventHandler,
    EventResult,
    Preposition,
    create_event,
    event,
    before_event,
    on_event,
    after_event,
    event_handler,
)
from .connector import (
    BaseConfiguration,
    BaseConnector,
    Optimizer,
    metadata,
)

from .configuration import *
from .checks import *
from .errors import *
from .logging import *
from .types import *
from .utilities import *
