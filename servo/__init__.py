import importlib.metadata

for pkg in {"servo", "servox"}:
    try:
        __version__ = importlib.metadata.version(__name__)
        break
    except importlib.metadata.PackageNotFoundError:
        pass

import servo.connector
import servo.events
import servo.types
import servo.cli
import servo.utilities
import servo.logging

# Import the core classes
# These are what most developers will need
from .events import (
    Event,
    EventHandler,
    EventResult, 
    Preposition,
    EventError,
    CancelEventError,
)
from .connector import (
    Optimizer,
    BaseConfiguration,
    Connector,
    metadata,
    event,
    before_event,
    on_event,
    after_event,
    event_handler,
)

# Pull the types up to the top level
from .types import *
