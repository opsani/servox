import importlib.metadata

for pkg in {"servo", "servox"}: # pragma: no cover
    try:
        __version__ = importlib.metadata.version(pkg)
        break
    except importlib.metadata.PackageNotFoundError:
        pass

__codename__ = "pass the calamari"

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
from .servo import *
from .types import *
from .utilities import *

# Resolve forward references
servo.events.EventResult.update_forward_refs()
servo.events.EventHandler.update_forward_refs()
