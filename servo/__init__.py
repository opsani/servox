import importlib.metadata

for pkg in {"servo", "servox"}:
    try:
        __version__ = importlib.metadata.version(pkg)
        break
    except importlib.metadata.PackageNotFoundError:
        pass

__codename__ = "pass the calimari"

# Add the devtools debug() function to builtins if available
try:
    import builtins
    import devtools
    
    builtins.debug = devtools.debug
except ImportError:
    pass

# Promote all symbols from submodules to the top-level package
from .assembly import *
from .configuration import *
from .connector import *
from .checks import *
from .events import *
from .errors import *
from .logging import *
from .servo import *
from .types import *
from .utilities import *
