# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import pathlib
from typing import Optional

import toml


def __get_version() -> Optional[str]:
    path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"

    if path.exists():
        pyproject = toml.loads(open(str(path)).read())
        return pyproject["tool"]["poetry"]["version"]
    else:
        try:
            return importlib.metadata.version("servox")
        except importlib.metadata.PackageNotFoundError:
            pass

    return None


__version__ = __get_version() or "0.0.0"
__cryptonym__ = "genesis"

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
from .types.api import *
from .types.core import *
from .types.settings import *
from .types.slo import *
from .utilities import *

# Resolve forward references
servo.events.EventResult.update_forward_refs()
servo.events.EventHandler.update_forward_refs()
