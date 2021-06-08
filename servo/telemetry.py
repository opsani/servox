import os
import platform
from typing import Dict
import pydantic

import servo

class Telemetry(pydantic.BaseModel):
    """Class and convenience methods for storage of arbitrary servo metadata
    """

    _values: Dict[str, str] = pydantic.PrivateAttr(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self["servox.version"] = str(servo.__version__)
        self["servox.platform"] = platform.platform()

        if servo_ns := os.environ.get("POD_NAMESPACE"):
            self["servox.namespace"] = servo_ns

    def __getitem__(self, k: str) -> str:
        return self._values.__getitem__(k)

    def __setitem__(self, k: str, v: str) -> None:
        self._values.__setitem__(k, v)

    def remove(self, key: str) -> None:
        """Safely remove an arbitrary key from telemetry metadata"""
        self._values.pop(key, None)


    @property
    def values(self) -> Dict[str, Dict[str, str]]:
        # TODO return copy to ensure read only?
        return self._values
