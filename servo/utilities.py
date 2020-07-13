import signal
import sys
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional, Union

import durationpy
from pydantic import BaseModel, validator


def join_to_series(
    items: Iterable[str], *, conjunction="and", oxford_comma=True
) -> str:
    """
    Concatenate any number of strings into a series suitable for use in English output.

    Items are joined using a comma and a configurable conjunction, defaulting to 'and'.
    """
    count = len(items)
    if count == 0:
        return ""
    elif count == 1:
        return items[0]
    elif count == 2:
        return f" {conjunction} ".join(items)
    else:
        series = ", ".join(items[0:-1])
        last_item = items[-1]
        delimiter = "," if oxford_comma else ""
        return f"{series}{delimiter} {conjunction} {last_item}"


class DurationProgress(BaseModel):
    duration: Union[str, timedelta]
    started_at: Optional[datetime]

    def __init__(self, duration: Union[str, timedelta], **kwargs) -> None:
        super().__init__(duration=duration, **kwargs)

    @validator("duration", pre=True)
    def coerce_duration(cls, duration) -> timedelta:
        if isinstance(duration, str):
            try:
                return durationpy.from_str(duration)
            except Exception as e:
                raise ValueError(str(e)) from e
        return duration

    def start(self) -> None:
        assert not self.is_started()
        self.started_at = datetime.now()

    def is_started(self) -> bool:
        return self.started_at is not None

    def is_completed(self) -> bool:
        return self.progress() >= 100

    def progress(self) -> float:
        return min(100.0, 100.0 * (self.elapsed()) / self.duration)

    def elapsed(self) -> timedelta:
        return datetime.now() - self.started_at

    def annotate(self, str_to_annotate: str) -> str:
        elapsed = durationpy.to_str(self.elapsed())
        return f"{self.progress():.2f}% complete, {elapsed} elapsed - {str_to_annotate}"


SignalCallback = Callable[[int], None]


class SignalHandler:
    """
    Provides an interface for handling common UNIX signals.

    Handles `SIGTERM` and `SIGINT` for termination and
    `SIGUSR1` and `SIGHUP` for stop and restart, respectively.

    Callbacks are provided for hooking arbitrary logic into
    the signal handling process.
    """

    stop_callback: Optional[SignalCallback] = None
    restart_callback: Optional[SignalCallback] = None
    terminate_callback: Optional[SignalCallback] = None

    def __init__(
        self,
        *,
        stop_callback: Optional[SignalCallback] = None,
        restart_callback: Optional[SignalCallback] = None,
        terminate_callback: Optional[SignalCallback] = None,
    ):
        # intercept SIGINT to provide graceful, traceback-less Ctrl-C/SIGTERM handling
        signal.signal(signal.SIGTERM, self._signal_handler)  # container kill
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl-C
        signal.signal(signal.SIGUSR1, self._graceful_stop_handler)
        signal.signal(signal.SIGHUP, self._graceful_restart_handler)

        self.stop_callback = stop_callback
        self.restart_callback = restart_callback
        self.terminate_callback = terminate_callback

        super().__init__()

    def _signal_handler(self, sig_num, unused_frame):
        # restore original signal handler (to prevent reentry)
        signal.signal(sig_num, signal.SIG_DFL)

        # invoke the callback
        if self.terminate_callback:
            self.terminate_callback(sig_num)

        sys.exit(0)

    def _graceful_stop_handler(self, sig_num, unused_frame):
        """handle signal for graceful termination - simply set a flag to have the main loop exit after the current operation is completed"""
        # self._stop_flag = "exit"
        if self.stop_callback:
            self.stop_callback(sig_num)

    def _graceful_restart_handler(self, sig_num, unused_frame):
        """handle signal for restart - simply set a flag to have the main loop exit and restart the process after the current operation is completed"""
        # self._stop_flag = "restart"
        if self.restart_callback:
            self.restart_callback(sig_num)
