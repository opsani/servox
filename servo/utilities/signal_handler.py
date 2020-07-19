import sys
import signal
from typing import Callable, Optional

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
        if self.stop_callback:
            self.stop_callback(sig_num)

    def _graceful_restart_handler(self, sig_num, unused_frame):
        """handle signal for restart - simply set a flag to have the main loop exit and restart the process after the current operation is completed"""
        if self.restart_callback:
            self.restart_callback(sig_num)
