from __future__ import annotations

import datetime
from typing import Optional

__all__ = (
    "BaseError",
    "ServoError",
    "ConnectorError",
    "EventError",
    "EventCancelledError",
    "AdjustmentFailedError",
    "AdjustmentRejectedError",
    "UnexpectedEventError",
)

class BaseError(RuntimeError):
    """The base class for all errors in the servo package."""

    def __init__(
        self,
        message: str = '',
        reason: Optional[str] = None,
        *args,
        assembly: Optional[servo.Assembly] = None,
        servo_: Optional[servo.Servo] = None,
        connector: Optional[servo.Connector] = None,
        event: Optional[servo.Event] = None,
    ) -> None:
        super().__init__(message, *args)

        # Use the context vars to infer the assembly, servo, connector, and event
        import servo
        self._reason = reason
        self._assembly = assembly or servo.current_assembly()
        self._servo = servo_ or servo.current_servo()
        self._connector = connector or getattr(self._servo, "connector", None)
        self._event = event or getattr(self._servo, "event", None)
        self._created_at = datetime.datetime.now()

    @property
    def reason(self) -> Optional[str]:
        """A supplemental reason explaining why the error occurred."""
        return self._reason

    @reason.setter
    def reason(self, value: Optional[str]) -> None:
        self._reason = value

    @property
    def created_at(self) -> datetime.datetime:
        """The date and time when the error occurred."""
        return self._created_at

    @property
    def assembly(self) -> servo.Assembly:
        """The assembly in which the error occurred."""
        return self._assembly

    @property
    def servo(self) -> Optional[servo.Servo]:
        """The servo that was active when the error occurred."""
        return self._servo

    @property
    def connector(self) -> Optional[servo.Connector]:
        """The connector that was active when the error occurred."""
        return self._connector

    @property
    def event(self) -> Optional[servo.Event]:
        """The event that was executing when the error occurred."""
        return self._event

class ServoError(BaseError):
    """An error occurred within a servo."""
    @property
    def servo(self) -> servo.Servo:
        return self._servo

class ConnectorError(ServoError):
    """An error occurred within a connector."""
    @property
    def connector(self) -> servo.Connector:
        return self._connector

class EventError(ConnectorError):
    """An error occurred during the processing of an event by a connector."""
    @property
    def event(self) -> servo.Event:
        return self._event

class UnexpectedEventError(EventError):
    """The optimizer reported that an unexpected error was submitted."""

class EventCancelledError(EventError):
    """The event was cancelled and processing was halted."""

class AdjustmentFailedError(EventError):
    """A failure occurred while attempting to perform an adjustment.

    Adjustment failures are potentially recoverable errors in which the
    adjustment was not fully applied due to a transient failure, lost
    connection, interruption, etc. and be retried by the optimizer.
    """

class AdjustmentRejectedError(AdjustmentFailedError):
    """The adjustment was irrecoverably rejected when applied.

    Rejections occur in circumstances where the target application fails
    to start, becomes unstable, the orchestrator refuses to apply it, or
    other such definitive error condition is encountered that excludes the
    applied configuration from further consideration by the optimizer.
    """
