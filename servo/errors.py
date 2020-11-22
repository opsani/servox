from servo.events import CancelEventError, EventError

__all__ = (
    "AdjustmentError",
    "AdjustmentFailure",
    "AdjustmentRejection",
    "DescriptionError",
    "MeasurementError",
    "CancelEventError",
    "ConnectorError",
    "EventError",
)

# TODO: Create status and reason enums

class ConnectorError(Exception):
    """Exception indicating that a connector failed"""

    def __init__(self, *args, status="failed", reason="unknown") -> None: # noqa: D107
        self.status = status
        self.reason = reason
        super().__init__(*args)

class CommandError(EventError): # TODO: Operation error?
    def __init__(self, *args, status="failed", reason="unknown") -> None: # noqa: D107
        self.status = status
        self.reason = reason
        super().__init__(*args)

    # TODO: add command enum link?

class DescriptionError(CommandError):
    ...

class MeasurementError(CommandError):
    ...

class AdjustmentError(CommandError):
    def __init__(self, *args) -> None: # noqa: D107
        super().__init__(*args)
        self.status = "failed"
        self.reason = "unknown"

class AdjustmentFailure(AdjustmentError):
    def __init__(self, *args) -> None: # noqa: D107
        super().__init__(*args)
        self.reason = "adjust-failed"

class AdjustmentRejection(AdjustmentError):
    def __init__(self, *args) -> None: # noqa: D107
        super().__init__(*args)
        self.status = "rejected"

    # TODO: rejected class method?
