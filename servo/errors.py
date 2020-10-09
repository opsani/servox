from servo.events import CancelEventError, EventError


# TODO: Review and expand all the error classes
class ConnectorError(Exception):
    """Exception indicating that a connector failed"""

    def __init__(self, *args, status="failed", reason="unknown"):
        self.status = status
        self.reason = reason
        super().__init__(*args)


__all__ = (
    "CancelEventError",
    "ConnectorError",
    "EventError",
)
