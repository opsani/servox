from pydantic import BaseModel, validator
from datetime import datetime, timedelta
from typing import Callable, Iterable, Optional, Union, Type, Any
from .duration_str import timedelta_to_duration_str

class DurationProgress(BaseModel):
    duration: 'Duration'
    started_at: Optional[datetime]

    def __init__(self, duration: 'Duration', **kwargs) -> None:
        super().__init__(duration=duration, **kwargs)

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
        elapsed = timedelta_to_duration_str(self.elapsed())
        return f"{self.progress():.2f}% complete, {elapsed} elapsed - {str_to_annotate}"
