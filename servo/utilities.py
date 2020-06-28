import durationpy
from datetime import datetime, timedelta
from typing import Union, Optional
from pydantic import BaseModel, validator

class DurationProgress(BaseModel):
    duration: Union[str, timedelta]
    started_at: Optional[datetime]
    
    def __init__(self, duration: Union[str, timedelta], **kwargs) -> None:
        super().__init__(duration=duration, **kwargs)

    @validator('duration', pre=True)
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
