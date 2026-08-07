from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int
    total: int | None
    status: str
    path: str = ""
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]