"""
progress.py
-----------
Stage-based progress tracking. Every backend step has a NAME and a weight; the tracker converts
"stage X is 40 % done" into one overall percentage and tells the UI which stages are finished,
running or still waiting.
"""
from typing import Callable, List, Optional, Tuple


class ProgressTracker:
    """Tracks weighted stages and notifies `on_update(tracker)` after every change."""

    def __init__(self, stages: List[Tuple[str, float]],
                 on_update: Optional[Callable[["ProgressTracker"], None]] = None):
        total = sum(w for _, w in stages) or 1.0
        self.stages = [(name, w / total) for name, w in stages]   # weights normalised to sum to 1
        self.on_update = on_update
        self.current = stages[0][0] if stages else ""
        self.detail = ""
        self.overall = 0.0
        self._fraction = {name: 0.0 for name, _ in stages}

    @property
    def done(self) -> List[str]:
        """Names of the stages that are 100 % complete."""
        return [name for name, _ in self.stages if self._fraction[name] >= 1.0]

    def update(self, stage: str, fraction: float, detail: str = "") -> None:
        """Set `stage` to `fraction` (0-1, never moves backwards). Unknown stage names are ignored."""
        if stage not in self._fraction:
            return
        self._fraction[stage] = max(self._fraction[stage], min(max(fraction, 0.0), 1.0))
        self.current, self.detail = stage, detail
        self.overall = sum(w * self._fraction[name] for name, w in self.stages)
        if self.on_update:
            self.on_update(self)

    def finish(self, stage: str, detail: str = "") -> None:
        """Mark one stage as complete."""
        self.update(stage, 1.0, detail)

    def complete(self) -> None:
        """Mark every stage as complete (100 %)."""
        for name, _ in self.stages:
            self._fraction[name] = 1.0
        self.overall, self.detail = 1.0, "done"
        if self.on_update:
            self.on_update(self)


def tick(progress: Optional[ProgressTracker], stage: str, fraction: float, detail: str = "") -> None:
    """Null-safe helper so library code can report progress without checking for None."""
    if progress is not None:
        progress.update(stage, fraction, detail)
