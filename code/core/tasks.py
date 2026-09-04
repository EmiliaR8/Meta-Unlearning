"""Task setup: which families arrive when.

A task setup is written "<task0>+<step>x<n_increments>", e.g. 50+5x10 for the
EMBER protocol (50 base families, then 5 new families for 10 tasks, 100 total
across 11 tasks). Presets name the ones in use; any conforming string is parsed,
so a new schedule needs no code change.

Class ids follow PRESENTATION order -- class 0..task0-1 arrive in task 0, the
next `step` in task 1, and so on. Evaluation masks logits to [:active_count],
which is only meaningful under that convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPEC = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)

PRESETS = {
    "ember":      "50+5x10",   # EMBER 2018 / 2024 mainline, 100 families
    "ember_t20":  "50+5x20",   # sequence-length study, 150 families
    "lamda":      "30+5x10",   # LAMDA: only 80 families clear the >=200 bar
}


@dataclass(frozen=True)
class TaskSchedule:
    task0: int
    step: int
    n_increments: int

    @property
    def n_tasks(self) -> int:
        return self.n_increments + 1

    @property
    def n_classes(self) -> int:
        return self.task0 + self.step * self.n_increments

    @property
    def spec(self) -> str:
        return f"{self.task0}+{self.step}x{self.n_increments}"

    def classes_for(self, tid: int) -> list[int]:
        """Classes introduced BY task `tid` (task 0 introduces the base set)."""
        self._check(tid)
        if tid == 0:
            return list(range(self.task0))
        lo = self.task0 + (tid - 1) * self.step
        return list(range(lo, lo + self.step))

    def seen_classes(self, tid: int) -> list[int]:
        """Every class introduced up to and including task `tid`."""
        return list(range(self.active_count(tid)))

    def active_count(self, tid: int) -> int:
        """Width of the active logit prefix after task `tid`."""
        self._check(tid)
        return self.task0 + tid * self.step

    def prev_active_count(self, tid: int) -> int:
        return self.task0 if tid == 0 else self.active_count(tid - 1)

    def _check(self, tid: int) -> None:
        if not 0 <= tid < self.n_tasks:
            raise IndexError(f"task {tid} outside 0..{self.n_tasks - 1} for {self.spec}")

    def as_dict(self) -> dict:
        return {"spec": self.spec, "task0_classes": self.task0,
                "step_classes": self.step, "n_increments": self.n_increments,
                "n_tasks": self.n_tasks, "n_classes": self.n_classes}


def build_schedule(spec: str) -> TaskSchedule:
    """Accepts a preset name or a '<task0>+<step>x<n>' string."""
    raw = PRESETS.get(str(spec).lower(), str(spec))
    m = _SPEC.match(raw)
    if not m:
        raise ValueError(
            f"task setup {spec!r} not understood. Use a preset "
            f"({', '.join(sorted(PRESETS))}) or '<task0>+<step>x<n>', e.g. '50+5x10'.")
    task0, step, n_inc = (int(g) for g in m.groups())
    if task0 <= 0 or step <= 0 or n_inc <= 0:
        raise ValueError(f"task setup {spec!r}: all three components must be positive")
    return TaskSchedule(task0=task0, step=step, n_increments=n_inc)
