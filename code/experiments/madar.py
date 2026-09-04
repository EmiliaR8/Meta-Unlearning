"""MADAR: experience replay + knowledge distillation + synaptic intelligence.

The continual baseline everything else is measured against. Task 0 trains
normally; each later task runs a fixed iteration budget mixing current-task
batches with replay batches drawn from a bounded, diversity-aware buffer, with
distillation against the previous task's model and an SI penalty on parameters
that mattered to earlier tasks.
"""

from __future__ import annotations

from .base import ContinualExperiment, register


@register("madar")
class MadarExperiment(ContinualExperiment):
    """All behaviour is the shared continual shape; no method-specific step."""
