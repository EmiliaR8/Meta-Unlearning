"""Experiment registry.

Adding a method: write a module here with an @register("<name>") subclass of
Experiment (or ContinualExperiment) and import it below. Nothing else changes --
the runner, the logging schema and the aggregator are method-agnostic.
"""

from __future__ import annotations

from .base import Experiment, ContinualExperiment, available, build_experiment, register
from . import (naive, joint, madar, madar_unlearn,      # noqa: F401
               er_only, si_only, malcl, agem)           # noqa: F401

__all__ = ["Experiment", "ContinualExperiment", "available", "build_experiment",
           "register"]
