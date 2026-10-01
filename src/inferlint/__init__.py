"""Measures what your inference server actually did, and flags what it didn't tell you."""

from .result import CheckResult, Status, TripwireFailed

__version__ = "0.3.0"

__all__ = ["CheckResult", "Status", "TripwireFailed", "__version__"]
