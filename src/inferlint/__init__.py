"""Things your inference server doesn't tell you, turned into checks."""

from .result import CheckResult, Status, TripwireFailed

__version__ = "0.1.0.dev0"

__all__ = ["CheckResult", "Status", "TripwireFailed", "__version__"]
